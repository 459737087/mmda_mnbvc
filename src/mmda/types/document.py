import itertools
import logging
import warnings
import os
import base64
import io
from typing import Dict, Iterable, List, Optional
from PIL import Image, ImageDraw, ImageFont

from mmda.types.annotation import Annotation, BoxGroup, SpanGroup
from mmda.types.image import PILImage
from mmda.types.indexers import Indexer, SpanGroupIndexer
from mmda.types.metadata import Metadata
from mmda.types.names import ImagesField, MetadataField, SymbolsField
from mmda.utils.tools import MergeSpans, allocate_overlapping_tokens_for_box, box_groups_to_span_groups


class Document:
    SPECIAL_FIELDS = [SymbolsField, ImagesField, MetadataField]
    UNALLOWED_FIELD_NAMES = ["fields"]

    def __init__(self, symbols: str, metadata: Optional[Metadata] = None):
        self.symbols = symbols
        self.images = []
        self.__fields = []
        self.__indexers: Dict[str, Indexer] = {}
        self.metadata = metadata if metadata else Metadata()

    @property
    def fields(self) -> List[str]:
        return self.__fields

    # TODO: extend implementation to support DocBoxGroup
    def find_overlapping(self, query: Annotation, field_name: str) -> List[Annotation]:
        if not isinstance(query, SpanGroup):
            raise NotImplementedError(
                f"Currently only supports query of type SpanGroup"
            )
        return self.__indexers[field_name].find(query=query)

    def add_metadata(self, **kwargs):
        """Copy kwargs into the document metadata"""
        for k, value in kwargs.items():
            self.metadata.set(k, value)

    def annotate(
            self, is_overwrite: bool = False, **kwargs: Iterable[Annotation]
    ) -> None:
        """Annotate the fields for document symbols (correlating the annotations with the
        symbols) and store them into the papers.
        """
        # 1) check validity of field names
        for field_name in kwargs.keys():
            assert (
                    field_name not in self.SPECIAL_FIELDS
            ), f"The field_name {field_name} should not be in {self.SPECIAL_FIELDS}."

            if field_name in self.fields:
                # already existing field, check if ok overriding
                if not is_overwrite:
                    raise AssertionError(
                        f"This field name {field_name} already exists. To override, set `is_overwrite=True`"
                    )
            elif field_name in dir(self):
                # not an existing field, but a reserved class method name
                raise AssertionError(
                    f"The field_name {field_name} should not conflict with existing class properties"
                )

        # Kyle's preserved comment:
        # Is it worth deepcopying the annotations? Safer, but adds ~10%
        # overhead on large documents.

        # 2) register fields into Document
        for field_name, annotations in kwargs.items():
            if len(annotations) == 0:
                warnings.warn(f"The annotations is empty for the field {field_name}")
                setattr(self, field_name, [])
                self.__fields.append(field_name)
                continue

            annotation_types = {type(a) for a in annotations}
            assert (
                    len(annotation_types) == 1
            ), f"Annotations in field_name {field_name} more than 1 type: {annotation_types}"
            annotation_type = annotation_types.pop()

            if annotation_type == SpanGroup:
                span_groups = self._annotate_span_group(
                    span_groups=annotations, field_name=field_name
                )
            elif annotation_type == BoxGroup:
                # TODO: not good. BoxGroups should be stored on their own, not auto-generating SpanGroups.
                span_groups = self._annotate_span_group(
                    span_groups=box_groups_to_span_groups(annotations, self), field_name=field_name
                )
            else:
                raise NotImplementedError(
                    f"Unsupported annotation type {annotation_type} for {field_name}"
                )

            # register fields
            setattr(self, field_name, span_groups)
            self.__fields.append(field_name)

    def remove(self, field_name: str):
        delattr(self, field_name)
        self.__fields = [f for f in self.__fields if f != field_name]
        del self.__indexers[field_name]

    def annotate_images(
            self, images: Iterable[PILImage], is_overwrite: bool = False
    ) -> None:
        if not is_overwrite and len(self.images) > 0:
            raise AssertionError(
                "This field name {Images} already exists. To override, set `is_overwrite=True`"
            )

        if len(images) == 0:
            raise AssertionError("No images were provided")

        image_types = {type(a) for a in images}
        assert len(image_types) == 1, f"Images contain more than 1 type: {image_types}"
        image_type = image_types.pop()

        if not issubclass(image_type, PILImage):
            raise NotImplementedError(
                f"Unsupported image type {image_type} for {ImagesField}"
            )

        self.images = images

    def _annotate_span_group(
            self, span_groups: List[SpanGroup], field_name: str
    ) -> List[SpanGroup]:
        """Annotate the Document using a bunch of span groups.
        It will associate the annotations with the document symbols.
        """
        assert all([isinstance(group, SpanGroup) for group in span_groups])

        # 1) add Document to each SpanGroup
        for span_group in span_groups:
            span_group.attach_doc(doc=self)

        # 2) Build fast overlap lookup index
        self.__indexers[field_name] = SpanGroupIndexer(span_groups)

        return span_groups

    #
    #   to & from JSON
    #

    def to_json(self, fields: Optional[List[str]] = None, with_images=False) -> Dict:
        """Returns a dictionary that's suitable for serialization

        Use `fields` to specify a subset of groups in the Document to include (e.g. 'sentences')
        If `with_images` is True, will also turn the Images into base64 strings.  Else, won't include them.

        Output format looks like
            {
                symbols: "...",
                field1: [...],
                field2: [...],
                metadata: {...}
            }
        """
        doc_dict = {SymbolsField: self.symbols, MetadataField: self.metadata.to_json()}
        if with_images:
            doc_dict[ImagesField] = [image.to_json() for image in self.images]

        # figure out which fields to serialize
        fields = (
            self.fields if fields is None else fields
        )  # use all fields unless overridden

        # add to doc dict
        for field in fields:
            doc_dict[field] = [
                doc_span_group.to_json() for doc_span_group in getattr(self, field)
            ]

        return doc_dict

    @classmethod
    def from_json(cls, doc_dict: Dict) -> "Document":
        # 1) instantiate basic Document
        symbols = doc_dict[SymbolsField]
        doc = cls(symbols=symbols, metadata=Metadata(**doc_dict.get(MetadataField, {})))

        if Metadata in doc_dict:
            doc.add_metadata(**doc_dict[Metadata])

        images_dict = doc_dict.get(ImagesField, None)
        if images_dict:
            doc.annotate_images(
                [PILImage.frombase64(image_str) for image_str in images_dict]
            )

        # 2) convert span group dicts to span gropus
        field_name_to_span_groups = {}
        for field_name, span_group_dicts in doc_dict.items():
            if field_name not in doc.SPECIAL_FIELDS:
                span_groups = [
                    SpanGroup.from_json(span_group_dict=span_group_dict)
                    for span_group_dict in span_group_dicts
                ]
                field_name_to_span_groups[field_name] = span_groups

        # 3) load annotations for each field
        doc.annotate(**field_name_to_span_groups)

        return doc
    
    def _group_layouts_and_tokens(self):
        """
        将JSON格式的layout和token按页面编号分组。
        方法调用对象的 `to_json` 方法，遍历layout，按照每个layout所在的页面编号分组存放在一个字典中。
        # 2) convert each annotate fields in doc images
        Returns:
            tuple: 返回一个元组，第一个元素是包含图片的JSON字典，第二个元素是按页面分组的layout字典和token字典。
        """
        grouped_data = {}
        json_dict = self.to_json(with_images=True)

        # 分组 layout
        for layout in json_dict["layout"]:
            page_number = layout["box_group"]["boxes"][0]["page"]
            if page_number not in grouped_data:
                grouped_data[page_number] = {"layouts": [], "tokens": []}
            grouped_data[page_number]["layouts"].append(layout)

        # 分组 tokens
        for token in json_dict["tokens"]:
            for span in token["spans"]:
                page_number = span["box"]["page"]
                if page_number not in grouped_data:
                    grouped_data[page_number] = {"layouts": [], "tokens": []}
                grouped_data[page_number]["tokens"].append(token)

        return json_dict, grouped_data
    
    def vis_annotate(self, output_dir='tmp', layout_font_size=20, token_font_size=8, 
                 font_path='data/fonts/Tahoma.ttf',
                 vis_token=False, vis_layout=True, layout_outline_color='blue', 
                 layout_outline_width=5, token_outline_color='red', token_outline_width=3,
                 token_text_color='white', token_bg_color='black'):
        """
        获取layout，然后在图片上绘制每个layout标注的边界框和类型标签、token标注的边界框和id。
        处理后的图片将保存在指定的输出目录中。

        Args:
            output_dir (str): 图像保存的目标文件夹，默认为 'tmp'。
            layout_font_size (int): 用于布局标签的字体大小，默认为 20。
            token_font_size (int): 用于标记标签的字体大小，默认为 8。
            font_path (str): 字体文件的路径，默认为 'data/fonts/Tahoma.ttf'。
            vis_token (bool): 是否可视化标记，即是否在图像上绘制标记的边界框和标签，默认为 False。
            vis_layout (bool): 是否可视化布局，即是否在图像上绘制布局的边界框和标签，默认为 True。
            layout_outline_color (str): 布局边界框的颜色，默认为 'blue'。
            layout_outline_width (int): 布局边界框的线宽，默认为 5。
            token_outline_color (str): 标记边界框的颜色，默认为 'red'。
            token_outline_width (int): 标记边界框的线宽，默认为 3。
            token_text_color (str): 标记文本的颜色，默认为 'white'。
            token_bg_color (str): 标记文本背景的颜色，默认为 'black'。
        Return:
            None
        """
        font_layout = ImageFont.truetype(font_path, layout_font_size)
        font_token = ImageFont.truetype(font_path, token_font_size)
        json_dict, grouped_data = self._group_layouts_and_tokens()
        for index, image_data in enumerate(json_dict['images']):
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            draw = ImageDraw.Draw(image)

            page_data = grouped_data.get(index, {})

            # 绘制layout（如果指定）
            if vis_layout:
                for layout in page_data.get("layouts", []):
                    box_group = layout['box_group']
                    for box in box_group['boxes']:
                        left = box['left'] * image.width
                        top = box['top'] * image.height
                        right = left + box['width'] * image.width
                        bottom = top + box['height'] * image.height

                        box_type = box_group['metadata']['type']

                        draw.rectangle([left, top, right, bottom], outline=layout_outline_color, width=layout_outline_width)
                        text_size = font_layout.getsize(box_type)
                        text_position = (left, top)
                        bg_rectangle_coords = (text_position[0], text_position[1], text_position[0] + text_size[0], text_position[1] + text_size[1])

                        draw.rectangle(bg_rectangle_coords, fill=token_bg_color)
                        draw.text(text_position, box_type, font=font_layout, fill=token_text_color)

            # 绘制token（如果指定）
            if vis_token:
                for token in page_data.get("tokens", []):
                    token_id = token["id"]
                    for span in token["spans"]:
                        box = span["box"]
                        left = box['left'] * image.width
                        top = box['top'] * image.height
                        right = left + box['width'] * image.width
                        bottom = top + box['height'] * image.height

                        draw.rectangle([left, top, right, bottom], outline=token_outline_color, width=token_outline_width)
                        
                        # 绘制token ID
                        text_size = font_token.getsize(str(token_id))
                        text_position = (left, top)  # 在边框下方显示ID
                        bg_rectangle_coords = (text_position[0], text_position[1], text_position[0] + text_size[0], text_position[1] + text_size[1])

                        draw.rectangle(bg_rectangle_coords, fill=token_bg_color) 
                        draw.text(text_position, str(token_id), font=font_token, fill=token_text_color)

            # 保存处理后的图片
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_path = os.path.join(output_dir, f'output_image_{index}.png')
            image.save(output_path)
