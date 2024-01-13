"""
 Use PaddlePaddle to extract tables from jpg images
 model: ch_PP-OCRv3_rec_infer
 detection_algorithm: DB
 rec_algorithm: SVTR_LCNet
 NOT USE TABLE MASTER
"""
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
print(parent_dir)
sys.path.insert(0, parent_dir)
import time
import copy
import numpy as np
import cv2

import pptable_parser_utils.predict_det as predict_det
import pptable_parser_utils.predict_rec as predict_rec
from pptable_parser_utils.utility import parse_args, sorted_boxes
from pptable_parser_utils.matcher_utils import TableMatch
from pptable_parser_utils.predict_structure import TableStructurer


def base64_to_cv2(base64_str):
    """
    Convert a Base64 encoded image to a cv2 (OpenCV) format.

    Args:
    - base64_str (str): The Base64 encoded string of the image.

    Returns:
    - image (ndarray): The decoded image in cv2 format.
    """
    img_bytes = base64.b64decode(base64_str)
    img_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    return img


class TablePredictor:
    def __init__(self, args=parse_args()):
        self.table_sys = TableSystem(args)
        
    
    def get_html_result(self, image_base64):
        # 尝试解析图像并获取HTML结果
        img = base64_to_cv2(image_base64)
        pred_res, _ = self.table_sys(img)
        pred_html = pred_res['html']

            
        return pred_html



def expand(pix, det_box, shape):
    x0, y0, x1, y1 = det_box
    h, w, c = shape
    tmp_x0 = x0 - pix
    tmp_x1 = x1 + pix
    tmp_y0 = y0 - pix
    tmp_y1 = y1 + pix
    x0_ = tmp_x0 if tmp_x0 >= 0 else 0
    x1_ = tmp_x1 if tmp_x1 <= w else w
    y0_ = tmp_y0 if tmp_y0 >= 0 else 0
    y1_ = tmp_y1 if tmp_y1 <= h else h

    return x0_, y0_, x1_, y1_


class TableSystem:
    def __init__(self, args, text_detector=None, text_recognizer=None):
        self.args = args
        benchmark_tmp = False
        if args.benchmark:
            benchmark_tmp = args.benchmark
            args.benchmark = False
        self.text_detector = predict_det.TextDetector(copy.deepcopy(
            args)) if text_detector is None else text_detector
        self.text_recognizer = predict_rec.TextRecognizer(copy.deepcopy(
            args)) if text_recognizer is None else text_recognizer
        if benchmark_tmp:
            args.benchmark = True
        self.table_structurer = TableStructurer(args)
        self.match = TableMatch(filter_ocr_result=True)

    def __call__(self, img, return_ocr_result_in_table=False):
        result = dict()
        time_dict = {'det': 0, 'rec': 0, 'table': 0, 'all': 0, 'match': 0}
        start = time.time()
        structure_res, elapse = self._structure(copy.deepcopy(img))

        dt_boxes, rec_res, det_elapse, rec_elapse = self._ocr(
            copy.deepcopy(img))
        time_dict['det'] = det_elapse
        time_dict['rec'] = rec_elapse

        tic = time.time()
        pred_html = self.match(structure_res, dt_boxes, rec_res)
        toc = time.time()
        time_dict['match'] = toc - tic
        result['html'] = pred_html
        end = time.time()
        time_dict['all'] = end - start
        
        return result, time_dict

    def _structure(self, img):
        structure_res, elapse = self.table_structurer(copy.deepcopy(img))
        return structure_res, elapse

    def _ocr(self, img):
        h, w = img.shape[:2]
        dt_boxes, det_elapse = self.text_detector(copy.deepcopy(img))
        dt_boxes = sorted_boxes(dt_boxes)

        r_boxes = []
        for box in dt_boxes:
            x_min = max(0, box[:, 0].min() - 1)
            x_max = min(w, box[:, 0].max() + 1)
            y_min = max(0, box[:, 1].min() - 1)
            y_max = min(h, box[:, 1].max() + 1)
            box = [x_min, y_min, x_max, y_max]
            r_boxes.append(box)
        dt_boxes = np.array(r_boxes)
        if dt_boxes is None:
            return None, None

        img_crop_list = []
        for i in range(len(dt_boxes)):
            det_box = dt_boxes[i]
            x0, y0, x1, y1 = expand(2, det_box, img.shape)
            text_rect = img[int(y0):int(y1), int(x0):int(x1), :]
            img_crop_list.append(text_rect)
        rec_res, rec_elapse = self.text_recognizer(img_crop_list)

        return dt_boxes, rec_res, det_elapse, rec_elapse
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--img_path", default="data/test/table_en.png", type=str, help="The path to the image file.")
    args = parser.parse_args()
    img_path = args.img_path
    table_predictor = TablePredictor()
    import base64
    def image_to_base64(image_path: str) -> str:
            """
            Convert an image to its base64 representation.
            Parameters:
                image_path (str): The path to the image file.
            Returns:
                str: The base64 representation of the image.
            """
            with open(image_path, "rb") as image_file:
                # Convert the binary data to base64 encoded string
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return encoded_string
    img = image_to_base64(img_path)
    res  = table_predictor.get_html_result(img)
    from pprint import pprint
    pprint(res)
    