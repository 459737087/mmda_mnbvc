import time
import numpy as np

from pptable_parser_utils.logger import get_logger
from pptable_parser_utils.infer_utils import create_predictor, get_infer_gpuid, transform, \
                                                       build_post_process, create_operators
from pptable_parser_utils.post_process import *
from pptable_parser_utils.operators import *
logger = get_logger()


class TableStructurer(object):
    def __init__(self, args):
        self.args = args
        self.use_onnx = args.use_onnx
        pre_process_list = build_pre_process_list(args)
        if args.table_algorithm not in ['TableMaster']:
            postprocess_params = {
                'name': 'TableLabelDecode',
                "character_dict_path": args.table_char_dict_path,
                'merge_no_span_structure': args.merge_no_span_structure
            }
        else:
            postprocess_params = {
                'name': 'TableMasterLabelDecode',
                "character_dict_path": args.table_char_dict_path,
                'box_shape': 'pad',
                'merge_no_span_structure': args.merge_no_span_structure
            }
        self.preprocess_op = create_operators(pre_process_list)
        self.postprocess_op = build_post_process(postprocess_params)
        self.predictor, self.input_tensor, self.output_tensors, self.config = \
            create_predictor(args, 'table', logger)

    def __call__(self, img):
        starttime = time.time()
        if self.args.benchmark:
            self.autolog.times.start()

        ori_im = img.copy()
        data = {'image': img}
        data = transform(data, self.preprocess_op)
        img = data[0]
        if img is None:
            return None, 0
        img = np.expand_dims(img, axis=0)
        img = img.copy()
        self.input_tensor.copy_from_cpu(img)
        self.predictor.run()
        outputs = []
        for output_tensor in self.output_tensors:
            output = output_tensor.copy_to_cpu()
            outputs.append(output)

        preds = {}
        preds['structure_probs'] = outputs[1]
        preds['loc_preds'] = outputs[0]

        shape_list = np.expand_dims(data[-1], axis=0)
        post_result = self.postprocess_op(preds, [shape_list])

        structure_str_list = post_result['structure_batch_list'][0]
        bbox_list = post_result['bbox_batch_list'][0]
        structure_str_list = structure_str_list[0]
        structure_str_list = [
            '<html>', '<body>', '<table>'
        ] + structure_str_list + ['</table>', '</body>', '</html>']
        elapse = time.time() - starttime
        if self.args.benchmark:
            self.autolog.times.end(stamp=True)

        return (structure_str_list, bbox_list), elapse
    

def build_pre_process_list(args):
    resize_op = {'ResizeTableImage': {'max_len': args.table_max_len, }}
    pad_op = {
        'PaddingTableImage': {
            'size': [args.table_max_len, args.table_max_len]
        }
    }
    normalize_op = {
        'NormalizeImage': {
            'std': [0.229, 0.224, 0.225] if
            args.table_algorithm not in ['TableMaster'] else [0.5, 0.5, 0.5],
            'mean': [0.485, 0.456, 0.406] if
            args.table_algorithm not in ['TableMaster'] else [0.5, 0.5, 0.5],
            'scale': '1./255.',
            'order': 'hwc'
        }
    }
    to_chw_op = {'ToCHWImage': None}
    keep_keys_op = {'KeepKeys': {'keep_keys': ['image', 'shape']}}
    if args.table_algorithm not in ['TableMaster']:
        pre_process_list = [
            resize_op, normalize_op, pad_op, to_chw_op, keep_keys_op
        ]
    else:
        pre_process_list = [
            resize_op, pad_op, normalize_op, to_chw_op, keep_keys_op
        ]
        
    return pre_process_list