import os
import sys
import copy
import argparse
import platform
import paddle

from paddle import inference
# from src.utils.pptable_parser_utils.post_process import *
# from src.utils.pptable_parser_utils.operators import *

from pptable_parser_utils.post_process import *
from pptable_parser_utils.operators import *

def str2bool(v):
    return v.lower() in ("true", "yes", "t", "y", "1")

def str2int_tuple(v):
    return tuple([int(i.strip()) for i in v.split(",")])

def init_args():
    parser = argparse.ArgumentParser()
    # params for prediction engine
    parser.add_argument("--use_gpu", type=str2bool, default=False)
    parser.add_argument("--use_xpu", type=str2bool, default=False)
    parser.add_argument("--use_npu", type=str2bool, default=False)
    parser.add_argument("--ir_optim", type=str2bool, default=True)
    parser.add_argument("--use_tensorrt", type=str2bool, default=False)
    parser.add_argument("--min_subgraph_size", type=int, default=15)
    parser.add_argument("--precision", type=str, default="fp32")
    parser.add_argument("--gpu_mem", type=int, default=500)
    parser.add_argument("--gpu_id", type=int, default=0)

    # params for text detector
    parser.add_argument("--page_num", type=int, default=0)
    parser.add_argument("--det_algorithm", type=str, default='DB')
    parser.add_argument("--det_model_dir", type=str, default="data/table/inference/ch_PP-OCRv3_det_infer")
    parser.add_argument("--det_limit_side_len", type=float, default=960)
    parser.add_argument("--det_limit_type", type=str, default='max')
    parser.add_argument("--det_box_type", type=str, default='quad')

    # DB parmas
    parser.add_argument("--det_db_thresh", type=float, default=0.3)
    parser.add_argument("--det_db_box_thresh", type=float, default=0.6)
    parser.add_argument("--det_db_unclip_ratio", type=float, default=1.5)
    parser.add_argument("--max_batch_size", type=int, default=10)
    parser.add_argument("--use_dilation", type=str2bool, default=False)
    parser.add_argument("--det_db_score_mode", type=str, default="fast")

    # EAST parmas
    parser.add_argument("--det_east_score_thresh", type=float, default=0.8)
    parser.add_argument("--det_east_cover_thresh", type=float, default=0.1)
    parser.add_argument("--det_east_nms_thresh", type=float, default=0.2)

    # SAST parmas
    parser.add_argument("--det_sast_score_thresh", type=float, default=0.5)
    parser.add_argument("--det_sast_nms_thresh", type=float, default=0.2)

    # PSE parmas
    parser.add_argument("--det_pse_thresh", type=float, default=0)
    parser.add_argument("--det_pse_box_thresh", type=float, default=0.85)
    parser.add_argument("--det_pse_min_area", type=float, default=16)
    parser.add_argument("--det_pse_scale", type=int, default=1)

    # FCE parmas
    parser.add_argument("--scales", type=list, default=[8, 16, 32])
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--fourier_degree", type=int, default=5)

    # params for text recognizer
    parser.add_argument("--rec_algorithm", type=str, default='SVTR_LCNet')
    parser.add_argument("--rec_model_dir", type=str, default="data/table/inference/ch_PP-OCRv3_rec_infer")
    parser.add_argument("--rec_image_inverse", type=str2bool, default=True)
    parser.add_argument("--rec_image_shape", type=str, default="3, 48, 320")
    parser.add_argument("--rec_batch_num", type=int, default=6)
    parser.add_argument("--max_text_length", type=int, default=25)
    parser.add_argument(
        "--rec_char_dict_path",
        type=str,
        default="data/table/ppocr_keys_v1.txt")
    parser.add_argument("--use_space_char", type=str2bool, default=True)
    parser.add_argument(
        "--vis_font_path", type=str, default="data/fonts/simfang.ttf")
    parser.add_argument("--drop_score", type=float, default=0.5)

    # params for e2e
    parser.add_argument("--e2e_algorithm", type=str, default='PGNet')
    parser.add_argument("--e2e_model_dir", type=str)
    parser.add_argument("--e2e_limit_side_len", type=float, default=768)
    parser.add_argument("--e2e_limit_type", type=str, default='max')

    # PGNet parmas
    parser.add_argument("--e2e_pgnet_score_thresh", type=float, default=0.5)
    parser.add_argument(
        "--e2e_char_dict_path", type=str, default="data/table/ic15_dict.txt")
    parser.add_argument("--e2e_pgnet_valid_set", type=str, default='totaltext')
    parser.add_argument("--e2e_pgnet_mode", type=str, default='fast')

    parser.add_argument("--enable_mkldnn", type=str2bool, default=False)
    parser.add_argument("--cpu_threads", type=int, default=10)
    parser.add_argument("--use_pdserving", type=str2bool, default=False)
    parser.add_argument("--warmup", type=str2bool, default=False)

    # multi-process
    parser.add_argument("--use_mp", type=str2bool, default=False)
    parser.add_argument("--total_process_num", type=int, default=1)
    parser.add_argument("--process_id", type=int, default=0)

    parser.add_argument("--benchmark", type=str2bool, default=False)
    parser.add_argument("--use_onnx", type=str2bool, default=False)

    return parser


def parse_args():
    parser = init_args()
    return parser.parse_args()


def create_operators(op_param_list, global_config=None):
    """
    create operators based on the config

    Args:
        params(list): a dict list, used to create some operators
    """
    assert isinstance(op_param_list, list), ('operator config should be a list')
    ops = []
    for operator in op_param_list:
        assert isinstance(operator,
                          dict) and len(operator) == 1, "yaml format error"
        op_name = list(operator)[0]
        param = {} if operator[op_name] is None else operator[op_name]
        if global_config is not None:
            param.update(global_config)
        op = eval(op_name)(**param)
        ops.append(op)

    return ops


def create_predictor(args, mode, logger):
    if mode == "det":
        model_dir = args.det_model_dir
    elif mode == 'rec':
        model_dir = args.rec_model_dir
    elif mode == 'table':
        model_dir = args.table_model_dir     

    if model_dir is None:
        logger.info("not find {} model file path {}".format(mode, model_dir))
        sys.exit(0)

    model_file_path = '{}/{}.pdmodel'.format(model_dir, "inference")
    params_file_path = '{}/{}.pdiparams'.format(model_dir, "inference")
    if not os.path.exists(model_file_path):
        raise ValueError(
            "not find model.pdmodel or inference.pdmodel in {}".format(
                model_dir))
    if not os.path.exists(params_file_path):
        raise ValueError(
            "not find model.pdiparams or inference.pdiparams in {}".format(
                model_dir))

    config = inference.Config(model_file_path, params_file_path)

    if hasattr(args, 'precision'):
        if args.precision == "fp16" and args.use_tensorrt:
            precision = inference.PrecisionType.Half
        elif args.precision == "int8":
            precision = inference.PrecisionType.Int8
        else:
            precision = inference.PrecisionType.Float32
    else:
        precision = inference.PrecisionType.Float32

    if args.use_gpu:
        gpu_id = get_infer_gpuid()
        if gpu_id is None:
            logger.warning(
                "GPU is not found in current device by nvidia-smi. Please check your device or ignore it if run on jetson."
            )
        config.enable_use_gpu(args.gpu_mem, args.gpu_id)
        if args.use_tensorrt:
            config.enable_tensorrt_engine(
                workspace_size=1 << 30,
                precision_mode=precision,
                max_batch_size=args.max_batch_size,
                min_subgraph_size=args.
                min_subgraph_size,  # skip the minmum trt subgraph
                use_calib_mode=False)

            # collect shape
            trt_shape_f = os.path.join(model_dir,
                                        f"{mode}_trt_dynamic_shape.txt")

            if not os.path.exists(trt_shape_f):
                config.collect_shape_range_info(trt_shape_f)
                logger.info(
                    f"collect dynamic shape info into : {trt_shape_f}")
            try:
                config.enable_tuned_tensorrt_dynamic_shape(trt_shape_f,
                                                            True)
            except Exception as E:
                logger.info(E)
                logger.info("Please keep your paddlepaddle-gpu >= 2.3.0!")

    elif args.use_npu:
        config.enable_custom_device("npu")
    elif args.use_xpu:
        config.enable_xpu(10 * 1024 * 1024)
    else:
        config.disable_gpu()
        if args.enable_mkldnn:
            # cache 10 different shapes for mkldnn to avoid memory leak
            config.set_mkldnn_cache_capacity(10)
            config.enable_mkldnn()
            if args.precision == "fp16":
                config.enable_mkldnn_bfloat16()
            if hasattr(args, "cpu_threads"):
                config.set_cpu_math_library_num_threads(args.cpu_threads)
            else:
                # default cpu threads as 10
                config.set_cpu_math_library_num_threads(10)
    # enable memory optim
    config.enable_memory_optim()
    config.disable_glog_info()
    config.delete_pass("conv_transpose_eltwiseadd_bn_fuse_pass")
    config.delete_pass("matmul_transpose_reshape_fuse_pass")
    if mode == 're':
        config.delete_pass("simplify_with_basic_ops_pass")
    if mode == 'table':
        config.delete_pass("fc_fuse_pass")  # not supported for table
    config.switch_use_feed_fetch_ops(False)
    config.switch_ir_optim(True)

    # create predictor
    predictor = inference.create_predictor(config)
    input_names = predictor.get_input_names()
    if mode in ['ser', 're']:
        input_tensor = []
        for name in input_names:
            input_tensor.append(predictor.get_input_handle(name))
    else:
        for name in input_names:
            input_tensor = predictor.get_input_handle(name)
    output_tensors = get_output_tensors(args, mode, predictor)

    return predictor, input_tensor, output_tensors, config
    

def get_output_tensors(args, mode, predictor):
    output_names = predictor.get_output_names()
    output_tensors = []
    if mode == "rec" and args.rec_algorithm in [
            "CRNN", "SVTR_LCNet", "SVTR_HGNet"
    ]:
        output_name = 'softmax_0.tmp_0'
        if output_name in output_names:
            return [predictor.get_output_handle(output_name)]
        else:
            for output_name in output_names:
                output_tensor = predictor.get_output_handle(output_name)
                output_tensors.append(output_tensor)
    else:
        for output_name in output_names:
            output_tensor = predictor.get_output_handle(output_name)
            output_tensors.append(output_tensor)

    return output_tensors


def get_infer_gpuid():
    sysstr = platform.system()
    if sysstr == "Windows":
        return 0

    if not paddle.device.is_compiled_with_rocm:
        cmd = "env | grep CUDA_VISIBLE_DEVICES"
    else:
        cmd = "env | grep HIP_VISIBLE_DEVICES"
    env_cuda = os.popen(cmd).readlines()
    if len(env_cuda) == 0:
        return 0
    else:
        gpu_id = env_cuda[0].strip().split("=")[1]
        return int(gpu_id[0])


def build_post_process(config, global_config=None):
    support_dict = [
        'DBPostProcess', 'EASTPostProcess', 'SASTPostProcess', 'FCEPostProcess',
        'CTCLabelDecode', 'AttnLabelDecode', 'ClsPostProcess', 'SRNLabelDecode',
        'PGPostProcess', 'DistillationCTCLabelDecode', 'TableLabelDecode',
        'DistillationDBPostProcess', 'NRTRLabelDecode', 'SARLabelDecode',
        'SEEDLabelDecode', 'VQASerTokenLayoutLMPostProcess',
        'VQAReTokenLayoutLMPostProcess', 'PRENLabelDecode',
        'DistillationSARLabelDecode', 'ViTSTRLabelDecode', 'ABINetLabelDecode',
        'TableMasterLabelDecode', 'SPINLabelDecode',
        'DistillationSerPostProcess', 'DistillationRePostProcess',
        'VLLabelDecode', 'PicoDetPostProcess', 'CTPostProcess',
        'RFLLabelDecode', 'DRRGPostprocess', 'CANLabelDecode',
        'SATRNLabelDecode'
    ]
    
    config = copy.deepcopy(config)
    module_name = config.pop('name')
    if module_name == "None":
        return
    if global_config is not None:
        config.update(global_config)
    assert module_name in support_dict, Exception(
        'post process only support {}'.format(support_dict))
    module_class = eval(module_name)(**config)
    
    return module_class


def transform(data, ops=None):
    """ transform """
    if ops is None:
        ops = []
    for op in ops:
        data = op(data)
        if data is None:
            return None
        
    return data