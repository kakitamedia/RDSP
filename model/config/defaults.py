from yacs.config import CfgNode as CN

_C = CN()

_C.SOLVER = CN()
_C.SOLVER.LR = 1e-4
_C.SOLVER.BATCH_SIZE = 6
_C.SOLVER.MAX_ITER = 500000
_C.SOLVER.LR_DECAY = [300000, 450000]
_C.SOLVER.SYNC_BATCHNORM = True
_C.SOLVER.MASK_LOSS_FN = 'BCE'
_C.SOLVER.SR_LOSS_FN = 'L1'
_C.SOLVER.DET_LOSS_WEIGHT = 1.
_C.SOLVER.MASK_LOSS_WEIGHT = 1.
_C.SOLVER.SR_LOSS_WEIGHT = 1.
_C.SOLVER.BOX_THRESHOLD = 0.45
_C.SOLVER.OPTIMIZER = 'Adam'
_C.SOLVER.GRAD_CLIP = 0.

_C.MODEL = CN()
_C.MODEL.ACTIVATION = 'relu'
_C.MODEL.NORMALIZATION = 'batch'
_C.MODEL.BIAS = False

_C.MODEL.SR = CN()
_C.MODEL.SR.FLAG = True
_C.MODEL.SR.ARCITECTURE = 'DDBPN' # 'DBPN", 'DDBPN', 'ESRT', 'IMDN'
_C.MODEL.SR.FACTOR = 4
_C.MODEL.SR.FEAT = 64
_C.MODEL.SR.PRETRAINED_MODEL = 'weights/dbpn_stage4_x4.pth'
_C.MODEL.SR.WEIGHT_FIX = False
_C.MODEL.SR.INTERP_METHOD = 'bilinear'

_C.MODEL.MASK = CN()
_C.MODEL.MASK.FLAG = True
_C.MODEL.MASK.ARCITECTURE = 'UNet'
_C.MODEL.MASK.FEAT = 64
_C.MODEL.MASK.WEIGHT_FIX = False
_C.MODEL.MASK.PRETRAINED_MODEL = 'weights/mask_x4.pth'

_C.MODEL.MASK.CONTEXT = CN()
_C.MODEL.MASK.CONTEXT.FLAG = True
_C.MODEL.MASK.CONTEXT.NUM_SCALES = 4
_C.MODEL.MASK.CONTEXT.NUM_CONVS = 3
_C.MODEL.MASK.CONTEXT.NUM_FCS = 3
_C.MODEL.MASK.CONTEXT.MODE = 'add' # 'add' or 'mul'

_C.MODEL.MASK.COOD = CN()
_C.MODEL.MASK.COOD.FLAG = True
_C.MODEL.MASK.COOD.NUM_LAYERS = 1
_C.MODEL.MASK.COOD.MODE = 'add' # 'add' or 'mul'


_C.MODEL.DETECTOR = CN()
_C.MODEL.DETECTOR.CONFIG_FILE = 'model/modeling/detector_configs/ssd512.py'
_C.MODEL.DETECTOR.WEIGHT_FIX = False
_C.MODEL.DETECTOR.MAX_OBJECTS = 128
_C.MODEL.DETECTOR.PRETRAINED_MODEL = 'weights/ssd512_coco_20210803_022849-0a47a1ca.pth'

_C.DATA = CN()
_C.DATA.INPUT_MEAN = (123.675, 116.28, 103.53)
_C.DATA.INPUT_STD = (58.395, 57.12, 57.375)
_C.DATA.INPUT_SIZE = (512, 512)

_C.DATA.TRAIN = CN()
_C.DATA.TRAIN.IMAGE_DIR = 'datasets/CityScapes/leftImg8bit'
_C.DATA.TRAIN.ANN_FILE = 'datasets/CityScapes/coco_format/train.json'
_C.DATA.TRAIN.MASK_DIR = 'datasets/CityScapes/mask_gts'

_C.DATA.TEST = CN()
_C.DATA.TEST.IMAGE_DIR = 'datasets/CityScapes/leftImg8bit'
_C.DATA.TEST.ANN_FILE = 'datasets/CityScapes/coco_format/val.json'
_C.DATA.TEST.MASK_DIR = 'datasets/CityScapes/mask_gts'

_C.OUTPUT_DIR = ''
_C.MIXED_PRECISION = False
_C.SEED = 123