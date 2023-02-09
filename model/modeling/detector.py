import torch.nn as nn
import numpy as np
from mmcv import Config
from mmdet.models.builder import build_detector as mmdet_builder

def build_detector(cfg):
    model = DetectorWrapper(Config.fromfile(cfg.MODEL.DETECTOR.CONFIG_FILE))
    if cfg.MODEL.DETECTOR.WEIGHT_FIX:
        for param in model.parameters():
            param.requires_grad = False

    return model

class DetectorWrapper(nn.Module):
    def __init__(self, mmdet_cfg):
        super(DetectorWrapper, self).__init__()

        model_cfg = mmdet_cfg.model
        train_cfg = mmdet_cfg.get('train_cfg')
        test_cfg = mmdet_cfg.get('test_cfg')

        self.detector = mmdet_builder(model_cfg, train_cfg=train_cfg, test_cfg=test_cfg)

    def forward(self, x, return_loss=False):
        imgs = []
        img_metas = []
        batch, channel, height, width = x.shape

        for i in range(batch):
            imgs.append(x[i:i+1])
            img_metas.append([{
                'img_shape': (height, width, channel),
                'ori_shape': (height, width, channel),
                'pad_shape': (height, width, channel),
                'scale_factor': np.array([1.0 for _ in range(batch)]),
                'flip': False,
                'flip_direction': None,
                # 'border': [0., 1024., 0., 2048.], # metas for centernet
                # 'batch_input_shape': (1024, 2048) # metas for centernet
                'border': [0., 512., 0., 512.], # metas for centernet
                'batch_input_shape': (512, 512) # metas for centernet
            }])

        return self.detector(img=imgs, img_metas=img_metas, return_loss=return_loss)