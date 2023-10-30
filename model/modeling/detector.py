import torch
import torch.nn as nn
import numpy as np
from mmdet.models.builder import build_detector as mmdet_builder
from mmcv import Config


def build_detector(cfg):
    model = DetectorWrapper(Config.fromfile(cfg.MODEL.DETECTOR.CONFIG_FILE))
    if cfg.MODEL.DETECTOR.WEIGHT_FIX:
        for param in model.parameters():
            param.requires_grad = False

    if cfg.MODEL.DETECTOR.PRETRAINED_MODEL:
        print(f'Pretrained detector was loaded from {cfg.MODEL.DETECTOR.PRETRAINED_MODEL}')
        model.detector.load_state_dict(torch.load(cfg.MODEL.DETECTOR.PRETRAINED_MODEL)['state_dict'])

    return model

class DetectorWrapper(nn.Module):
    def __init__(self, mmdet_cfg):
        super(DetectorWrapper, self).__init__()

        model_cfg = mmdet_cfg.model
        train_cfg = mmdet_cfg.get('train_cfg')
        test_cfg = mmdet_cfg.get('test_cfg')

        self.detector = mmdet_builder(model_cfg, train_cfg=train_cfg, test_cfg=test_cfg)

    def forward(self, x, targets=None):
        imgs = []
        img_metas = []
        batch, channel, height, width = x.shape

        if targets:
            for i in range(batch):
                img_metas.append({
                    'img_shape': (height, width, channel),
                    'ori_shape': (height, width, channel),
                    'pad_shape': (height, width, channel),
                    'scale_factor': np.array([1.0 for _ in range(batch)]),
                    'flip': False,
                    'flip_direction': None,
                })


            boxes, labels = targets
            boxes, labels = self._foramt_gts(boxes, labels)
            losses = self.detector(img=x, img_metas=img_metas, gt_bboxes=boxes, gt_labels=labels, return_loss=True)
            loss, _ = self.detector._parse_losses(losses)

            return loss

        else:
            for i in range(batch):
                imgs.append(x[i:i+1])
                img_metas.append([{
                    'img_shape': (height, width, channel),
                    'ori_shape': (height, width, channel),
                    'pad_shape': (height, width, channel),
                    'scale_factor': np.array([1.0 for _ in range(batch)]),
                    'flip': False,
                    'flip_direction': None,
                    'border': [0., 1024., 0., 2048.], # metas for centernet
                    'batch_input_shape': (1024, 2048), # metas for centernet
                    # 'border': [0., 512., 0., 512.], # metas for centernet
                    # 'batch_input_shape': (512, 512) # metas for centernet
                }])
            return self.detector(img=imgs, img_metas=img_metas, return_loss=False)

    def _foramt_gts(self, boxes, labels):
        boxes, labels = [torch.tensor(box).to('cuda') for box in boxes.tolist()], [torch.tensor(label).to('cuda') for label in labels.tolist()]
        for i in range(len(boxes)):
            keep = (labels[i] != 0)
            boxes[i] = boxes[i][keep]
            labels[i] = labels[i][keep]

        return boxes, labels