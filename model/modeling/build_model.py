import numpy as np

import torch
import torch.nn as nn

from .sr import build_sr_model
from .mask import build_mask_model
from .detector import build_detector
from model.engine.loss_functions import get_loss_fn

import time

class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()

        self.sr_model = build_sr_model(cfg)
        self.mask_model = build_mask_model(cfg)
        self.det_model = build_detector(cfg)

        self.mask_loss_fn = get_loss_fn(cfg.SOLVER.MASK_LOSS_FN)
        self.sr_loss_fn = get_loss_fn(cfg.SOLVER.SR_LOSS_FN)

        self.det_loss_weight = cfg.SOLVER.DET_LOSS_WEIGHT
        self.mask_loss_weight = cfg.SOLVER.MASK_LOSS_WEIGHT
        self.sr_loss_weight = cfg.SOLVER.SR_LOSS_WEIGHT

        self.sr_flag = cfg.MODEL.SR.FLAG
        self.mask_flag = cfg.MODEL.MASK.FLAG

        self.up = nn.Upsample(scale_factor=cfg.MODEL.SR.FACTOR)


    def forward(self, x, targets=None):
        x = {k:v.to('cuda', non_blocking=True) for k, v in x.items()}

        sr = self.sr_model(x['detect_input'])

        mask = self.mask_model(x['mask_input'], context=x['context_image'], cood=x['cood'])

        if targets is None:
            mask = self.up(mask)
            det = self.det_model(sr*mask)

            return sr, mask, det

        loss, loss_dict = 0, {}

        det_loss = self.det_model(sr*mask, targets=targets['det'])
        loss += self.det_loss_weight * det_loss
        loss_dict['det_loss'] = det_loss.detach()

        if self.sr_flag:
            sr_loss = self.sr_loss_fn(sr, targets['sr'].to('cuda'))
            loss += self.sr_loss_weight * sr_loss
            loss_dict['sr_loss'] = sr_loss.detach()

        if self.mask_flag:
            mask_loss = self.mask_loss_fn(mask, targets['mask'].to('cuda'))
            loss += self.mask_loss_weight * mask_loss
            loss_dict['mask_loss'] = mask_loss.detach()

        return loss, loss_dict