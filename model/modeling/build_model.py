import torch.nn as nn

from .sr import build_sr_model
from .mask import build_mask_model
from .detector import build_detector

class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()

        self.sr_model = build_sr_model(cfg)
        self.mask_modek = build_mask_model(cfg)
        self.det_model = build_detector(cfg)

    def forward(self, x, targets=None):
        x = {k:v.to('cuda', non_blocking=True) for k, v in x.items()}

        sr = self.sr_model(x['input_image'])

        mask = self.mask_model(x['input_image'], context=x['context_image'], cood=x['cood'])
        mask = self.up(mask)

        if targets is None:
            det = self.det_model(sr * mask, return_loss=False)

            results = {
                'sr': sr,
                'mask': mask,
                'det': det,
            }

            return results

        targets = {k:v.to('cuda', non_blocking=True) for k, v in targets.items()}
        loss, loss_dict = 0, {}

        loss, loss_dict = self.det_loss_fn(sr, mask, return_loss=True)
        loss, loss_dict = self.sr_loss_fn(sr, targets['sr'])
        loss, loss_dict = self.mask_loss_fn(mask, targets['mask'])

        return loss, loss_dict