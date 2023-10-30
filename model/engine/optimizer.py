import torch

def build_optimizer(cfg, model):
    if cfg.SOLVER.OPTIMIZER == 'Adam':
        return torch.optim.Adam(filter(lambda p:p.requires_grad, model.parameters()), lr=cfg.SOLVER.LR)
    elif cfg.SOLVER.OPTIMIZER == 'SGD':
        return torch.optim.SGD(filter(lambda p:p.requires_grad, model.parameters()), lr=cfg.SOLVER.LR, momentum=0.9, weight_decay=5e-4)
    elif cfg.SOLVER.OPTIMIZER == 'AdamW':
        return torch.optim.AdamW(filter(lambda p:p.requires_grad, model.parameters()), lr=cfg.SOLVER.LR, weight_decay=1e-4)
