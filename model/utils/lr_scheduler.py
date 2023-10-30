from torch.optim.lr_scheduler import MultiStepLR

class WarmupMultiStepLR(MultiStepLR):
    def __init__(self, optimizer, init_lr, milestones, gamma=0.1, warmup_factor=1.0 / 3,
                 warmup_iters=500, last_epoch=-1):
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.init_lr = [init_lr]
        super().__init__(optimizer, milestones, gamma, last_epoch)

    def get_lr(self):
        if self.last_epoch <= self.warmup_iters:
            lr = self.init_lr
            alpha = self.last_epoch / self.warmup_iters
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            return [l * warmup_factor for l in lr]

        if self.last_epoch not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]

        return [group['lr'] * self.gamma ** self.milestones[self.last_epoch]
                for group in self.optimizer.param_groups]
