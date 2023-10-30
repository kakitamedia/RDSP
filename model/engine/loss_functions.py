import torch.nn as nn

def get_loss_fn(option):
    if option == 'L1':
        return nn.L1Loss()
    elif option == 'BCE':
        return nn.BCELoss()
    else:
        raise NotImplementedError(f'{option} loss function is not supported.')