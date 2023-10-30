import argparse
import torch
from model.config import cfg
from model.modeling.build_model import Model
from mmcv.cnn.utils.flops_counter import get_model_complexity_info

def parse_args():
    parser = argparse.ArgumentParser(description='RDSP training')
    parser.add_argument('--config_file', type=str, default='', metavar='FILE')

    return parser.parse_args()

def main():
    args = parse_args()

    if len(args.config_file) > 0:
        print('Configration file is loaded from {}'.format(args.config_file))
        cfg.merge_from_file(args.config_file)

    cfg.freeze()

    model = Model(cfg)
    get_model_complexity_info(model, (1, 3, 2048, 1024), input_constructor=input_constructor)

def input_constructor(shape):
    x = {}
    x['detect_input'] = torch.rand(*shape)
    x['mask_input'] = torch.rand(*shape)
    x['context_image'] = torch.rand(*shape)
    x['cood'] = torch.rand(shape[0], 2, shape[2], shape[3])

    return x


if __name__ == '__main__':
    main()