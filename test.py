import argparse
import os
from model.utils.misc import str2bool
import numpy as np
import datetime

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler, BatchSampler

from model.config import cfg
from model.modeling.build_model import Model
from model.data.transform.data_preprocess import TestTransforms
from model.data.dataset.base_dataset import BaseDataset
from model.engine.inference import do_test

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config_file', type=str, default='', metavar='FILE', help='')
    parser.add_argument('--output_dirname', type=str, default='')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--trained_model', type=str, default='')
    parser.add_argument('--split_forward', action='store_true')
    parser.add_argument('--patch_size', type=int, default=512)
    parser.add_argument('--save_sr', action='store_true')

    return parser.parse_args()

def test(args, cfg):
    model = Model(cfg).to('cuda')

    total_params = sum(p.numel() for p in model.parameters())

    print('------------Model Architecture-------------')
    print(model)

    print(f'number of parameters: {total_params}')

    print('Loading Datasets...')
    test_transforms = TestTransforms(cfg)
    test_dataset = BaseDataset(args, cfg, cfg.DATA.TEST.IMAGE_DIR, cfg.DATA.TEST.ANN_FILE, cfg.DATA.TEST.MASK_DIR, test_transforms, inference=True)
    sampler = SequentialSampler(test_dataset)
    batch_sampler = BatchSampler(sampler=sampler, batch_size=args.batch_size, drop_last=False)
    test_loader = DataLoader(test_dataset, num_workers=args.num_workers, batch_sampler=batch_sampler)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)

    if args.trained_model:
        print(f'Trained model was loaded from {args.trained_model}')
        model.load_state_dict(torch.load(args.trained_model), strict=True)
    else:
        print('!!!Warning!!!: Trained model was not specified.')

    if args.num_gpus > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpus)))

    do_test(args, cfg, model, test_loader)


def main():
    args = parse_args()

    if len(args.config_file) > 0:
        print('Configuration file is loaded from {}'.format(args.config_file))
        cfg.merge_from_file(args.config_file)

    if len(args.output_dirname) == 0:
        dt_now = datetime.datetime.now()
        output_dirname = os.path.join('output', str(dt_now.date()) + '_' + str(dt_now.time()))
    else:
        output_dirname = args.output_dirname
    cfg.OUTPUT_DIR = output_dirname
    cfg.MODEL.DETECTOR.FLAG = True
    cfg.freeze()

    print('OUTPUT DIRNAME: {}'.format(cfg.OUTPUT_DIR))

    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        # torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(cfg.SEED)
    else:
        raise Exception('GPU not found')

    test(args, cfg)

if __name__ == '__main__':
    main()
