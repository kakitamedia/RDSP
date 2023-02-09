import argparse
import datetime
import os
import numpy as np
import shutil

import torch

from model.config import cfg
from model.modeling.build_model import Model
from model.utils.sync_batchnorm import convert_model
from model.data.dataset.base_dataset import Datasetf
rom model.data.


def parse_args():
    parser = argparse.ArgumentParser(description='RDSP training')
    parser.add_argument('--config_file', type=str, default='', metavar='FILE')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--run_name', type=str, default='')
    parser.add_argument('--log_step', type=int, default=50, help='')
    parser.add_argument('--eval_step', type=int, default=0, help='')
    parser.add_argument('--save_step', type=int, default=50000, help='')
    parser.add_argument('--num_gpus', type=int, default=1, help='')
    parser.add_argument('--num_workers', type=int, default=16, help='')

    return parser.parse_args()

def train(args, cfg):
    model = Model(cfg).to('cuda')
    if cfg.SOLVER.SYNC_BATCHNORM:
        model = convert_model(model).to('cuda')

    print('------------Model Architecture-------------')
    print(model)

    print('Loading Datasets...')
    data_loader = {}

    train_transforms = TrainTransforms(cfg)
    train_dataset = Dataset()

    print('Done.')


def main():
    args = parse_args()

    if len(args.config_file) > 0:
        print('Configration file is loaded from {}'.format(args.config_file))
        cfg.merge_from_file(args.config_file)

    if len(args.run_name) == 0:
        dt_now = datetime.datetime.now()
        args.run_name = str(dt_now.date()) + '_' + str(dt_now.time())

    output_dirname = os.path.join('output', args.run_name)
    cfg.OUTPUT_DIR = output_dirname
    cfg.freeze()

    # fix the random seeds
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.SEED)
        torch.backends.cudnn.benchmark = True
    else:
        raise Exception('GPU not found')

    if not args.debug:
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        if not len(args.config_file) == 0:
            shutil.copy2(args.config_file, os.path.join(cfg.OUTPUT_DIR, 'config.yaml'))

        import wandb
        wandb.init(project='RDSP', entity='kakita', config=dict(yaml=cfg))
        wandb.run.name = args.run_name
        wandb.run.save()

    train(args, cfg)

if __name__ == '__main__':
    main()