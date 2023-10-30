import argparse
import datetime
import os
import numpy as np
import time
import shutil
import wandb


import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.sampler import BatchSampler, SequentialSampler

from model.config import cfg
from model.modeling.sr import build_sr_model
from model.engine.loss_functions import get_loss_fn
from model.utils.sync_batchnorm import convert_model
from model.data.transform.data_preprocess import TrainTransforms
from model.data.dataset.base_dataset import BaseDataset
from model.data.sampler.iteration_based_batch_sampler import IterationBasedBatchSampler
from model.engine.optimizer import build_optimizer
from model.utils.lr_scheduler import WarmupMultiStepLR


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
    parser.add_argument('--resume_from', type=str, default='')


    return parser.parse_args()

def train(args, cfg):
    model = build_sr_model(cfg).to('cuda')
    if cfg.SOLVER.SYNC_BATCHNORM:
        model = convert_model(model).to('cuda')
    loss_fn = get_loss_fn(cfg.SOLVER.SR_LOSS_FN)

    print('------------Model Architecture-------------')
    print(model)

    print('Loading Datasets...')
    data_loader = {}

    train_transforms = TrainTransforms(cfg)
    train_dataset = BaseDataset(args, cfg, cfg.DATA.TRAIN.IMAGE_DIR, cfg.DATA.TRAIN.ANN_FILE, cfg.DATA.TRAIN.MASK_DIR, train_transforms)
    sampler = RandomSampler(train_dataset)
    batch_sampler = BatchSampler(sampler=sampler, batch_size=cfg.SOLVER.BATCH_SIZE, drop_last=True)
    batch_sampler = IterationBasedBatchSampler(batch_sampler=batch_sampler, num_iterations=cfg.SOLVER.MAX_ITER)
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_sampler=batch_sampler, pin_memory=True)
    print('Done.')

    data_loader['train'] = train_loader

    optimizer = build_optimizer(cfg, model)
    scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.LR, cfg.SOLVER.LR_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.MIXED_PRECISION)

    model_path = os.path.join(cfg.OUTPUT_DIR, 'model', f'iteration_0.pth')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)

    if args.num_gpus > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpus)))

    max_iter =  len(data_loader['train'])
    tic = time.time()
    end = time.time()
    trained_time = 0
    logging_loss = 0

    model.train()
    for iteration, (images, targets) in enumerate(data_loader['train'], 1):
        optimizer.zero_grad()
        sr = model(images['detect_input'].to('cuda'))
        loss = loss_fn(sr, targets['sr'].to('cuda'))
        loss = loss.mean()

        logging_loss += loss.detach()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scheduler.step()

        trained_time += time.time() - end
        end = time.time()

        if iteration % args.log_step == 0:
            eta_seconds = int((trained_time / iteration) * (max_iter - iteration))
            logging_loss /= args.log_step
            print('===> Iter: {:07d}, LR: {:.06f}, Cost: {:2f}s, Eta: {}, Loss: {:.6f}'.format(iteration, optimizer.param_groups[0]['lr'], time.time() - tic, str(datetime.timedelta(seconds=eta_seconds)), logging_loss))
            if not args.debug:
                log_dict = {
                    'train/{loss}': logging_loss
                }
                wandb.log(log_dict, step=iteration)

            tic = time.time()

        if iteration % args.save_step == 0:
            model_path = os.path.join(cfg.OUTPUT_DIR, 'model', 'iteration_{}.pth'.format(iteration))
            optimizer_path = os.path.join(cfg.OUTPUT_DIR, 'optimizer', 'iteration_{}.pth'.format(iteration))
            scaler_path = os.path.join(cfg.OUTPUT_DIR, 'scaler', 'iteration_{}.pth'.format(iteration))
            scheduler_path = os.path.join(cfg.OUTPUT_DIR, 'scheduler', 'iteration_{}.pth'.format(iteration))

            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            os.makedirs(os.path.dirname(optimizer_path), exist_ok=True)
            os.makedirs(os.path.dirname(scheduler_path), exist_ok=True)
            os.makedirs(os.path.dirname(scaler_path), exist_ok=True)

            if args.num_gpus > 1:
                torch.save(model.module.state_dict(), model_path)
            else:
                torch.save(model.state_dict(), model_path)

            torch.save(optimizer.state_dict(), optimizer_path)
            torch.save(scaler.state_dict(), scaler_path)
            torch.save(scheduler.state_dict(), scheduler_path)

            print('=====> Save Checkpoint to {}'.format(model_path))


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