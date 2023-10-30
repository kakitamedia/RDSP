import time
import os
import wandb
import datetime
from tqdm import tqdm

import torch
import torch.nn as nn

def do_train(args, cfg, model, optimizer, scheduler, scaler, data_loader):
    max_iter = len(data_loader['train'])
    trained_time = 0
    tic = time.time()
    end = time.time()

    logging_loss = {}
    print('Training starts!')
    model.train()
    for iteration, (images, targets) in enumerate(data_loader['train'], 1):
        optimizer.zero_grad()
        loss, loss_dict = model(images, targets=targets)
        loss = loss.mean()

        scaler.scale(loss).backward()
        if cfg.SOLVER.GRAD_CLIP > 0:
            scheduler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.SOLVER.GRAD_CLIP)
        scaler.step(optimizer)
        scheduler.step()

        loss_dict = {k:v.mean().item() for k, v in loss_dict.items()}
        for k in loss_dict.keys():
            if k in logging_loss:
                logging_loss[k] += loss_dict[k]
            else:
                logging_loss[k] = loss_dict[k]

        trained_time += time.time() - end
        end = time.time()

        if iteration % args.log_step == 0:
            eta_seconds = int((trained_time / iteration) * (max_iter - iteration))
            logging_loss = {k:v / args.log_step for k, v in logging_loss.items()}
            logging_loss['total_loss'] = sum([v for v in logging_loss.values()])
            print('===> Iter: {:07d}, LR: {:.06f}, Cost: {:2f}s, Eta: {}, Loss: {:.6f}'.format(iteration, optimizer.param_groups[0]['lr'], time.time() - tic, str(datetime.timedelta(seconds=eta_seconds)), logging_loss['total_loss']))

            if not args.debug:
                log_dict = {'train/{}'.format(k): logging_loss[k] for k in logging_loss.keys()}
                log_dict['train/lr'] = optimizer.param_groups[0]['lr']
                wandb.log(log_dict, step=iteration)

            logging_loss = {}

            tic = time.time()

        if iteration % args.save_step == 0 and not args.debug:
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

        if 'val' in data_loader.keys() and iteration % args.eval_step == 0:
            print('Validating...')
            model.eval()
            val_loss = 0
            for _ in tqdm(data_loader['val']):
                with torch.inference_mode():
                    loss = model()
                    val_loss += loss.item()

            val_loss /= len(data_loader['val'])

            validation_time = time.time() - end
            trained_time += validation_time
            end = time.time()
            tic = time.time()
            print('======> Cost: {:2f}s, Loss: {:.06f}'.format(validation_time, val_loss))

            if not args.debug:
                log_dict = {'val/loss': val_loss}
                wandb.log(log_dict, step=iteration)

            model.train()
