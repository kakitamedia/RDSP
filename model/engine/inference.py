import itertools
import numpy as np
import os
from copy import deepcopy
import time

from tqdm import tqdm
from PIL import Image
import cv2

import torch

def do_test(args, cfg, model, data_loader):
    dataset = data_loader.dataset
    factor = cfg.MODEL.SR.FACTOR
    patch_size = (args.patch_size, args.patch_size)
    save_dir = os.path.join(args.output_dirname, 'results')

    det_results = []
    model.eval()
    print('inference starts!')
    total_second = 0
    for i, (images, filename) in enumerate(tqdm(data_loader)):
        with torch.inference_mode():
            tic = time.time()
            if args.split_forward:
                sr, mask, det = split_forward(images, model, factor, patch_size)
            else:
                sr, mask, det = forward(images, model, factor)
            toc = time.time()
            total_second += toc - tic

            if args.save_sr:
                for j in range(sr.shape[0]):
                    save_path = os.path.join(save_dir, 'sr', filename[j])
                    save_sr_results(deepcopy(sr[j]), save_path)
                    save_path = os.path.join(save_dir, 'mask', filename[j])
                    save_mask_results(mask[j], deepcopy(sr[j]),  save_path)

        det_results.extend(det)

    save_dir = os.path.join(save_dir, 'det')
    save_det_results(det_results, save_dir)

    print(f'Avarage inference time: {total_second/len(data_loader)} sec.')


def forward(images, model, factor):
    sr, mask, det = model(images)
    for batch_id in range(len(det)):
            for class_id in range(len(det[batch_id])):
                det[batch_id][class_id][:, :4] /= factor

    return sr, mask, det

def split_forward(images, model, factor, patch_size):
    batch, channel, height, width = images['detect_input'].shape
    stride = [patch_size[0] // 2, patch_size[1] // 2]
    num_patches = [height // stride[0] - 1, width // stride[1] -1]

    results = {
        'sr': torch.zeros(batch, channel, height*factor, width*factor, dtype=torch.float32),
        'mask': torch.zeros(batch, 1, height*factor, width*factor, dtype=torch.float32),
        'det': [],
    }

    for i, j in itertools.product(range(num_patches[0]), range(num_patches[1])):
        top, left = stride[0] * i, stride[1] * j
        bottom, right = top + patch_size[0], left + patch_size[1]

        # print(top, bottom, left, right)

        patch_images = {
            'detect_input': images['detect_input'][:, :, top:bottom, left:right],
            'mask_input': images['mask_input'][:, :, top:bottom, left:right],
            'context_image': images['context_image'],
            'cood': images['cood'][:, :, top:bottom, left:right],
        }

        sr, mask, det = model(patch_images)

        for batch_id in range(len(det)):
            for class_id in range(len(det[batch_id])):
                det[batch_id][class_id][:, :4] /= factor
                det[batch_id][class_id][:, 1:4:2] += top
                det[batch_id][class_id][:, 0:4:2] += left
                try:
                    results['det'][batch_id][class_id] = np.concatenate([results['det'][batch_id][class_id], det[batch_id][class_id]], axis=0)
                except IndexError:
                    results['det'] = det

        results['sr'][:, :, top*factor:bottom*factor, left*factor:right*factor] += sr.detach().cpu()
        results['mask'][:, :, top*factor:bottom*factor, left*factor:right*factor] += mask.detach().cpu()

    scaler = torch.ones(batch, channel, height*factor, width*factor)
    scaler[:, :, stride[0]*factor:-stride[0]*factor, :] /= 2
    scaler[:, :, :, stride[1]*factor:-stride[1]*factor] /= 2

    results['sr'] = results['sr'] * scaler
    results['mask'] = results['mask'] * scaler

    return results['sr'], results['mask'], results['det']


def save_sr_results(result, save_path, mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375)):
    result = to_image(result, mean, std)
    image = Image.fromarray(result[:, :, ::-1])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    image.save(save_path)


def save_mask_results(mask_result, sr_result, save_path, mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375)):
    sr_result = to_image(sr_result, mean, std)

    mask_result = mask_result.detach().cpu().permute(1, 2, 0).numpy()
    mask_result = np.clip(mask_result * 255, 0, 255).astype(np.uint8)
    mask_result = cv2.applyColorMap(mask_result, cv2.COLORMAP_JET)

    image = cv2.addWeighted(sr_result, 0.0, mask_result, 1.0, 0)
    image = Image.fromarray(image[:, :, ::-1])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    image.save(save_path)


def save_det_results(results, save_dir):
    print(f'Saving detection results to {save_dir}')
    os.makedirs(save_dir, exist_ok=True)
    for i in range(len(results)):
        save_path = os.path.join(save_dir, f'{str(i).zfill(4)}.pt')
        torch.save(results[i][1], save_path)


def to_image(tensor, mean, std):
    tensor = tensor.detach().cpu().permute(1, 2, 0).numpy()
    tensor *= std
    tensor += mean
    tensor = np.clip(tensor, 0, 255).astype(np.uint8)

    return tensor
