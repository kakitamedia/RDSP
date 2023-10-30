import sys
sys.path.append('../')

import argparse
import json
import os

import numpy as np
from scipy import ndimage
import cv2
from pycocotools.coco import COCO
from joblib import Parallel, delayed
from tqdm import  tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='datasets/CityScapes/leftImg8bit')
    parser.add_argument('--ann_file', type=str, default='datasets/CityScapes/coco_format/train.json')
    parser.add_argument('--output_dirname', type=str, default='datasets/CityScapes/mask_gts')
    parser.add_argument('--num_workers', type=int, default=8)

    return parser.parse_args()

height_thresholds = {
    1: [64, sys.maxsize],
    2: [32, 64],
    4: [0, 32],
}
gaussian_sigma = 21


def main():
    args = parse_args()

    with open(args.ann_file) as f:
        w = json.load(f)

    coco = COCO(args.ann_file)
    ids = list(coco.imgToAnns.keys())

    def make(id):
        filename = coco.loadImgs(id)[0]['file_name']
        image = cv2.imread(os.path.join(args.image_dir, filename))

        ann_ids = coco.getAnnIds(imgIds=id, catIds=[1, 2, 3, 4, 6, 7, 8])
        ann = coco.loadAnns(ann_ids)

        boxes = np.array([obj['bbox'] for obj in ann], np.float32).reshape((-1, 4))
        keep = (boxes[:, 3] > 0) & (boxes[:, 2] > 0)
        boxes = boxes[keep]
        num_objects = boxes.shape[0]

        height, width, _ = image.shape

        for scale in height_thresholds.keys():
            heatmap = np.zeros((height, width), dtype=np.float32)

            for i in range(num_objects):
                box = boxes[i]
                box_height = box[3]

                if height_thresholds[scale][0] <= box_height < height_thresholds[scale][1]:
                    heatmap = draw_heatmap(heatmap, box)

            heatmap = heatmap[:, :, np.newaxis]

            save_path = os.path.join(args.output_dirname, f'x{scale}', filename)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            save_compressed_array(save_path, heatmap)


    with tqdm_joblib(tqdm(total=len(ids))) as progess_bar:
        Parallel(n_jobs=args.num_workers, backend="threading")([delayed(make)(id) for id in ids])



def draw_heatmap(heatmap, box):
    left, top, right, bottom = [int(cood) for cood in xywh2xyxy(box)]

    heatmap_per_obj = np.zeros_like(heatmap)
    heatmap_per_obj[top:bottom, left:right] = 1.
    heatmap_per_obj = ndimage.gaussian_filter(heatmap_per_obj, sigma=gaussian_sigma)
    heatmap_per_obj = heatmap_per_obj / heatmap_per_obj.max()

    return np.maximum(heatmap, heatmap_per_obj)


def xywh2xyxy(box):
    x1, y1, w, h = box
    return [x1, y1, x1+w, y1+h]


def save_compressed_array(path, array):
    path = os.path.join(os.path.dirname(path), os.path.splitext(os.path.basename(path))[0] + '.npz')
    np.savez_compressed(path, array)


import contextlib
import joblib
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()



if __name__ == '__main__':
    main()