import argparse
import datetime
import os
from tqdm import tqdm
import numpy as np
import cv2
import json

import torch
from mmcv.ops.nms import nms

from model.config import cfg
from model.data.dataset.base_dataset import BaseDataset

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config_file', type=str, default='', metavar='FILE')
    parser.add_argument('--result_dirs', type=lambda x:list(map(str, x.split(','))), default='')
    parser.add_argument('--output_dirname', type=str, default='')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--score_threshold', type=float, default=0.5)
    parser.add_argument('--iou_threshold', type=float, default=0.5)

    return parser.parse_args()


def merge_results(args, cfg):
    test_dataset = BaseDataset(args, cfg, cfg.DATA.TEST.IMAGE_DIR, cfg.DATA.TEST.ANN_FILE, cfg.DATA.TEST.MASK_DIR, None, inference=True)

    results = []
    for i in tqdm(range(len(test_dataset))):
        result = torch.empty(0, 5)
        for dirname in args.result_dirs:
            result = torch.cat([result, torch.from_numpy(torch.load(os.path.join(dirname, str(i).zfill(4) + '.pt')))], dim=0)
        boxes, scores = result[:, :4], result[:, 4]
        result, _ = nms(boxes, scores, 0.45, score_threshold=0.02)
        results.append([torch.empty(0, 5) if i != 1 else result for i in range(80)])

    results = post_processing(results)
    coco_evaluation(test_dataset, results, args.output_dirname)
    draw_results(test_dataset, results, args.output_dirname, args.score_threshold, args.iou_threshold)


def post_processing(predictions):
    results = []
    for pred in predictions:
        boxes, labels, scores = [], [], []
        for class_id in range(len(pred)):
            for box_id in range(pred[class_id].shape[0]):
                labels.append(class_id)
                boxes.append(list(pred[class_id][box_id][:4]))
                scores.append(pred[class_id][box_id][4])

        results.append([boxes, labels, scores])

    return results


def coco_evaluation(dataset, predictions, output_dir):
    coco_results = []
    for i, pred in enumerate(predictions):
        boxes, labels, scores = pred
        image_id, _ = dataset.get_annotation(i)

        if len(labels) == 0:
            continue

        coco_results.extend(
            {
                'image_id': int(image_id),
                'category_id': int(labels[k]),
                'bbox': [float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])],
                'score': float(scores[k]),
            }
            for k, box in enumerate(boxes)
        )

    from pycocotools.cocoeval import COCOeval
    iou_type = 'bbox'
    json_result_file = os.path.join(output_dir, iou_type + '.json')
    os.makedirs(os.path.dirname(json_result_file), exist_ok=True)
    print(f'Writing results to {json_result_file}')
    with open(json_result_file, 'w') as f:
        json.dump(coco_results, f)
    coco_gt = dataset.coco
    coco_dt = coco_gt.loadRes(json_result_file)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.params.catIds = [1]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval


def draw_results(dataset, predictions, output_dir, score_threshold, iou_threshold):
    for i, pred in enumerate(tqdm(predictions)):
        boxes, labels, scores = pred
        image, filename = dataset._read_image(i)
        image_id, annotation = dataset.get_annotation(i)
        gt_boxes = annotation[0].tolist()
        image = np.ascontiguousarray(image, dtype=np.uint8)

        tp, fp, fn = [], [], []

        for k, box in enumerate(boxes):
            if scores[k] < score_threshold:
                break

            max_iou, gt_id = 0., 0
            for kk, gt_box in enumerate(gt_boxes):
                temp = iou(box, gt_box)
                if max_iou < temp:
                    max_iou = temp
                    gt_id = kk

            if max_iou > iou_threshold:
                gt_boxes.pop(gt_id)
                tp.append(box)
            else:
                fp.append(box)

        fn.extend(gt_boxes)

        for box in tp:
            left = int(box[0])
            top = int(box[1])
            right = int(box[2])
            bottom = int(box[3])

            image = cv2.rectangle(image, (left, top), (right, bottom), (128, 255, 128), 8)

        for box in fp:
            left = int(box[0])
            top = int(box[1])
            right = int(box[2])
            bottom = int(box[3])

            image = cv2.rectangle(image, (left, top), (right, bottom), (51, 51, 220), 8)

        for box in fn:
            left = int(box[0])
            top = int(box[1])
            right = int(box[2])
            bottom = int(box[3])

            image = cv2.rectangle(image, (left, top), (right, bottom), (255, 102, 51), 8)

        save_path = os.path.join(output_dir, 'results', 'visualize_bbox_tpfpfn', filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, image)


def iou(a, b):
    ax_mn, ay_mn, ax_mx, ay_mx = a[0], a[1], a[2], a[3]
    bx_mn, by_mn, bx_mx, by_mx = b[0], b[1], b[2], b[3]

    a_area = (ax_mx - ax_mn + 1) * (ay_mx - ay_mn + 1)
    b_area = (bx_mx - bx_mn + 1) * (by_mx - by_mn + 1)

    abx_mn = max(ax_mn, bx_mn)
    aby_mn = max(ay_mn, by_mn)
    abx_mx = min(ax_mx, bx_mx)
    aby_mx = min(ay_mx, by_mx)
    w = max(0, abx_mx - abx_mn + 1)
    h = max(0, aby_mx - aby_mn + 1)
    intersect = w*h

    iou = intersect / (a_area + b_area - intersect)
    return iou


def main():
    args = parse_args()
    if len(args.config_file) > 0:
        print('Configration file is loaded form {}'.format(args.config_file))
        cfg.merge_from_file(args.config_file)

    if len(args.output_dirname) == 0:
        dt_now = datetime.datetime.now()
        args.output_dirname = os.path.join('output', str(dt_now.date()) + '_' + str(dt_now.time()))

    cfg.freeze()

    merge_results(args, cfg)


if __name__ == '__main__':
    main()