import os
from copy import copy
from PIL import Image
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset

from model.utils.misc import load_compressed_array

class BaseDataset(Dataset):
    def __init__(self, args, cfg, data_dir, ann_file, mask_dir, transforms, inference=False, remove_empty=True):
        from pycocotools.coco import COCO

        self.data_dir = data_dir
        self.ann_file = ann_file
        self.mask_dir = mask_dir
        self.transforms = transforms
        self.inference = inference
        self.factor = cfg.MODEL.SR.FACTOR

        self.coco = COCO(ann_file)
        if remove_empty:
            self.ids = list(self.coco.imgToAnns.keys())
        else:
            self.ids = list(self.coco.imgs.keys())

        self.size = cfg.DATA.INPUT_SIZE
        self.max_crops = 50
        self.box_threshold = cfg.SOLVER.BOX_THRESHOLD
        self.max_objects = cfg.MODEL.DETECTOR.MAX_OBJECTS


    def __getitem__(self, index):
        context_image, filename = self._read_image(index)

        if self.inference:
            if self.transforms:
                context_image, _, _, _ = self.transforms(context_image, None, None, None)

                height, width, _ = context_image.shape
                cood_x = np.repeat(np.arange(0, width, dtype=np.float32)[np.newaxis, :], height, axis=0) / 2048
                cood_y = np.repeat(np.arange(0, height, dtype=np.float32)[:, np.newaxis], width, axis=1) / 2048
                cood = np.stack((cood_x, cood_y), axis=2)

                inputs = {
                    'detect_input': self._to_tensor(np.ascontiguousarray(context_image)),
                    'mask_input': self._to_tensor(np.ascontiguousarray(context_image)),
                    'context_image': self._to_tensor(np.ascontiguousarray(context_image)),
                    'cood': self._to_tensor(np.ascontiguousarray(cood)),
                }

            return inputs, filename

        boxes, labels = self._get_annotation(index)
        mask_target = self._read_mask(index)

        context_image, mask_target, boxes, labels = self.transforms(context_image, mask_target, boxes, labels)
        input_image, mask_target, boxes, labels, top, bottom, left, right = self._random_crop(context_image, mask_target, boxes, labels)

        cood_x = np.repeat(np.arange(left, right, dtype=np.float32)[np.newaxis, :], self.size[0], axis=0) / 2048
        cood_y = np.repeat(np.arange(top, bottom, dtype=np.float32)[:, np.newaxis], self.size[1], axis=1) / 2048
        cood = np.stack((cood_x, cood_y), axis=2)

        boxes = np.pad(boxes, [(0, self.max_objects - boxes.shape[0]), (0, 0)])
        labels = np.pad(labels, [(0, self.max_objects - labels.shape[0])])

        sr_target = copy(input_image)
        mask_input = copy(input_image)
        detect_input = cv2.resize(input_image, dsize=None, fx=1/self.factor, fy=1/self.factor, interpolation=cv2.INTER_AREA)

        inputs = {
            'detect_input': self._to_tensor(np.ascontiguousarray(detect_input)),
            'mask_input': self._to_tensor(np.ascontiguousarray(mask_input)),
            'context_image': self._to_tensor(np.ascontiguousarray(context_image)),
            'cood': self._to_tensor(np.ascontiguousarray(cood)),
        }
        targets = {
            'det': (torch.from_numpy(boxes), torch.from_numpy(labels)),
            'sr': self._to_tensor(np.ascontiguousarray(sr_target)),
            'mask': self._to_tensor(np.ascontiguousarray(mask_target))
        }

        return inputs, targets


    def _read_image(self, index):
        image_id = self.ids[index]
        filename = self.coco.loadImgs(image_id)[0]['file_name']
        image_file = os.path.join(self.data_dir, filename)
        image = Image.open(image_file).convert('RGB')
        image = np.array(image)[:, :, ::-1]

        return image, filename

    def _get_annotation(self, index):
        image_id = self.ids[index]
        # only uses category_id = 1 (i.e., person)
        ann_ids = self.coco.getAnnIds(imgIds=image_id, catIds=[1])
        ann = self.coco.loadAnns(ann_ids)

        ann = [obj for obj in ann if obj['iscrowd'] == 0]
        boxes = np.array([self._xywh2xyxy(obj['bbox']) for obj in ann], np.float32).reshape((-1, 4))
        labels = np.array([obj['category_id'] for obj in ann], np.int64)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        labels = labels[keep]

        return boxes, labels

    def _read_mask(self, index):
        image_id = self.ids[index]
        filename = self.coco.loadImgs(image_id)[0]['file_name']
        mask_file = os.path.join(self.mask_dir, f'x{self.factor}', os.path.splitext(filename)[0] + '.npz')
        mask = load_compressed_array(mask_file)

        return mask

    def get_annotation(self, index):
        image_id = self.ids[index]
        annotation = self._get_annotation(index)

        return image_id, annotation

    def _random_crop(self, image, mask, boxes, labels):
        height, width, _ = image.shape

        # When the image is smaller than the crop size, the image is padded by 0
        padding = ((0, max(0, self.size[0]-height)), (0, max(0, self.size[1]-width)), (0, 0))
        image = np.pad(image, padding, 'constant')
        if mask is not None:
            mask = np.pad(mask, padding, 'constant')

        original_boxes = boxes
        original_labels = labels

        for _ in range(self.max_crops):
            boxes = copy(original_boxes)
            labels = copy(original_labels)

            left = np.random.randint(max(0, width - self.size[1]) + 1)
            top = np.random.randint(max(0, height - self.size[0]) + 1)
            right = left + self.size[1]
            bottom = top + self.size[0]

            if boxes is None:
                break

            else:
                original_box_sizes = boxes[:, 2:] - boxes[:, :2]
                boxes[:, 0::2] = boxes[:, 0::2] - left
                boxes[:, 1::2] = boxes[:, 1::2] - top

                boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, self.size[1])
                boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, self.size[0])
                cropped_box_sizes = boxes[:, 2:] - boxes[:, :2]

                # Remove invalid boxes
                keep1 = cropped_box_sizes > 0
                keep1 = keep1.all(1)

                keep2 = cropped_box_sizes/original_box_sizes >= self.box_threshold
                keep2 = keep2.all(1)

                keep = np.logical_and(keep1, keep2)

                boxes = boxes[keep]
                labels = labels[keep]

                # There are no objects in cropped image, try cropping again up to 'max_crops' times.
                if boxes.shape[0] > 0:
                    break

        image = image[top:bottom, left:right, :]
        if mask is not None:
            mask = mask[top:bottom, left:right, :]

        return image, mask, boxes, labels, top, bottom, left, right

    def _to_tensor(self, tensor):
        return torch.from_numpy(tensor).permute(2, 0, 1)

    def __len__(self):
        return len(self.ids)

    def _xywh2xyxy(self, box):
        x1, y1, w, h = box
        return [x1, y1, x1+w, y1+h]