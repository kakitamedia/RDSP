from copy import copy

import cv2
import numpy as np

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask=None, boxes=None, labels=None):
        for t in self.transforms:
            image, mask, boxes, labels = t(image, mask, boxes, labels)

        return image, mask, boxes, labels

class ConvertFromInts:
    def __call__(self, image, mask=None, boxes=None, labels=None):
        if mask is not None:
            return image.astype(np.float32), mask.astype(np.float32), boxes, labels
        else:
            return image.astype(np.float32), mask, boxes, labels

class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, mask=None, boxes=None, labels=None):
        # image /= 255
        image -= self.mean
        image /= self.std

        return image, mask, boxes, labels

class Denormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, mask=None, boxes=None, labels=None):
        image *= self.std
        image += self.mean
        # image *= 255

        return image, mask, boxes, labels

class RandomMirror:
    def __call__(self, image, mask=None, boxes=None, labels=None):
        _, width, _ = image.shape
        if np.random.randint(2):
            image = image[:, ::-1]
            if mask is not None:
                mask = mask[:, ::-1]
            if boxes is not None:
                boxes = boxes.copy()
                boxes[:, 0::2] = width - boxes[:, 2::-2]

        return image, mask, boxes, labels

class RandomCrop:
    def __init__(self, size=(512, 512), box_threshold=0.3, max_crops=10):
        self.box_threshold = box_threshold
        if size == int:
            self.size = (size, size)
        else:
            self.size = size

        self.max_crops = max_crops


    def __call__(self, image, mask=None, boxes=None, labels=None):
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

            if boxes is not None:
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

        return image, mask, boxes, labels

class ConvertColor:
    def __init__(self, current, transform):
        self.current = current
        self.transform = transform

    def __call__(self, image, mask=None, boxes=None, labels=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'RGB' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif self.current == 'BGR' and self.transform == 'RGB':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        elif self.current == 'HSV' and self.transform == "RGB":
            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        else:
            raise NotImplementedError

        return image, mask, boxes, labels

class RandomContrast:
    def __init__(self, lower=0.8, upper=1.2):
        assert upper >= lower, 'contrast upper must be >= lower.'
        assert lower >= 0, 'contrast lower must be non-negative.'
        self.lower = lower
        self.upper = upper

    def __call__(self, image, mask=None, boxes=None, labels=None):
        alpha = np.random.uniform(self.lower, self.upper)
        image *= alpha

        return image, mask, boxes, labels

class RandomBrightness:
    def __init__(self, delta=32.):
        assert delta >= 0.
        assert delta <= 255.
        self.delta = delta

    def __call__(self, image, mask=None, boxes=None, labels=None):
        delta = np.random.uniform(-self.delta, self.delta)
        image += delta

        return image, mask, boxes, labels

class RandomSaturation:
    def __init__(self, lower=0.8, upper=1.2):
        assert upper >= lower, 'saturation upper must be >= lower.'
        assert lower >= 0, 'saturation lower must be non-negative.'
        self.lower = lower
        self.upper = upper

    def __call__(self, image, mask=None, boxes=None, labels=None):
        image[:, :, 1] *= np.random.uniform(self.lower, self.upper)

        return image, mask, boxes, labels

class RandomHue:
    def __init__(self, delta=18.):
        assert delta >= 0. and delta <= 360.
        self.delta = delta

    def __call__(self, image, mask=None, boxes=None, labels=None):
        image[:, :, 0] += np.random.uniform(-self.delta, self.delta)
        image[:, :, 0][image[:, :, 0] > 360.] -= 360.
        image[:, :, 0][image[:, :, 0] < 0.] += 360.

        return image, mask, boxes, labels

class RandomChannelSwap:
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))

    def __call__(self, image, mask=None, boxes=None, labels=None):
        swap = self.perms[np.random.randint(len(self.perms))]
        image = image[:, :, swap]

        return image, mask, boxes, labels

class PhotometricDistort:
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor('RGB', 'HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor('HSV', 'RGB'),
            RandomContrast(),
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_channel_swap = RandomChannelSwap()

    def __call__(self, image, mask=None, boxes=None, labels=None):
        if np.random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])

        image, mask, boxes, labels = self.rand_brightness(image, mask, boxes, labels)
        image, mask, boxes, labels = distort(image, mask, boxes, labels)
        image, mask, boxes, labels = self.rand_channel_swap(image, mask, boxes, labels)

        return image, mask, boxes, labels

class Clip:
    def __init__(self, min=0., max=255.):
        self.min = min
        self.max = max

    def __call__(self, image, mask=None, boxes=None, labels=None):
        image = np.clip(image, self.min, self.max)

        return image, mask, boxes, labels

class ZeroPadTo():
    def __init__(self, size):
        self.size = size

    def __call__(self, image, mask=None, boxes=None, labels=None):
        height, width, _ = image.shape
        pad_size = ((0, self.size[0] - height), (0, self.size[1] - width), (0, 0))
        image = np.pad(image, pad_size)

        return image, mask, boxes, labels