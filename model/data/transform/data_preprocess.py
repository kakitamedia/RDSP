from .transforms import *

class TrainTransforms:
    def __init__(self, cfg):
        self.transforms = Compose([
            ConvertFromInts(),
            RandomMirror(),
            Clip(),
            Normalize(cfg.DATA.INPUT_MEAN, cfg.DATA.INPUT_STD),
        ])

    def __call__(self, image, mask, boxes, labels):
        image, mask, boxes, labels = self.transforms(image, mask, boxes, labels)

        return image, mask, boxes, labels


class TestTransforms:
    def __init__(self, cfg):
        self.transforms = Compose([
            ConvertFromInts(),
            # ZeroPadTo((768, 1280)),
            Normalize(cfg.DATA.INPUT_MEAN, cfg.DATA.INPUT_STD),
        ])

    def __call__(self, image, mask, boxes, labels):
        image, mask, boxes, labels = self.transforms(image, mask, boxes, labels)

        return image, mask, boxes, labels