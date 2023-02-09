import os
import cv2

from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, args, cfg, data_dir, ann_file, transforms, inference=False, remove_empty=True):
        from pycocotools.coco import COCO

        self.data_dir = data_dir
        self.ann_file = ann_file
        self.transforms = transforms
        self.inference = inference

        self.coco = COCO(ann_file)
        if remove_empty:
            self.ids = list(self.coco.imgToAnns.keys())
        else:
            self.ids = list(self.coco.imgs.keys())

        coco_categories = sorted(self.coco.getCatIds())
        self.coco_id_to_contiguous_id = {coco_id: i+1 for i, coco_id in enumerate(coco_categories)}
        self.contiguous_id_to_coco_id = {v: k for k, v in self.coco_id_to_contiguous_id.items()}


    def __getitem__(self, index):
        context_image, filename = self._read_image(index)

    def _read_image(self, index):
        image_id = self.ids[index]
        filename = self.coco.loadImgs(image_id)[0]['file_name']
        image_file = os.path.join(self.data_dir, filename)
        image = Image.open(image_file).convert('RGB')
        image = np.array(image)[:, :, ::-1]
