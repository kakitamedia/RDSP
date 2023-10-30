import argparse
import glob
import os
from copy import copy
import re
import json

from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann_dir', type=str, default='datasets/CityScapes/gtFine')
    parser.add_argument('--output_dirname', type=str, default='datasets/CityScapes/coco_format')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--only_person', action='store_true')

    return parser.parse_args()




def main():
    args = parse_args()

    if args.only_person:
        cityscapes_categorie_to_coco_id = {
        'person': 1,
        'rider': 1,
    }
    else:
        cityscapes_categorie_to_coco_id = {
            'person': 1,
            'rider': 1,
            'car': 3,
            'truck': 8,
            'bus': 6,
            'on rails': 7,
            'motorcycle': 4,
            'bicycle': 2,
            'caravan': 3,
            'trailer': 8,
        }

    output_dict = {
        'info': {'description': 'CityScapes Dataset'},
        'annotations': [],
        'images': [],
    }

    output_dict['categories'] = [
        {'name': 'person', 'supercategory': 'person', 'id': 1},
        {'name': 'bicycle', 'supercategory': 'vehicle', 'id': 2},
        {'name': 'car', 'supercategory': 'vehicle', 'id': 3},
        {'name': 'motorcycle', 'supercategory': 'vehicle', 'id': 4},
        {'name': 'bus', 'supercategory': 'vehicle', 'id': 6},
        {'name': 'train', 'supercategory': 'vehicle', 'id': 7},
        {'name': 'truck', 'supercategory': 'vehicle', 'id': 8},
    ]

    ann_pathes = glob.glob(os.path.join(args.ann_dir, args.split, '*', '*.json'))

    image_id = 0
    id = 0
    for ann_path in tqdm(ann_pathes):
        filename = copy(ann_path)
        filename = re.sub(fr'.*{args.split}', f'{args.split}', filename)
        filename = re.sub(r'gt.*json', 'leftImg8bit.png', filename)

        with open(ann_path) as f:
            json_file = json.load(f)

        height = json_file['imgHeight']
        width = json_file['imgWidth']

        for object in json_file['objects']:
            label = object['label']
            polygon = object['polygon']

            try:
                category_id = cityscapes_categorie_to_coco_id[label]
            except KeyError:
                continue

            segmentation = [[xy[0] for xy in polygon], [xy[1] for xy in polygon]]
            area = shoelace_area(segmentation[0], segmentation[1])

            left, top = width, height
            right, bottom = 0, 0

            for x, y in polygon:
                if x > right:
                    right = x
                if x < left:
                    left = x
                if y > bottom:
                    bottom = y
                if y < top:
                    top = y

            if left < 0:
                left = 0
            if top < 0:
                top = 0

            annotation = {
                'category_id': category_id,
                'segmentation': segmentation, # [[x coods], [y coods]]
                'bbox': [left, top, right-left, bottom-top], # [x, y, w, h]
                'height': height,
                'width': width,
                'id': id,
                'image_id': image_id,
                'iscrowd': 0,
                'area': area,
            }

            output_dict['annotations'].append(annotation)
            id += 1

        image = {
            'id': image_id,
            'file_name': filename,
            'height': height,
            'width': width,
            'data_captured': '',
            'flickr_url': '',
            'license': 1,
        }

        output_dict['images'].append(image)
        image_id += 1

    save_path = os.path.join(args.output_dirname, f'{args.split}.json')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(output_dict, f)


def shoelace_area(x_list,y_list):
    a1, a2 = 0, 0
    x_list.append(x_list[0])
    y_list.append(y_list[0])
    for j in range(len(x_list)-1):
        a1 += x_list[j] * y_list[j+1]
        a2 += y_list[j] * x_list[j+1]
    l = abs(a1-a2) / 2
    return l

def str2bool(s):
    return s.lower() in ('true', '1')

if __name__ == '__main__':
    main()