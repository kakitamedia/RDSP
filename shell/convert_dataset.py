import os
import args


def main():
    os.system('python scripts/convert_to_coco_foramt.py --split train')
    os.system('python scripts/convert_to_coco_foramt.py --split val --only_person')

    os.system('python scripts/make_mask_gts')

if __name__ == '__main__':
    main()