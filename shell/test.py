import os
import args

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', action='store_ture')
    parser.add_argument('--split_forward', action='store_true')

    return parser.parse_args()

def main():
    args = parse_args()

    if args.split_forward:
        split = '--split_forward'
    else:
        split = ''

    if args.base:
        os.system(f'python test.py --config_file configs/CityScapes/ssd/x4/base.yaml --output_dirname output/CityScapes/ssd/x4/{args.config} --trained_model weights/ssd_x4.pth {split}')
        os.system(f'python test.py --config_file configs/CityScapes/ssd/x2/base.yaml --output_dirname output/CityScapes/ssd/x2/{args.config} --trained_model weights/ssd_x2.pth {split}')
        os.system(f'python test.py --config_file configs/CityScapes/ssd/x1/base.yaml --output_dirname output/CityScapes/ssd/x1/{args.config} --trained_model weights/ssd_x1.pth {split}')
        os.system(f'python test.py --config_file configs/CityScapes/ssd/x1/original.yaml --output_dirname output/CityScapes/ssd/x1/{args.config} --trained_model weights/ssd_original.pth  {split}')

    else:
        os.system(f'python test.py --config_file configs/CityScapes/ssd/x4/base.yaml --output_dirname output/CityScapes/ssd/x1/original --trained_model output/CityScapes/ssd/x4/base/model/iteration_500000.pth {split}')
        os.system(f'python test.py --config_file configs/CityScapes/ssd/x2/base.yaml --output_dirname output/CityScapes/ssd/x1/original --trained_model output/CityScapes/ssd/x2/base/model/iteration_500000.pth {split}')
        os.system(f'python test.py --config_file configs/CityScapes/ssd/x1/base.yaml --output_dirname output/CityScapes/ssd/x1/original --trained_model output/CityScapes/ssd/x1/base/model/iteration_500000.pth {split}')
        os.system(f'python test.py --config_file configs/CityScapes/ssd/x1/original.yaml --output_dirname output/CityScapes/ssd/x1/original --trained_model output/CityScapes/ssd/x1/original/model/iteration_500000.pth {split}')

if __name__ == '__main__':
    main()