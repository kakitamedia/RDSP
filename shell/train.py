import os
import args

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='base')
    parser.add_argument('--base', action='store_ture')

    return parser.parse_args()

def main():
    args = parse_args()

    os.systen(f'python train.py --config_file configs/CityScapes/ssd/x4/base.yaml --run_name CityScapes/ssd/x4/base')
    os.systen(f'python train.py --config_file configs/CityScapes/ssd/x2/base.yaml --run_name CityScapes/ssd/x2/base')
    os.systen(f'python train.py --config_file configs/CityScapes/ssd/x1/base.yaml --run_name CityScapes/ssd/x1/base')
    os.systen(f'python train.py --config_file configs/CityScapes/ssd/x1/orignal.yaml --run_name CityScapes/ssd/x1/original')


if __name__ == '__main__':
    main()