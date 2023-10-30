import os

detector_name = 'output/CityScapes/ssd'
model_name = 'original'

# os.system(f'python merge_results.py --result_dirs {detector_name}/x1/{model_name}/results/det,{detector_name}/x2/{model_name}/results/det,{detector_name}/x4/{model_name}/results/det --output_dirname {detector_name}/merged/{model_name}')
# os.system(f'python merge_results.py --result_dirs {detector_name}/x1/{model_name}/results/det,{detector_name}/x2/{model_name}/results/det, --output_dirname {detector_name}/merged/{model_name}')
# os.system(f'python merge_results.py --result_dirs {detector_name}/x1/original/results/det,{detector_name}/x2/{model_name}/results/det,{detector_name}/x4/{model_name}/results/det --output_dirname {detector_name}/merged/{model_name}')
os.system(f'python merge_results.py --result_dirs {detector_name}/x1/original/results/det --output_dirname {detector_name}/x1/{model_name}')
# os.system(f'python merge_results.py --result_dirs {detector_name}/x1/{model_name}/results/det --output_dirname {detector_name}/x1/{model_name}')
# os.system(f'python merge_results.py --result_dirs {detector_name}/x1/unet/results/det,{detector_name}/x2/{model_name}/results/det,{detector_name}/x4/{model_name}/results/det --output_dirname {detector_name}/merged/{model_name}')
# os.system(f'python merge_results.py --config_file configs/debug.yaml --result_dirs {detector_name}/x1/{model_name}/results/det,{detector_name}/x2/{model_name}/results/det,{detector_name}/x4/{model_name}/results/det --output_dirname {detector_name}/merged/{model_name}')
# os.system(f'python merge_results.py --config_file configs/debug.yaml --result_dirs {detector_name}/x1/original/results/det,{detector_name}/x2/{model_name}/results/det,{detector_name}/x4/{model_name}/results/det --output_dirname {detector_name}/merged/{model_name}')
# os.system(f'python merge_results.py --config_file configs/debug.yaml --result_dirs {detector_name}/x1/original/results/det --output_dirname {detector_name}/x1/{model_name}')
# os.system(f'python merge_results.py --config_file configs/debug.yaml --result_dirs {detector_name}/x1/original/results/det,{detector_name}/x2/{model_name}/results/det,{detector_name}/x4/{model_name}/results/det --output_dirname {detector_name}/merged/{model_name}')
# os.system(f'python merge_results.py --config_file configs/debug.yaml --result_dirs {detector_name}/x1/{model_name}/results/det --output_dirname {detector_name}/x1/{model_name}')
# os.system(f'python merge_results.py --config_file configs/debug.yaml --result_dirs {detector_name}/x1/{model_name}/results/det,{detector_name}/x2/{model_name}/results/det --output_dirname {detector_name}/x1/{model_name}')

# original
# os.system(f'python merge_results.py --config_file configs/debug.yaml --result_dirs {detector_name}/x1/original/results/det --output_dirname {detector_name}/x1/original')

# wo mask
# os.system(f'python merge_results.py --config_file configs/debug.yaml --result_dirs {detector_name}/x1/original/results/det,{detector_name}/x2/unet_wo_mask/results/det,{detector_name}/x4/unet_wo_mask/results/det --output_dirname {detector_name}/merged/unet_wo_mask')

# unet
# os.system(f'python merge_results.py --config_file configs/debug.yaml --result_dirs {detector_name}/x1/unet/results/det,{detector_name}/x2/unet/results/det,{detector_name}/x4/unet/results/det --output_dirname {detector_name}/merged/unet')

# os.system(f'python merge_results.py --result_dirs {detector_name}/x1/original/results/det,{detector_name}/x1/{model_name}/results/det,{detector_name}/x2/{model_name}/results/det,{detector_name}/x4/{model_name}/results/det --output_dirname {detector_name}/merged_ex/{model_name}')
# os.system(f'python merge_results.py --config_file configs/debug.yaml --result_dirs {detector_name}/x1/original/results/det,{detector_name}/x1/{model_name}/results/det,{detector_name}/x2/{model_name}/results/det,{detector_name}/x4/{model_name}/results/det --output_dirname {detector_name}/merged_ex/{model_name}')

# os.system(f'python merge_results.py --result_dirs {detector_name}/x1/unet/results/det,{detector_name}/x2/{model_name}/results/det,{detector_name}/x4/{model_name}/results/det --output_dirname {detector_name}/merged/{model_name}')
# os.system(f'python merge_results.py --result_dirs {detector_name}/x1/original/results/det,{detector_name}/x1/unet/results/det,{detector_name}/x2/{model_name}/results/det,{detector_name}/x4/{model_name}/results/det --output_dirname {detector_name}/merged_ex/{model_name}')
