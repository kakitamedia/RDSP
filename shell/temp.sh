export DATA=CityScapes

# python mask_pretrain.py --config_file configs/${DATA}/pretrain/mask_${ARC}_x1.yaml --run_name ${DATA}/pretrain/mask_${ARC}_x1
# python mask_pretrain.py --config_file configs/${DATA}/pretrain/mask_${ARC}_x2.yaml --run_name ${DATA}/pretrain/mask_${ARC}_x2
# python mask_pretrain.py --config_file configs/${DATA}/pretrain/mask_${ARC}_x4.yaml --run_name ${DATA}/pretrain/mask_${ARC}_x4

# python train.py --config_file configs/${DATA}/ssd/x2/${ARC}.yaml --run_name ${DATA}/ssd/x2/${ARC} --num_gpus ${NUM_GPUS}
# python train.py --config_file configs/${DATA}/ssd/x4/${ARC}.yaml --run_name ${DATA}/ssd/x4/${ARC} --num_gpus ${NUM_GPUS}
python train.py --config_file configs/${DATA}/ssd/x1/${ARC}.yaml --run_name ${DATA}/ssd/x1/${ARC}
