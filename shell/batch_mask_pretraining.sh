export DATA=CityScapes

python mask_pretrain.py --config_file configs/${DATA}/pretrain/mask_${ARC}_${CONF}_x1.yaml --run_name ${DATA}/pretrain/mask_${ARC}_${CONF}_x1
python mask_pretrain.py --config_file configs/${DATA}/pretrain/mask_${ARC}_${CONF}_x2.yaml --run_name ${DATA}/pretrain/mask_${ARC}_${CONF}_x2
python mask_pretrain.py --config_file configs/${DATA}/pretrain/mask_${ARC}_${CONF}_x4.yaml --run_name ${DATA}/pretrain/mask_${ARC}_${CONF}_x4
