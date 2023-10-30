python train.py --config_file configs/$DATA/$DETECTOR/x1/original.yaml --run_name $DATA/$DETECTOR/x1/original
python train.py --config_file configs/$DATA/$DETECTOR/x1/unet.yaml --run_name $DATA/$DETECTOR/x1/unet
python train.py --config_file configs/$DATA/$DETECTOR/x2/unet.yaml --run_name $DATA/$DETECTOR/x2/unet
python train.py --config_file configs/$DATA/$DETECTOR/x2/unet_wo_mask.yaml --run_name $DATA/$DETECTOR/x2/unet_wo_mask
python train.py --config_file configs/$DATA/$DETECTOR/x2/unet_wo_sr.yaml --run_name $DATA/$DETECTOR/x2/unet_wo_sr
python train.py --config_file configs/$DATA/$DETECTOR/x4/unet.yaml --run_name $DATA/$DETECTOR/x4/unet
python train.py --config_file configs/$DATA/$DETECTOR/x4/unet_wo_mask.yaml --run_name $DATA/$DETECTOR/x4/unet_wo_mask
python train.py --config_file configs/$DATA/$DETECTOR/x4/unet_wo_sr.yaml --run_name $DATA/$DETECTOR/x4/unet_wo_sr
