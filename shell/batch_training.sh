DATA='CityScapes'
DETECTOR='ssd'

python train.py --config_file configs/$DATA/$DETECTOR/x1/unet.yaml --run_name $DATA/$DETECTOR/x1/unet --debug
python train.py --config_file configs/$DATA/$DETECTOR/x2/unet.yaml --run_name $DATA/$DETECTOR/x2/unet --debug
python train.py --config_file configs/$DATA/$DETECTOR/x4/unet.yaml --run_name $DATA/$DETECTOR/x4/unet --debug
