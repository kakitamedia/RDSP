# export RUN=CityScapes/centernet_wo_dcn/x1/original

# python test.py --config_file output/$RUN/config.yaml --trained_model output/$RUN/model/iteration_500000.pth --output_dirname output/$RUN --split_forward
# python test.py --config_file output/$RUN/config.yaml --trained_model output/$RUN/model/iteration_500000.pth --output_dirname output/$RUN
python test.py --config_file output/$RUN/config.yaml --trained_model output/$RUN/model/iteration_500000.pth --output_dirname output/$RUN --split_forward --save_sr
# python test.py --config_file output/$RUN/config.yaml --trained_model output/$RUN/model/iteration_500000.pth --output_dirname output/$RUN --save_sr
