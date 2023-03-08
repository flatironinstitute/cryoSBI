cd ../../scr/cryo_sbi/inference

python NPE_train_without_saving.py \
    --image_config_file ../../../experiments/6wxb/image_params_snr01_128.json \
    --train_config_file ../../../experiments/6wxb/resnet18_encoder.json \
    --epochs 300 \
    --estimator_file ../../../experiments/6wxb/6wxb_resnet18.estimator \
    --loss_file ../../../experiments/6wxb/6wxb_resnet18.loss
