python3 -m cryo_sbi.inference.NPE_train_without_saving \
    --image_config_file 6wxb/image_params_training.json \
    --train_config_file 6wxb/resnet18_fft_encoder.json\
    --epochs 300 \
    --estimator_file 6wxb/posterior_6wxb_fft.estimator \
    --loss_file 6wxb/posterior_6wxb_fft.loss \
    --n_workers 24
