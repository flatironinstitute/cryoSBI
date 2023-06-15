n_workers=$(nproc)
train_npe_model \
    --image_config_file 6wxb/image_params_bending_training.json \
    --train_config_file 6wxb/resnet18_fft_encoder.json\
    --epochs 600 \
    --estimator_file 6wxb/posterior_6wxb_bending.estimator \
    --loss_file 6wxb/posterior_6wxb_bending.loss \
    --n_workers $n_workers \
    --train_device cuda 