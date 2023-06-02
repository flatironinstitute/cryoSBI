n_workers=$(nproc)
train_npe_model \
    --image_config_file 6wxb/image_params_mixed_training.json \
    --train_config_file 6wxb/resnet18_fft_encoder.json\
    --epochs 300 \
    --estimator_file 6wxb/posterior_6wxb_mixed.estimator \
    --loss_file 6wxb/posterior_6wxb_mixed.loss \
    --n_workers $n_workers \
    --train_device cuda \
    --whitening_filter /mnt/home/ldingeldein/ceph/cryo_sbi_data/whitening_filter/average_psd_downsampled_128.pt