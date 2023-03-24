python3 -m cryo_sbi.inference.NPE_train_without_saving \
    --image_config_file 6wxb/image_params_training.json \
    --train_config_file 6wxb/resnet18_encoder.json\
    --epochs 300 \
    --estimator_file 6wxb/posterior_6wxb.estimator \
    --loss_file 6wxb/posterior_6wxb.loss \
    --n_workers 24
