train_npe_model \
    --image_config_file 6wxb/image_params_mixed_training.json \
    --train_config_file 6wxb/resnet18_fft_encoder.json\
    --epochs 200 \
    --estimator_file 6wxb/posterior_6wxb_mixed.estimator \
    --loss_file 6wxb/posterior_6wxb_mixed.loss \
    --n_workers 24 \
    --train_from_checkpoint \
    --state_dict_file 6wxb/posterior_6wxb.estimator \
    --train_from_checkpoint \
    --state_dict_file 6wxb/posterior_6wxb.estimator \
    --train_device cuda

train_npe_model \
    --image_config_file 6wxb/image_params_torsion_training.json \
    --train_config_file 6wxb/resnet18_fft_encoder.json\
    --epochs 200 \
    --estimator_file 6wxb/posterior_6wxb_torsion.estimator \
    --loss_file 6wxb/posterior_6wxb_torsion.loss \
    --n_workers 24 \
    --train_from_checkpoint \
    --state_dict_file 6wxb/posterior_6wxb.estimator \
    --train_from_checkpoint \
    --state_dict_file 6wxb/posterior_6wxb.estimator \
    --train_device cuda

train_npe_model \
    --image_config_file 6wxb/image_params_bending_training.json \
    --train_config_file 6wxb/resnet18_fft_encoder.json\
    --epochs 200 \
    --estimator_file 6wxb/posterior_6wxb_bending.estimator \
    --loss_file 6wxb/posterior_6wxb_bending.loss \
    --n_workers 24 \
    --train_from_checkpoint \
    --state_dict_file 6wxb/posterior_6wxb.estimator \
    --train_from_checkpoint \
    --state_dict_file 6wxb/posterior_6wxb.estimator \
    --train_device cuda

train_npe_model \
    --image_config_file 6wxb/image_params_torsiobend_training.json \
    --train_config_file 6wxb/resnet18_fft_encoder.json\
    --epochs 200 \
    --estimator_file 6wxb/posterior_6wxb_torsiobend.estimator \
    --loss_file 6wxb/posterior_6wxb_torsiobend.loss \
    --n_workers 24 \
    --train_from_checkpoint \
    --state_dict_file 6wxb/posterior_6wxb.estimator \
    --train_from_checkpoint \
    --state_dict_file 6wxb/posterior_6wxb.estimator \
    --train_device cuda