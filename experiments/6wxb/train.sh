python3 -m cryo_sbi.inference.NPE_train_without_saving \
    --image_config_file 6wxb/image_params_mixed_training.json \
    --train_config_file 6wxb/resnet18_fft_encoder.json\
    --epochs 200 \
    --estimator_file 6wxb/6wxb_mixed_posterior.estimator \
    --loss_file 6wxb/6wxb_mixed_posterior.loss \
    --n_workers 24 \
    --train_from_checkpoint \
    --state_dict_file 6wxb/posterior_torsion.estimator

python3 -m cryo_sbi.inference.NPE_train_without_saving \
    --image_config_file 6wxb/image_params_mixed_training.json \
    --train_config_file 6wxb/resnet18_fft_noise_encoder.json\
    --epochs 200 \
    --estimator_file 6wxb/6wxb_mixed_posterior_noise.estimator \
    --loss_file 6wxb/6wxb_mixed_posterior_noise.loss \
    --n_workers 24 \
    --train_from_checkpoint \
    --state_dict_file 6wxb/6wxb_mixed_posterior.estimator