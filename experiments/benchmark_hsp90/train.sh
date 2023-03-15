python3 -m cryo_sbi.inference.NPE_train_without_saving \
    --image_config_file experiments/benchmark_hsp90/image_params_training.json \
    --train_config_file experiments/benchmark_hsp90/resnet18_encoder.json\
    --epochs 100 \
    --estimator_file experiments/benchmark_hsp90/posterior_hsp90.estimator \
    --loss_file experiments/benchmark_hsp90/posterior_hsp90.loss