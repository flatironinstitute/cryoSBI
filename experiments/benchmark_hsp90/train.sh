python3 -m cryo_sbi.inference.NPE_train_without_saving \
    --image_config_file experiments/benchmark_hsp90/image_params_training.json \
    --train_config_file experiments/benchmark_hsp90/resnet18_encoder.json\
    --epochs 150 \
    --estimator_file experiments/benchmark_hsp90/posterior_hsp90_ex.estimator \
    --loss_file experiments/benchmark_hsp90/posterior_hsp90_ex.loss \
    --train_from_checkpoint \
    --state_dict_file experiments/benchmark_hsp90/posterior_hsp90.estimator
