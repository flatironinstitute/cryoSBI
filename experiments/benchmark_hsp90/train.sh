cd ../../../scr/inference/

python NPE_train_without_saving.py \
    --image_config_file ../../../experiments/benchmark_hsp90_real/image_params_snr01_128.json \
    --train_config_file ../../../experiments/benchmark_hsp90_real/wideres50_encoder.json \
    --epochs 350 \
    --estimator_file ../../../experiments/benchmark_hsp90_real/wideres50_encoder_snr01.estimator \
    --loss_file ../../../experiments/benchmark_hsp90_real/wideres50_encoder_snr01.loss
