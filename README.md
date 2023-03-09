# cryo_em_SBI


## Collaborators


David Silva-Sánchez, Lars Dingeldein, Roberto Covino, Pilar Cossio

# Dependencies

1. Lampe
2. SciPy
3. NumPy
4. PyTorch
5. json

# Installing

## Download this repository

`https://github.com/DSilva27/cryo_em_SBI.git`

## Install the module
`python3 -m pip install -e .`

# Using the code


## Generating data from command line

```bash
python3 -m cryo_sbi.generate_training_set --config_file image_params_snr01_128.json --num_train_samples 10 --num_val_samples 10 --file_name "test1" --save_as_tensor false
```