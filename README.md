# cryo_em_SBI


## Collaborators


David Silva-Sánchez, Lars Dingeldein, Roberto Covino, Pilar Cossio

# Dependencies

1. [SBI - Mackelab](https://www.mackelab.org/sbi/install/) (and all their dependencies)
2. SciPy
3. NumPy
4. Pickle
5. json

# Installing

## Download this repository

`https://github.com/DSilva27/cryo_em_SBI.git`

## Install the module
`python3 setup.py install`

# Using the code


## What can you find in this repo?

1. `cryo_em_sbi`: module for doing sbi with CryoEM data
2. `Tutorials`: learning how to run the code with two toy models, a simple square and the HSP90
3. `models`: .npy files containing a 20x20 grid of models. For now only for HSP90 and the square-like system.
4. `production`: a folder where everything you create is untracked by git.
5. `README.md`: that's me!


## The config file

The config file is simply a json file where you will write the parameters to be used in both generating data and training. It has three main categories called `IMAGES`, `SIMULATION`, `PREPROCESSING`, and `TRAINING`.

### IMAGES

Here you need to set up the parameters to generate the images from 3D atomic coordinates

1. PIXEL_SIZE: the size of the pixel in Angstroms
2. N_PIXELS: Number of pixels the image will have in one dimension, i.e, the image is N_PIXELS * N_PIXELS.
3. SIGMA: every atom in the atomic model will be represented as a gaussian centered around its center of mass. This parameter is the standard deviation of that gaussian.


### SIMULATION

1. N_SIMULATIONS: the number of images to generate.
2. MODEL_FILE: the name of the file that contains all the atomics coordinates (check the folder `models`)
3. DEVICE: either `"cpu"` or `"cuda"`. While for running with with a gpu you can simply write `"cuda"`, in some cases it can be a bit more complicated. Check recommendations at the end of the README.
4. ROTATIONS: wheter or not to generate images with different orientations (true or false)

### PREPROCESSING

1. SHIFT: wheter or not to randomly shift images (true or false)
2. CTF: wheter or not to apply CTF effects to images (true or false)
3. NOISE: wheter or not to add CTF effects to images (true or false)
4. DEFOCUS: value fo the defocus for the CTF
5. SNR: The signal-to-noise ratio used to generate the images. Based on https://www.biorxiv.org/content/10.1101/864116v1.

### TRAINING

Parameters used to train an SNPE network using an `"maf"` model.

1. MODEL: model for the neural network (check the SBI documentation)
2. HIDDEN_FEATURES: number of hidden features in the network.
3. NUM_TRANSFORMS: number of transforms in the network.
4. DEVICE: either `"cpu"` or `"cuda"`. While for running with with a gpu you can simply write `"cuda"`, in some cases it can be a bit more complicated. Check recommendations at the end of the README.
5. POSTERIOR_NAME: the trained posteriors are pickled to use for post-processing. This is simply the name of that file. This is the only optional parameter, if you don't provide it, the name is simply "posterior.pkl".


## Prepare for simulating/training

I conveniently created a folder called `production` where nothing will be tracked, i.e, seen by git. You can create all the folders and whatever you want in there. I will refer to the folder where you will be working as your `working directory`.

1. Copy the files from the tutorial you want to reproduce to your `working directory`
2. Check the example config files in `example_config` and copy the one that suits you the most over to your `working directoy`.
3. Move to your `working directoy`.
4. Load modules and activate virtual environment (if running locally on a workstation with slurm)

```
ml python
ml gcc/7
source /mnt/home/dsilvasanchez/virtual_envs/sbi_env_try/bin/activate
```

## Simulating, Preprocessing, and Training

The tutorials have everything you need to do these actions with or without SLURM


## Some tips for running with a GPU

I still have to learn more, but here are a few tips.

1. First check that pytorch actually recognizes your GPU

```python
>>> import torch
>>> torch.cuda.is_available() # Check if pytorch sees your GPU
True # or False

>>> torch.cuda.current_device() # Check the index of your GPU
0 # Or other integer

>>> torch.cuda.get_device_name(0)
'Name of your GPU'
```

2. If `torch.cuda.current_device()` returns `0` in your config file you should `"DEVICE"` to `"cuda:0"`.

3. If you have multiple GPUs you can choose which one to use using `"cuda:identifier_of_your_gpu"`. For example if you have two GPUs identified by `0` and `1` and you want to use `1`, then you should write `"cuda:1"`. If you wanted to use both GPUs you should write `"cuda:{0, 1}`.

