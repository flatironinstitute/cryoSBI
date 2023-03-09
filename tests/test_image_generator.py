import numpy as np
import torch
import cryo_em_sbi
import matplotlib.pyplot as plt

from cryo_em_sbi.simulating import image_generation
from cryo_em_sbi.utils import validate_config
import cryo_em_sbi.preprocessing as preprocessing

model = torch.tensor(np.load("../models/hsp90_models.npy")[0, 0])

config = {
    "IMAGES": {"N_PIXELS": 128, "PIXEL_SIZE": 1.0, "SIGMA": 2.0},
    "SIMULATION": {
        "N_SIMULATIONS": 100,
        "MODEL_FILE": "../../models/hsp90_models.npy",
        "DEVICE": "cpu",
        "ROTATIONS": "QUAT_576",
    },
    "PREPROCESSING": {
        "SHIFT": True,
        "CTF": True,
        "NOISE": True,
        "VOLTAGE": 300,
        "DEFOCUS": [1.5, 2.5],
        "SNR": 0.1,
        "DEVICE": "cpu",
        "REDUCED_PIXELS": 64,
    },
    "TRAINING": {
        "MODEL": "maf",
        "HIDDEN_FEATURES": 100,
        "NUM_TRANSFORMS": 4,
        "DEVICE": "cpu",
        "BATCH_SIZE": 1000,
        "POSTERIOR_NAME": "posterior_all_effects_noise_01.pkl",
    },
}

preproc_images = image_generation.gen_img(model, config["IMAGES"]).reshape(1, -1)


preproc_images = preprocessing.pad_dataset(
    preproc_images, config["IMAGES"], config["PREPROCESSING"]
)

image_ctf = preprocessing.apply_ctf_to_dataset(
    preproc_images.reshape(1, -1),
    config["IMAGES"],
    config["PREPROCESSING"],
)

preproc_images = preprocessing.shift_dataset(
    preproc_images, config["SIMULATION"], config["IMAGES"]
)

preproc_images = preprocessing.add_noise_to_dataset(
    preproc_images, config["PREPROCESSING"]
)

plt.imshow(preproc_images.reshape(128, 128))
plt.show()
