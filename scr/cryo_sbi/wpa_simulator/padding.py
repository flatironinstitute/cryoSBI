import numpy as np
import torch
from torch.nn.functional import pad
from torch.nn import ConstantPad2d


def pad_image(image, image_params):
    pad_width = int(np.ceil(image_params["N_PIXELS"] * 0.1)) + 1

    padder = ConstantPad2d(pad_width, 0.0)

    padded_image = padder(image)

    return padded_image


### Preprocessing functions for datasets ###
def pad_dataset(dataset, image_params, preproc_params):
    images = dataset.reshape(
        dataset.shape[0], image_params["N_PIXELS"], image_params["N_PIXELS"]
    )

    pad_width = int(np.ceil(image_params["N_PIXELS"] * 0.1)) + 1

    padded_images = pad(
        images, (pad_width, pad_width, pad_width, pad_width), "constant", 0.0
    )

    return padded_images.reshape(dataset.shape[0], padded_images.shape[1] ** 2).to(preproc_params["DEVICE"])
