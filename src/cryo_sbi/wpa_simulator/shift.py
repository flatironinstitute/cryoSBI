import torch
import numpy as np


def apply_random_shift(padded_image, image_params, seed=None):
    if seed is not None:
        torch.manual_seed(seed)

    max_shift = int(np.round(image_params["N_PIXELS"] * 0.1))
    shift_x = int(torch.randint(low=-max_shift, high=max_shift + 1, size=(1,)))
    shift_y = int(torch.randint(low=-max_shift, high=max_shift + 1, size=(1,)))

    pad_width = int(np.ceil(image_params["N_PIXELS"] * 0.1)) + 1

    low_ind_x = pad_width - shift_x
    high_ind_x = padded_image.shape[0] - pad_width - shift_x

    low_ind_y = pad_width - shift_y
    high_ind_y = padded_image.shape[0] - pad_width - shift_y

    shifted_image = padded_image[low_ind_x:high_ind_x, low_ind_y:high_ind_y]

    return shifted_image


def apply_no_shift(padded_image, image_params):
    pad_width = int(np.ceil(image_params["N_PIXELS"] * 0.1)) + 1

    low_ind_x = pad_width
    high_ind_x = padded_image.shape[0] - pad_width

    low_ind_y = pad_width
    high_ind_y = padded_image.shape[0] - pad_width

    shifted_image = padded_image[low_ind_x:high_ind_x, low_ind_y:high_ind_y]

    return shifted_image
