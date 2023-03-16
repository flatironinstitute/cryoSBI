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


def shift_dataset(dataset, preproc_params, image_params):
    shifted_images = torch.empty(
        (dataset.shape[0], image_params["N_PIXELS"] ** 2),
        device=preproc_params["DEVICE"],
    )
    n_pixels = int(np.sqrt(dataset.shape[1]))

    for i in range(dataset.shape[0]):
        tmp_image = apply_random_shift(
            dataset[i].reshape(n_pixels, n_pixels), image_params
        )

        shifted_images[i] = tmp_image.reshape(1, -1).to(preproc_params["DEVICE"])

    return shifted_images
