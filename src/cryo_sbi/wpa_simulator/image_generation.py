import numpy as np
import torch


def gen_quat():
    count = 0
    while count < 1:
        quat = np.random.uniform(
            -1, 1, 4
        )  # note this is a half-open interval, so 1 is not included but -1 is
        norm = np.sqrt(np.sum(quat ** 2))

        if 0.2 <= norm <= 1.0:
            quat /= norm
            count += 1

    return quat


def gen_img(coord, image_params):
    n_atoms = coord.shape[1]
    norm = 1 / (2 * torch.pi * image_params["SIGMA"] ** 2 * n_atoms)

    grid_min = -image_params["PIXEL_SIZE"] * (image_params["N_PIXELS"] - 1) * 0.5
    grid_max = (
        image_params["PIXEL_SIZE"] * (image_params["N_PIXELS"] - 1) * 0.5
        + image_params["PIXEL_SIZE"]
    )

    grid = torch.arange(grid_min, grid_max, image_params["PIXEL_SIZE"])

    gauss_x = torch.exp(
        -0.5 * (((grid[:, None] - coord[0, :]) / image_params["SIGMA"]) ** 2)
    )

    gauss_y = torch.exp(
        -0.5 * (((grid[:, None] - coord[1, :]) / image_params["SIGMA"]) ** 2)
    )

    image = torch.matmul(gauss_x, gauss_y.T) * norm

    return image
