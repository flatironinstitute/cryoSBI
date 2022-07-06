import numpy as np


def gen_quat():
    # Sonya's code
    # Generates a single quaternion

    count = 0
    while count < 1:

        quat = np.random.uniform(
            -1, 1, 4
        )  # note this is a half-open interval, so 1 is not included but -1 is
        norm = 1 / np.sqrt(np.sum(quat**2))

        if 0.2 <= norm <= 1.0:
            quat *= norm
            count += 1

    return quat


def gen_img(coord, image_params):

    n_atoms = coord.shape[1]
    norm = 1 / (2 * np.pi * image_params["SIGMA"] ** 2 * n_atoms)

    grid_min = -image_params["PIXEL_SIZE"] * (image_params["N_PIXELS"] - 1) * 0.5
    grid_max = (
        image_params["PIXEL_SIZE"] * (image_params["N_PIXELS"] - 1) * 0.5
        + image_params["PIXEL_SIZE"]
    )

    grid = np.arange(grid_min, grid_max, image_params["PIXEL_SIZE"])

    gauss = np.exp(
        -0.5 * (((grid[:, None] - coord[0, :]) / image_params["SIGMA"]) ** 2)
    )[:, None] * np.exp(
        -0.5 * (((grid[:, None] - coord[1, :]) / image_params["SIGMA"]) ** 2)
    )

    image = gauss.sum(axis=2) * norm

    return image
