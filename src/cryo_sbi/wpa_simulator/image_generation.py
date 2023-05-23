import numpy as np
import torch


def gen_quat() -> np.ndarray:
    """
    Generate a random quaternion.

    Returns:
        quat (np.ndarray): Random quaternion

    """
    count = 0
    while count < 1:
        quat = np.random.uniform(
            -1, 1, 4
        )  # note this is a half-open interval, so 1 is not included but -1 is
        norm = np.sqrt(np.sum(quat**2))

        if 0.2 <= norm <= 1.0:
            quat /= norm
            count += 1

    return quat


@torch.jit.script
def project_density(
    coord: torch.Tensor,
    sigma: float,
    num_pxels: int,
    pixel_size: float
) -> torch.Tensor:
    """
    Generate a 2D projection from a set of coordinates.

    Args:
        coord (torch.Tensor): Coordinates of the atoms in the image
        sigma (float): Standard deviation of the Gaussian function used to model electron density.
        num_pxels (int): Number of pixels along one image size.
        pixel_size (float): Pixel size in Angstrom

    Returns:
        image (torch.Tensor): Image generated from the coordinates
    """

    num_atoms = coord.shape[1]
    norm = 1 / (2 * torch.pi * sigma**2 * num_atoms)

    grid_min = -pixel_size * (num_pxels - 1) * 0.5
    grid_max = pixel_size * (num_pxels - 1) * 0.5 + pixel_size

    grid = torch.arange(grid_min, grid_max, pixel_size)

    gauss_x = torch.exp_(-0.5 * (((grid[:, None] - coord[0, :]) / sigma) ** 2))

    gauss_y = torch.exp_(-0.5 * (((grid[:, None] - coord[1, :]) / sigma) ** 2))

    image = torch.matmul(gauss_x, gauss_y.T) * norm

    return image


def gen_img(coord: np.ndarray, image_params: dict) -> torch.Tensor:
    """
    Generate an image from a set of coordinates.

    Args:
        coord (np.ndarray): Coordinates of the atoms in the image
        image_params (dict): Dictionary containing the image parameters
            N_PIXELS (int): Number of pixels along one image size.
            PIXEL_SIZE (float): Pixel size in Angstrom
            SIGMA (float or list): Standard deviation of the Gaussian function used to model electron density.
            ELECWAVE (float): Electron wavelength in Angstrom

    Returns:
        image (torch.Tensor): Image generated from the coordinates
    """

    if isinstance(image_params["SIGMA"], float):
        atom_sigma = image_params["SIGMA"]

    elif isinstance(image_params["SIGMA"], list) and len(image_params["SIGMA"]) == 2:
        atom_sigma = np.random.uniform(
            low=image_params["SIGMA"][0], high=image_params["SIGMA"][1]
        )

    else:
        raise ValueError(
            "SIGMA should be a single value or a list of [min_sigma, max_sigma]"
        )

    image = project_density(
        torch.from_numpy(coord),
        atom_sigma,
        image_params["N_PIXELS"],
        image_params["PIXEL_SIZE"],
    )

    return image
