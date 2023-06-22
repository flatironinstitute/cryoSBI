import numpy as np
import torch


def gen_quat() -> torch.Tensor:
    """
    Generate a random quaternion.

    Returns:
        quat (np.ndarray): Random quaternion

    """
    count = 0
    while count < 1:
        quat = 2 * torch.rand(size=(4,)) - 1
        norm = torch.sqrt(torch.sum(quat**2))
        if 0.2 <= norm <= 1.0:
            quat /= norm
            count += 1

    return quat


def gen_rot_matrix_batched(quats: torch.Tensor) -> torch.Tensor:
    """
    Generate a rotation matrix from a quaternion.

    Args:
        quat (torch.Tensor): Quaternion

    Returns:
        rot_matrix (torch.Tensor): Rotation matrix
    """

    rot_matrix = torch.zeros((quats.shape[0], 3, 3), device=quats.device)

    rot_matrix[:, 0, 0] = 1 - 2 * (quats[:, 2] ** 2 + quats[:, 3] ** 2)
    rot_matrix[:, 0, 1] = 2 * (quats[:, 1] * quats[:, 2] - quats[:, 3] * quats[:, 0])
    rot_matrix[:, 0, 2] = 2 * (quats[:, 1] * quats[:, 3] + quats[:, 2] * quats[:, 0])

    rot_matrix[:, 1, 0] = 2 * (quats[:, 1] * quats[:, 2] + quats[:, 3] * quats[:, 0])
    rot_matrix[:, 1, 1] = 1 - 2 * (quats[:, 1] ** 2 + quats[:, 3] ** 2)
    rot_matrix[:, 1, 2] = 2 * (quats[:, 2] * quats[:, 3] - quats[:, 1] * quats[:, 0])

    rot_matrix[:, 2, 0] = 2 * (quats[:, 1] * quats[:, 3] - quats[:, 2] * quats[:, 0])
    rot_matrix[:, 2, 1] = 2 * (quats[:, 2] * quats[:, 3] + quats[:, 1] * quats[:, 0])
    rot_matrix[:, 2, 2] = 1 - 2 * (quats[:, 1] ** 2 + quats[:, 2] ** 2)

    return -rot_matrix


def project_density_batched(
    coords: torch.Tensor,
    quats: torch.Tensor,
    sigma: torch.Tensor,
    num_pxels: int,
    pixel_size: float,
) -> torch.Tensor:
    """
    Generate a 2D projections from a set of coordinates.

    Args:
        coords (torch.Tensor): Coordinates of the atoms in the images
        sigma (float): Standard deviation of the Gaussian function used to model electron density.
        num_pxels (int): Number of pixels along one image size.
        pixel_size (float): Pixel size in Angstrom

    Returns:
        image (torch.Tensor): Images generated from the coordinates
    """

    num_batch, _, num_atoms = coords.shape
    norm = 1 / (2 * torch.pi * sigma**2 * num_atoms)

    grid_min = -pixel_size * (num_pxels - 1) * 0.5
    grid_max = pixel_size * (num_pxels - 1) * 0.5 + pixel_size

    rot_matrix = gen_rot_matrix_batched(quats)
    grid = torch.arange(grid_min, grid_max, pixel_size, device=coords.device)
    gauss_x = torch.zeros((num_batch, num_pxels, num_atoms), device=coords.device)
    gauss_y = torch.zeros((num_batch, num_atoms, num_pxels), device=coords.device)

    for i in range(num_batch):
        coords_rot = torch.matmul(rot_matrix[i], coords[i])
        gauss_x[i] = torch.exp_(
            -0.5 * (((grid[:, None] - coords_rot[0, :]) / sigma[i]) ** 2)
        )
        gauss_y[i] = torch.exp_(
            -0.5 * (((grid[:, None] - coords_rot[1, :]) / sigma[i]) ** 2)
        ).T

    image = torch.bmm(gauss_x, gauss_y) * norm.reshape(-1, 1, 1)

    return image


def gen_img(coord: torch.Tensor, quaternions: torch.Tensor, image_params: dict) -> torch.Tensor:
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
    batched = False if coord.ndim == 2 else True

    if isinstance(image_params["SIGMA"], float):
        atom_sigma = (
            image_params["SIGMA"]
            if not batched
            else image_params["SIGMA"] * torch.ones(coord.shape[0])
        )

    elif isinstance(image_params["SIGMA"], list) and len(image_params["SIGMA"]) == 2:
        size = (1,) if not batched else (coord.shape[0], 1)
        atom_sigma = torch.from_numpy(
            np.random.uniform(
                low=image_params["SIGMA"][0],
                high=image_params["SIGMA"][1],
                size=size,
            )
        )

    else:
        raise ValueError(
            "SIGMA should be a single value or a list of [min_sigma, max_sigma]"
        )

    image = project_density_batched(
        coord,
        quaternions,
        atom_sigma.to(coord.device),
        image_params["N_PIXELS"],
        image_params["PIXEL_SIZE"],
    )

    return image
