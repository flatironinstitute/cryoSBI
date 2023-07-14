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


def gen_rot_matrix(quats: torch.Tensor) -> torch.Tensor:
    # TODO add docstring explaining the quaternion convention qr, qx, qy, qz
    """
    Generate a rotation matrix from a quaternion.

    Args:
        quat (torch.Tensor): Quaternion (n_batch, 4)

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

    return rot_matrix


def project_density(
    coords: torch.Tensor,
    quats: torch.Tensor,
    sigma: torch.Tensor,
    shift: torch.Tensor,
    num_pixels: int,
    pixel_size: float,
) -> torch.Tensor:
    """
    Generate a 2D projections from a set of coordinates.

    Args:
        coords (torch.Tensor): Coordinates of the atoms in the images
        sigma (float): Standard deviation of the Gaussian function used to model electron density.
        num_pixels (int): Number of pixels along one image size.
        pixel_size (float): Pixel size in Angstrom

    Returns:
        image (torch.Tensor): Images generated from the coordinates
    """

    num_batch, _, num_atoms = coords.shape
    norm = 1 / (2 * torch.pi * sigma**2 * num_atoms)

    grid_min = -pixel_size * num_pixels * 0.5
    grid_max = pixel_size * num_pixels * 0.5

    rot_matrix = gen_rot_matrix(quats)
    grid = torch.arange(grid_min, grid_max, pixel_size, device=coords.device)[0:num_pixels.long()].repeat(
        num_batch, 1
    ) # [0: num_pixels.long()] is needed due to single precision error in some cases
 
    coords_rot = torch.bmm(rot_matrix, coords)
    coords_rot[:, :2, :] += shift.unsqueeze(-1)

    gauss_x = torch.exp_(
        -0.5 * (((grid.unsqueeze(-1) - coords_rot[:, 0, :].unsqueeze(1)) / sigma) ** 2)
    )
    gauss_y = torch.exp_(
        -0.5 * (((grid.unsqueeze(-1) - coords_rot[:, 1, :].unsqueeze(1)) / sigma) ** 2)
    ).transpose(1, 2)

    image = torch.bmm(gauss_x, gauss_y) * norm.reshape(-1, 1, 1)

    return image
