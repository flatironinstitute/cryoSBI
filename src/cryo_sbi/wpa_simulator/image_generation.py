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
        norm = torch.sqrt(torch.sum(quat ** 2))
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
    atomic_model: torch.Tensor,
    quats: torch.Tensor,
    res: torch.Tensor,
    shift: torch.Tensor,
    num_pixels: int,
    pixel_size: float,
) -> torch.Tensor:
    """
    Generate a 2D projections from a set of coordinates.

    Args:
        atomic_model (torch.Tensor): Coordinates of the atoms in the images
        res (float): resolution of the images in Angstrom
        num_pixels (int): Number of pixels along one image size.
        pixel_size (float): Pixel size in Angstrom

    Returns:
        image (torch.Tensor): Images generated from the coordinates
    """

    num_batch, _, num_atoms = atomic_model.shape

    variances = atomic_model[:, 4, :] * res[:, 0] ** 2
    amplitudes = atomic_model[:, 3, :] / torch.sqrt((2 * torch.pi * variances))

    grid_min = -pixel_size * num_pixels * 0.5
    grid_max = pixel_size * num_pixels * 0.5

    rot_matrix = gen_rot_matrix(quats)
    grid = torch.arange(grid_min, grid_max, pixel_size, device=atomic_model.device)[
        0 : num_pixels.long()
    ].repeat(
        num_batch, 1
    )  # [0: num_pixels.long()] is needed due to single precision error in some cases

    coords_rot = torch.bmm(rot_matrix, atomic_model[:, :3, :])
    coords_rot[:, :2, :] += shift.unsqueeze(-1)

    gauss_x = torch.exp_(
        -((grid.unsqueeze(-1) - coords_rot[:, 0, :].unsqueeze(1)) ** 2)
        / variances.unsqueeze(1)
    ) * amplitudes.unsqueeze(1)

    gauss_y = torch.exp(
        -((grid.unsqueeze(-1) - coords_rot[:, 1, :].unsqueeze(1)) ** 2)
        / variances.unsqueeze(1)
    ) * amplitudes.unsqueeze(1)

    image = torch.bmm(gauss_x, gauss_y.transpose(1, 2))  # * norms
    image /= torch.norm(image, dim=[-2, -1]).reshape(-1, 1, 1)

    return image
