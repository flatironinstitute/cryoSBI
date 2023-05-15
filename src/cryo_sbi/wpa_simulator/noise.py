from typing import Union
import numpy as np
import torch


def circular_mask(n_pixels: int, radius: int) -> torch.Tensor:
    """
    Creates a circular mask of radius RADIUS_MASK centered in the image

    Args:
        n_pixels (int): Number of pixels along image side.
        radius (int): Radius of the mask.

    Returns:
        mask (torch.Tensor): Mask of shape (n_pixels, n_pixels).
    """

    grid = torch.linspace(-0.5 * (n_pixels - 1), 0.5 * (n_pixels - 1), n_pixels)
    r_2d = grid[None, :] ** 2 + grid[:, None] ** 2
    mask = r_2d < radius**2

    return mask


def add_noise(
    image: torch.Tensor, image_params: dict, seed: Union[None, int] = None
) -> torch.Tensor:
    """
    Adds noise to image

    Args:
        image (torch.Tensor): Image of shape (n_pixels, n_pixels).
        image_params (dict): Dictionary with image parameters.
        seed (int, optional): Seed for random number generator. Defaults to None.

    Returns:
        image_noise (torch.Tensor): Image with noise of shape (n_pixels, n_pixels) or (n_channels, n_pixels, n_pixels).
    """

    if seed is not None:
        torch.manual_seed(seed)

    mask = circular_mask(n_pixels=image.shape[0], radius=image_params["RADIUS_MASK"])
    signal_power = image[mask].pow(2).mean().sqrt()  # torch.std(image[mask])

    if isinstance(image_params["SNR"], float):
        snr = image_params["SNR"]

    elif isinstance(image_params["SNR"], list) and len(image_params["SNR"]) == 2:
        snr = 10 ** np.random.uniform(
            low=np.log10(image_params["SNR"][0]), high=np.log10(image_params["SNR"][1])
        )

    else:
        raise ValueError("SNR should be a single value or a list of [min_snr, max_snr]")

    noise_power = signal_power / np.sqrt(snr)
    image_noise = image + torch.distributions.normal.Normal(0, noise_power).sample(
        image.shape
    )

    return image_noise


def add_colored_noise(
    image: torch.Tensor,
    image_params: dict,
    seed: int,
    noise_intensity: float = 1,
    noise_scale: float = 1.5,
):
    """Adds colored noise to image.

    Args:
        image (torch.Tensor): Image of shape (n_pixels, n_pixels).
        image_params (dict): Dictionary with image parameters.
        seed (int, optional): Seed for random number generator. Defaults to None.
        noise_intensity (float, optional): Noise intensity. Defaults to 1.
        noise_scale (float, optional): Noise scale. Defaults to 1.5.

    Returns:
        image_noise (torch.Tensor): Image with noise of shape (n_pixels, n_pixels) or (n_channels, n_pixels, n_pixels).
    """

    # Similar to pink noise https://en.wikipedia.org/wiki/Pink_noise
    if seed is not None:
        torch.manual_seed(seed)

    image_L = image.shape[0]

    mask = circular_mask(n_pixels=image.shape[0], radius=image_params["RADIUS_MASK"])

    signal_std = image[mask].pow(2).mean().sqrt()
    noise_std = signal_std / np.sqrt(image_params["SNR"])

    image_noise = torch.distributions.normal.Normal(0, noise_std).sample(image.shape)
    fft_noise = torch.fft.fft2(image_noise)

    along_x, along_y = np.linspace(-1, 1, image_L), np.linspace(-1, 1, image_L)
    mesh_x, mesh_y = np.meshgrid(along_x, along_y)
    f = torch.zeros((image_L, image_L))

    for ix in range(image_L):
        for iy in range(image_L):
            f[ix, iy] = (
                np.abs(mesh_x[ix, iy]) ** noise_scale
                + np.abs(mesh_y[ix, iy]) ** noise_scale
            )

    t = torch.abs(torch.fft.ifft2(fft_noise / f))

    # Scaling with respect to the lenght max to median
    scale = noise_intensity / (t.max() - t.median())

    # Adjusting noise so that 50% of the pixels have higer and the other 50% lower snr
    t = ((t - t.median()) * scale) + 1

    image_noise = torch.distributions.normal.Normal(0, noise_std * t).sample()
    return image_noise + image


def add_shot_noise(image: torch.Tensor) -> torch.Tensor:
    """Adds shot noise to image"""
    raise NotImplementedError


def add_gradient_snr(
    image: torch.Tensor,
    image_params: dict,
    seed: Union[None, int] = None,
    delta_snr: float = 0.5,
) -> torch.Tensor:
    """Adds gaussian noise with gradient along x.

    Args:
        image (torch.Tensor): Image of shape (n_pixels, n_pixels).
        image_params (dict): Dictionary with image parameters.
        seed (int, optional): Seed for random number generator. Defaults to None.
        delta_snr (float, optional): Delta SNR. Defaults to 0.5.

    Returns:
        image_noise (torch.Tensor): Image with noise of shape (n_pixels, n_pixels) or (n_channels, n_pixels, n_pixels).
    """

    if seed is not None:
        torch.manual_seed(seed)

    mask = circular_mask(n_pixels=image.shape[0], radius=image_params["RADIUS_MASK"])
    signal_power = image[mask].pow(2).mean().sqrt()
    gradient_snr = np.logspace(
        np.log10(image_params["SNR"]) + delta_snr,
        np.log10(image_params["SNR"]) - delta_snr,
        image.shape[0],
    )

    noise = torch.stack(
        [
            torch.distributions.normal.Normal(0, signal_power / np.sqrt(snr)).sample(
                [
                    image.shape[0],
                ]
            )
            for snr in gradient_snr
        ],
        dim=1,
    )

    image_noise = image + noise
    return image_noise
