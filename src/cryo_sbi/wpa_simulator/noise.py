from typing import Union
import numpy as np
import torch


def circular_mask(n_pixels: int, radius: int, device: str = "cpu") -> torch.Tensor:
    """
    Creates a circular mask of radius RADIUS_MASK centered in the image

    Args:
        n_pixels (int): Number of pixels along image side.
        radius (int): Radius of the mask.

    Returns:
        mask (torch.Tensor): Mask of shape (n_pixels, n_pixels).
    """

    grid = torch.linspace(
        -0.5 * (n_pixels - 1), 0.5 * (n_pixels - 1), n_pixels, device=device
    )
    r_2d = grid[None, :] ** 2 + grid[:, None] ** 2
    mask = r_2d < radius**2

    return mask


def get_snr(images, snr):
    """
    Computes the SNR of the images
    """

    mask = circular_mask(
        n_pixels=images.shape[-1],
        radius=64,  # todo: make this a parameter
        device=images.device,
    )
    signal_power = images[:, mask].pow(2).mean().sqrt()  # torch.std(image[mask])
    noise_power = signal_power / torch.sqrt(torch.pow(10, snr))

    return noise_power


def add_noise(image: torch.Tensor, snr, seed=None) -> torch.Tensor:
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

    snr = get_snr(image, snr)

    noise = torch.randn_like(image, device=image.device)
    noise = noise * snr.reshape(-1, 1, 1)

    image_noise = image + noise

    return image_noise
