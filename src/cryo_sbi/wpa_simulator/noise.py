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
    mask = r_2d < radius ** 2

    return mask


def get_snr(images, snr):
    """
    Computes the SNR of the images
    """
    mask = circular_mask(
        n_pixels=images.shape[-1],
        radius=images.shape[-1] // 2,  # TODO: make this a parameter
        device=images.device,
    )
    signal_power = torch.std(
        images[:, mask], dim=[-1]
    )  # images are not centered at 0, so std is not the same as power
    assert signal_power.shape[0] == images.shape[0]
    noise_power = signal_power.reshape(-1, 1, 1) / torch.sqrt(torch.pow(torch.tensor(10), snr))
    print(torch.pow(snr, torch.tensor(10)))
    return noise_power


def add_noise(image: torch.Tensor, snr, seed=None) -> torch.Tensor:
    """
    Adds noise to image.

    Args:
        image (torch.Tensor): Image of shape (n_pixels, n_pixels).
        image_params (dict): Dictionary with image parameters.
        seed (int, optional): Seed for random number generator. Defaults to None.

    Returns:
        image_noise (torch.Tensor): Image with noise of shape (n_pixels, n_pixels) or (n_channels, n_pixels, n_pixels).
    """

    if seed is not None:
        torch.manual_seed(seed)

    noise_power = get_snr(image, snr)
    noise = torch.randn_like(image, device=image.device)
    print(noise.shape, noise_power.shape, image.shape, snr.shape)
    noise = noise * noise_power.reshape(-1, 1, 1)

    image_noise = image + noise

    return image_noise
