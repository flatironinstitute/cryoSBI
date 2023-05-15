import numpy as np
import torch


def gaussian_normalize_image(image: torch.Tensor) -> torch.Tensor:
    """
    Normalize an image by subtracting the mean and dividing by the standard deviation.

    Args:
        image (torch.Tensor): Image of shape (n_pixels, n_pixels) or (n_channels, n_pixels, n_pixels).

    Returns:
        normalized (torch.Tensor): Normalized image.
    """

    mean_img = torch.mean(image)
    std_img = torch.std(image)

    return (image - mean_img) / std_img
