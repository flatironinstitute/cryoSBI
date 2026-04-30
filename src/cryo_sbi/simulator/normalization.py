import torch
import torchvision.transforms as transforms


def gaussian_normalize_image(images: torch.Tensor) -> torch.Tensor:
    """
    Normalize an images by subtracting the mean and dividing by the standard deviation.

    Args:
        images (torch.Tensor): Image of shape (n_pixels, n_pixels) or (n_channels, n_pixels, n_pixels).

    Returns:
        normalized (torch.Tensor): Normalized image.
    """

    mean = images.mean(dim=[1, 2])
    std = images.std(dim=[1, 2]).clamp_min(torch.finfo(images.dtype).eps)

    return transforms.functional.normalize(images, mean=mean, std=std)
