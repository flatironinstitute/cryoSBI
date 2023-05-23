import torch
import numpy as np


@torch.jit.script
def generate_noise_field(
    pixel_size: float, num_pixels: int, sigma: float = 0.15, num_sin_func: int = 50
):
    """
    Generate a noise field with a given number of sinusoidal functions.

    Args:
        pixel_size (float): Pixel size in Angstrom.
        num_pixels (int): Number of pixels in image.
        sigma (float, optional): Sigma of distribution for sinus frequencies. Defaults to 0.15.
        num_sin_func (int, optional): Number of sinusoidal functions. Defaults to 50.

    Returns:
        torch.Tensor: Noise field.
    """

    a = 1 * torch.rand((num_sin_func, 1), dtype=torch.float32)
    b = sigma * torch.randn((num_sin_func, 2), dtype=torch.float32)
    c = 2 * torch.pi * torch.rand((num_sin_func, 1), dtype=torch.float32)

    x = torch.linspace(0, num_pixels * pixel_size, num_pixels, dtype=torch.float32)
    y = torch.linspace(0, num_pixels * pixel_size, num_pixels, dtype=torch.float32)
    xx, yy = torch.meshgrid(x, y, indexing="ij")

    xx = xx.unsqueeze(0).repeat(num_sin_func, 1, 1)
    yy = yy.unsqueeze(0).repeat(num_sin_func, 1, 1)

    sin_funcs = a[:, :, None] * torch.sin(
        b[:, 0, None, None] * xx + b[:, 1, None, None] * yy + c[:, 0, None, None]
    )
    noise_field = torch.sum(sin_funcs, dim=0)

    return noise_field


def gen_noise_field(image_params: dict, num_sin_func: int = 50) -> torch.Tensor:
    """
    Generate a noise field with a given number of sinusoidal functions.

    Args:
        image_params (dict): Dictionary containing the image parameters.
        num_sin_func (int, optional): Number of sinusoidal functions. Defaults to 50.

    Returns:
        torch.Tensor: Noise field.
    """

    # Attention look into def pad_image function to know the image size after padding
    image_size = (
        2 * (int(np.ceil(image_params["N_PIXELS"] * 0.1)) + 1)
        + image_params["N_PIXELS"]
    )

    noise_field = generate_noise_field(
        pixel_size=image_params["PIXEL_SIZE"],
        num_pixels=image_size,
        sigma=0.1,
        num_sin_func=num_sin_func,
    )

    return noise_field


def add_noise_field(image: torch.Tensor, image_params: dict) -> torch.Tensor:
    """
    Add a noise field to an image.

    Args:
        image (torch.Tensor): Image of shape (n_pixels, n_pixels) or (n_channels, n_pixels, n_pixels).
        min_intensity (float): Minimum intensity of the image.

    Returns:
        torch.Tensor: Image with noise field.
    """

    noise_field = gen_noise_field(image_params, num_sin_func=200)
    noise_field = (noise_field / noise_field.max()) * image.max()
    image += noise_field * torch.rand(1) * image_params["NOISE_INTENSITY"]

    return image
