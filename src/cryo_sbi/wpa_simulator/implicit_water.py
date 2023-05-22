import torch
import numpy as np


@torch.jit.script
def generate_noise_field(num_sin_func: int = 50):

    a = 1 * torch.rand((num_sin_func, 1), dtype=torch.float32)
    b = 0.15 * torch.randn((num_sin_func, 2), dtype=torch.float32)
    c = 2 * torch.pi * torch.rand((num_sin_func, 1), dtype=torch.float32)

    x = torch.linspace(0, 128 * 2.06, 128, dtype=torch.float32)
    y = torch.linspace(0, 128 * 2.06, 128, dtype=torch.float32)
    xx, yy = torch.meshgrid(x, y, indexing="ij")

    xx = xx.unsqueeze(0).repeat(num_sin_func, 1, 1)
    yy = yy.unsqueeze(0).repeat(num_sin_func, 1, 1)
  
    sin_funcs = a[:, :, None] * torch.sin(b[:, 0, None, None] * xx + b[:, 1, None, None] * yy + c[:, 0, None, None])
    noise_field = torch.sum(sin_funcs, dim=0)[0]

    return noise_field


def gen_noise_field(image_params: dict, num_sin_func: int = 50) -> torch.Tensor:
    """
    Generate a noise field with a given number of sinusoidal functions.

    Args:
        num_pixels (int): Number of pixels in the noise field.
        num_sin_func (int, optional): Number of sinusoidal functions. Defaults to 10.
        max_intensity (float, optional): Maximum intensity of the noise field. Defaults to 1e-3.

    Returns:
        torch.Tensor: Noise field.
    """


    x = torch.linspace(0, image_params["N_PIXELS"] * image_params["PIXEL_SIZE"], image_params["N_PIXELS"])
    y = torch.linspace(0, image_params["N_PIXELS"] * image_params["PIXEL_SIZE"], image_params["N_PIXELS"])

    max_freq = np.pi / (image_params["PIXEL_SIZE"])
    min_freq = 2 * np.pi / (image_params["N_PIXELS"] * image_params["PIXEL_SIZE"])
    exp_coeff = (max_freq - min_freq) / 20
    xx, yy = torch.meshgrid(x, y, indexing="ij")

    a = 1 * torch.rand(num_sin_func, 1)
    b = 0.15 * torch.randn((num_sin_func, 2)) #torch.from_numpy(np.random.exponential(exp_coeff, size=(num_sin_func, 2))) + min_freq
    c = 2 * torch.pi * (torch.rand(num_sin_func, 1))

    noise_field = torch.zeros_like(xx, dtype=torch.double)
    for i in range(num_sin_func):
        noise_field += (
            a[i] * torch.sin(b[i, 0] * xx + b[i, 1] * yy + c[i, 0])
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
    image += (noise_field * torch.rand(1) * image_params["NOISE_INTENSITY"])

    return image
