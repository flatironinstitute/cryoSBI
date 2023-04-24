import torch


def gen_noise_field(num_pixels, num_sin_func=10, max_intensity=1e-3):
    """Generate a noise field with a given number of sinusoidal functions.

    Args:
        num_pixels (int): Number of pixels in the noise field.
        num_sin_func (int, optional): Number of sinusoidal functions. Defaults to 10.
        max_intensity (float, optional): Maximum intensity of the noise field. Defaults to 1e-3.

    Returns:
        torch.Tensor: Noise field.
    """

    x = torch.linspace(-100, 100, num_pixels)
    y = torch.linspace(-100, 100, num_pixels)
    xx, yy = torch.meshgrid(x, y)

    b = 0.6 * (torch.rand((num_sin_func, 2)) - 0.5)
    c = 2 * torch.pi * (torch.rand(num_sin_func, 2) - 0.5)

    noise_field = torch.zeros_like(xx, dtype=torch.double)
    for i in range(num_sin_func):
        noise_field += torch.sin(b[i, 0] * xx + c[i, 0]) * torch.sin(
            b[i, 1] * yy + c[i, 1]
        )
    noise_field = max_intensity * (noise_field / noise_field.max())
    return noise_field


def add_noise_field(image, min_intensity):
    """Add a noise field to an image.

    Args:
        image (torch.Tensor): Image of shape (n_pixels, n_pixels) or (n_channels, n_pixels, n_pixels).
        min_intensity (float): Minimum intensity of the image.

    Returns:
        torch.Tensor: Image with noise field.
    """

    noise_field = gen_noise_field(image.shape[0], max_intensity=1e-12)
    idx_replace = image < min_intensity
    image[idx_replace] = noise_field[idx_replace]

    return image
