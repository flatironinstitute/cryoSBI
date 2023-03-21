import numpy as np
import torch


def circular_mask(n_pixels, radius):
    grid = torch.linspace(-0.5 * (n_pixels - 1), 0.5 * (n_pixels - 1), n_pixels)
    r_2d = grid[None, :] ** 2 + grid[:, None] ** 2
    mask = r_2d < radius ** 2

    return mask


def add_noise(image, image_params, seed=None):
    if seed is not None:
        torch.manual_seed(seed)

    mask = circular_mask(n_pixels=image.shape[0], radius=image_params["RADIUS_MASK"])
    signal_power = image[mask].pow(2).mean().sqrt()

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


def add_shot_noise(image):
    """Adds shot noise to image"""
    pass


def add_colored_noise(image):
    """Adds colored noise to image"""
    pass


def add_gradient_snr(image, image_params, delta_snr=0.5):
    """Adds gaussian noise with gradient along x"""

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
