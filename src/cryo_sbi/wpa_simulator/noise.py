import numpy as np
import torch


def circular_mask(n_pixels, radius):
    grid = torch.linspace(-0.5 * (n_pixels - 1), 0.5 * (n_pixels - 1), n_pixels)
    r_2d = grid[None, :] ** 2 + grid[:, None] ** 2
    mask = r_2d < radius**2

    return mask


def add_noise(img, image_params):
    mask = circular_mask(n_pixels=img.shape[0], radius=image_params["RADIUS_MASK"])

    signal_std = img[mask].pow(2).mean().sqrt()

    if isinstance(image_params["SNR"], float):
        snr = image_params["SNR"]

    elif isinstance(image_params["SNR"], list) and len(image_params["SNR"]) == 2:
        snr = np.random.uniform(low=image_params["SNR"][0], high=image_params["SNR"][1])

    else:
        raise ValueError("SNR should be a single value or a list of [min_defocus, max_defocus]")

    noise_std = signal_std / np.sqrt(snr)

    img_noise = img + torch.distributions.normal.Normal(0, noise_std).sample(img.shape)

    return img_noise


def add_shot_noise(img):
    """Adds shot noise to image"""
    pass


def add_colored_noise(img):
    """Adds colored noise to image"""
    pass


def add_gradient_snr(img, image_params, delta_snr=0.5):
    """Adds gaussian noise with gradient along x"""

    mask = circular_mask(n_pixels=img.shape[0], radius=image_params["RADIUS_MASK"])
    signal_std = img[mask].pow(2).mean().sqrt()
    gradient_snr = np.logspace(
        np.log10(image_params["SNR"]) + delta_snr, np.log10(image_params["SNR"]) - delta_snr, img.shape[0]
    )

    noise = torch.stack(
        [
            torch.distributions.normal.Normal(0, signal_std / np.sqrt(snr)).sample([img.shape[0],]) 
            for snr in gradient_snr
        ],
        dim=1
    )
    
    img_noise = img + noise
    return img_noise
