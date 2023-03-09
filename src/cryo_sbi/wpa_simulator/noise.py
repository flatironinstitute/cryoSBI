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

    if len(np.asarray(image_params["SNR"]).reshape(-1)) == 1:
        snr = image_params["SNR"]

    elif len(np.asarray(image_params["SNR"]).reshape(-1)) == 2:
        snr = (
            np.random.rand()
            * (image_params["SNR"][1] - image_params["SNR"][0])
            + image_params["SNR"][0]
        )

    else:
        raise ValueError("SNR should be a single value or a list of [min_defocus, max_defocus]")

    noise_std = signal_std / np.sqrt(snr)

    img_noise = img + torch.distributions.normal.Normal(0, noise_std).sample(img.shape)

    return img_noise


def add_noise_to_dataset(dataset, preproc_params):
    images_with_noise = torch.empty_like(dataset, device=preproc_params["DEVICE"])
    n_pixels = int(np.sqrt(dataset.shape[1]))

    for i in range(dataset.shape[0]):
        tmp_image = add_noise(dataset[i].reshape(n_pixels, n_pixels), preproc_params)

        images_with_noise[i] = tmp_image.reshape(1, -1).to(preproc_params["DEVICE"])

    return images_with_noise
