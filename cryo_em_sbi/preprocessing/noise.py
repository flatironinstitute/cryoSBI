import numpy as np
import torch


def add_noise(img, preproc_params):

    mean_image = torch.mean(img)
    std_image = torch.std(img)

    mask = torch.logical_or(
        img >= mean_image + 0.5 * std_image, img <= mean_image - 0.5 * std_image
    )

    signal_std = torch.std(img[mask])

    noise_mean = torch.mean(img[mask])
    noise_std = signal_std / np.sqrt(preproc_params["SNR"])

    noise = torch.normal(mean=noise_mean, std=noise_std, size=img.shape)

    img_noise = img + noise

    return img_noise


def add_noise_to_dataset(dataset, preproc_params):

    images_with_noise = torch.empty_like(dataset, device=preproc_params["DEVICE"])
    n_pixels = int(np.sqrt(dataset.shape[1]))

    for i in range(dataset.shape[0]):

        tmp_image = add_noise(dataset[i].reshape(n_pixels, n_pixels), preproc_params)

        images_with_noise[i] = tmp_image.reshape(1, -1).to(preproc_params["DEVICE"])

    return images_with_noise
