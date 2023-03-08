import numpy as np
import torch


def gaussian_normalize_image(image):
    mean_img = torch.mean(image)
    std_img = torch.std(image)

    return (image - mean_img) / std_img


def normalize_dataset(dataset, preproc_params):
    norm_images = torch.empty_like(dataset, device=preproc_params["DEVICE"])
    n_pixels = int(np.sqrt(dataset.shape[1]))

    for i in range(dataset.shape[0]):
        tmp_image = gaussian_normalize_image(dataset[i].reshape(n_pixels, n_pixels))

        norm_images[i] = tmp_image.reshape(1, -1).to(preproc_params["DEVICE"])

    return norm_images
