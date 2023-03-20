import numpy as np
import torch


def gaussian_normalize_image(image):
    mean_img = torch.mean(image)
    std_img = torch.std(image)

    return (image - mean_img) / std_img
