import numpy as np
import torch
from torch.nn.functional import pad
from torch.nn import ConstantPad2d


def pad_image(image: torch.Tensor, image_params: dict) -> torch.Tensor:
    """
    Pads image with zeros to randomly crop image later.

    Args:
        image (torch.Tensor): Image of shape (n_pixels, n_pixels) or (n_channels, n_pixels, n_pixels).
        image_params (dict): Dictionary containing image parameters.

    Returns:
        padded_image (torch.Tensor): Padded image of shape (n_pixels + 2 * pad_width, n_pixels + 2 * pad_width).
        With pad_width = int(np.ceil(image_params["N_PIXELS"] * 0.1)) + 1.
    """

    pad_width = int(np.ceil(image_params["N_PIXELS"] * 0.1)) + 1
    padder = ConstantPad2d(pad_width, 0.0)
    padded_image = padder(image)

    return padded_image
