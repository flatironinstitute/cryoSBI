import numpy as np
from torch.nn.functional import pad
from torch.nn import ConstantPad2d


def pad_image(image, image_params):
    pad_width = int(np.ceil(image_params["N_PIXELS"] * 0.1)) + 1
    padder = ConstantPad2d(pad_width, 0.0)
    padded_image = padder(image)

    return padded_image