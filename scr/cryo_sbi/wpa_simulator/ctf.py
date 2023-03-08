import numpy as np
import torch


def calc_ctf(image_params):

    # Attention look into padding.py function to know the image size a priori to padding
    image_size = 2 * ( int(np.ceil(image_params["N_PIXELS"] * 0.1)) + 1) + image_params["N_PIXELS"]

    freq_pix_1d = torch.fft.fftfreq(
        image_size,
        d=image_params["PIXEL_SIZE"]
    )
    
    phase = image_params["DEFOCUS"] * np.pi * 2.0 * 10000 * image_params["ELECWAVE"]

    x, y = torch.meshgrid(freq_pix_1d, freq_pix_1d)

    freq2_2d = x**2 + y**2
    imag = torch.zeros_like(freq2_2d) * 1j

    env = torch.exp(torch.tensor(-image_params["B_FACTOR"] * freq2_2d * 0.5))
    ctf = (
        image_params["AMP"] * torch.tensor(phase * freq2_2d * 0.5).cos()
        - torch.tensor(1 - image_params["AMP"]**2).sqrt()
        * torch.tensor(phase * freq2_2d * 0.5).sin()
        + imag
    )
    return ctf * env / image_params["AMP"]


def apply_ctf(image, ctf):

    conv_image_ctf = torch.fft.fft2(image) * ctf
    image_ctf = torch.fft.ifft2(conv_image_ctf).real

    return image_ctf
