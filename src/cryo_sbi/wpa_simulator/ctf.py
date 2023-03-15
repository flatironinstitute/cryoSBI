import numpy as np
import torch


def calc_ctf(image_params):
    # Attention look into def pad_image function to know the image size after padding 
    image_size = 2 * (int(np.ceil(image_params["N_PIXELS"] * 0.1)) + 1) + image_params["N_PIXELS"]

    freq_pix_1d = torch.fft.fftfreq(
        image_size,
        d=image_params["PIXEL_SIZE"]
    )

    if isinstance(image_params["DEFOCUS"], float):
        phase = image_params["DEFOCUS"] * np.pi * 2.0 * 10000 * image_params["ELECWAVE"]

    elif isinstance(image_params["DEFOCUS"], list) and len(image_params["DEFOCUS"]) == 2:
        defocus = np.random.uniform(low=image_params["DEFOCUS"][0], high=image_params["DEFOCUS"][1])
        phase = defocus * np.pi * 2.0 * 10000 * image_params["ELECWAVE"]

    else:
        raise ValueError("Defocus should be a single float value or a list of [min_defocus, max_defocus]")

    x, y = torch.meshgrid(freq_pix_1d, freq_pix_1d)

    freq2_2d = x**2 + y**2
    imag = torch.zeros_like(freq2_2d) * 1j

    env = torch.exp(-image_params["B_FACTOR"] * freq2_2d * 0.5)
    ctf = (
        image_params["AMP"] * torch.cos(phase * freq2_2d * 0.5)
        - np.sqrt(1 - image_params["AMP"] ** 2)
        * torch.sin(phase * freq2_2d * 0.5)
        + imag
    )
    return ctf * env / image_params["AMP"]


def apply_ctf(image, ctf):
    conv_image_ctf = torch.fft.fft2(image) * ctf
    image_ctf = torch.fft.ifft2(conv_image_ctf).real

    return image_ctf
