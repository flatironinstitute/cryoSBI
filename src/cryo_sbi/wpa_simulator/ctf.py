import numpy as np
import torch


def calc_ctf(image_params: dict, device='cpu') -> torch.Tensor:
    """
    Calculate the CTF for parameters specified in image_params.

    Args:
        image_params (dict): Dictionary containing the image parameters
            N_PIXELS (int): Number of pixels in the image
            PIXEL_SIZE (float): Pixel size in Angstrom
            DEFOCUS (float or list): Defocus in Angstrom
            B_FACTOR (float): B-factor in Angstrom
            AMP (float): Amplitude contrast
            ELECWAVE (float): Electron wavelength in Angstrom

    Returns:
        ctf (torch.Tensor): CTF for the given image size and defocus
    """

    # Attention look into def pad_image function to know the image size after padding
    image_size = (
        2 * (int(np.ceil(image_params["N_PIXELS"] * 0.1)) + 1)
        + image_params["N_PIXELS"]
    )

    freq_pix_1d = torch.fft.fftfreq(image_size, d=image_params["PIXEL_SIZE"])

    # Get phase shift from dictionary
    if isinstance(image_params["DEFOCUS"], float):
        phase = image_params["DEFOCUS"] * np.pi * 2.0 * 10000 * image_params["ELECWAVE"]

    elif (
        isinstance(image_params["DEFOCUS"], list) and len(image_params["DEFOCUS"]) == 2
    ):
        defocus = np.random.uniform(
            low=image_params["DEFOCUS"][0], high=image_params["DEFOCUS"][1]
        )
        phase = defocus * np.pi * 2.0 * 10000 * image_params["ELECWAVE"]

    else:
        raise ValueError(
            "Defocus should be a single float value or a list of [min_defocus, max_defocus]"
        )

    # Get B-factor from dictionary
    if isinstance(image_params["B_FACTOR"], float):
        b_factor = image_params["B_FACTOR"]
    elif (
        isinstance(image_params["B_FACTOR"], list)
        and len(image_params["B_FACTOR"]) == 2
    ):
        b_factor = np.random.uniform(
            low=image_params["B_FACTOR"][0], high=image_params["B_FACTOR"][1]
        )
    else:
        raise ValueError(
            "B-factor should be a single float value or a list of [min_b_factor, max_b_factor]"
        )

    x, y = torch.meshgrid(freq_pix_1d, freq_pix_1d, indexing="ij")

    freq2_2d = x.to(device)**2 + y.to(device)**2
    imag = torch.zeros_like(freq2_2d, device=device) * 1j

    env = torch.exp(-b_factor * freq2_2d * 0.5)
    ctf = (
        -image_params["AMP"] * torch.cos(phase * freq2_2d * 0.5)
        - np.sqrt(1 - image_params["AMP"] ** 2) * torch.sin(phase * freq2_2d * 0.5)
        + imag
    )
    return ctf * env / image_params["AMP"]


def apply_ctf(image: torch.Tensor, ctf: torch.Tensor) -> torch.Tensor:
    """
    Apply the CTF to an image.

    Args:
        image (torch.Tensor): Image to apply the CTF to
        ctf (torch.Tensor): CTF to apply to the image

    Returns:
        image_ctf (torch.Tensor): Image with the CTF applied
    """

    conv_image_ctf = torch.fft.fft2(image) * ctf
    image_ctf = torch.fft.ifft2(conv_image_ctf).real

    return image_ctf
