from typing import Union, Callable

from cryo_sbi.wpa_simulator.ctf import apply_ctf
from cryo_sbi.wpa_simulator.image_generation import project_density
from cryo_sbi.wpa_simulator.noise import add_noise
from cryo_sbi.wpa_simulator.normalization import gaussian_normalize_image


def cryo_em_simulator(
    models,
    index,
    quaternion,
    sigma,
    defocus,
    b_factor,
    amp,
    snr,
    num_pixels,
    pixel_size,
):
    models_selected = models[index.round().long().flatten()]
    image = project_density(
        models_selected,
        quaternion,
        sigma,
        num_pixels,
        pixel_size,
    )
    image = apply_ctf(image, defocus, b_factor, amp, pixel_size)
    image = add_noise(image, snr)
    image = gaussian_normalize_image(image)
    return image
