import numpy as np
import torch


def apply_ctf(image: torch.Tensor, defocus, b_factor, amp, pixel_size) -> torch.Tensor:
    """
    Apply the CTF convolution to an image.
    """

    num_batch, num_pixels, _ = image.shape
    freq_pix_1d = torch.fft.fftfreq(num_pixels, d=pixel_size, device=image.device)
    x, y = torch.meshgrid(freq_pix_1d, freq_pix_1d, indexing="ij")

    freq2_2d = x**2 + y**2
    freq2_2d = freq2_2d.expand(num_batch, -1, -1)
    imag = torch.zeros_like(freq2_2d, device=image.device) * 1j

    env = torch.exp(-b_factor * freq2_2d * 0.5)
    phase = defocus * torch.pi * 2.0 * 10000 * 0.019866

    ctf = (
        -amp * torch.cos(phase * freq2_2d * 0.5)
        - torch.sqrt(1 - amp**2) * torch.sin(phase * freq2_2d * 0.5)
        + imag
    )
    ctf = ctf * env / amp

    conv_image_ctf = torch.fft.fft2(image) * ctf
    image_ctf = torch.fft.ifft2(conv_image_ctf).real

    return image_ctf
