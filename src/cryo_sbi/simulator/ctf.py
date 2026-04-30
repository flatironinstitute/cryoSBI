import math

import torch


def electron_wavelength_angstrom(voltage_kv: float) -> float:
    """
    Relativistic electron wavelength in Angstrom for an accelerating voltage.

    Args:
        voltage_kv: Microscope voltage in kV (e.g. 300 for a Krios).

    Returns:
        Wavelength in Angstrom.
    """
    v = float(voltage_kv) * 1.0e3
    return 12.2643 / math.sqrt(v * (1.0 + 0.97845e-6 * v))


def apply_ctf(
    image: torch.Tensor,
    defocus,
    b_factor,
    amp,
    pixel_size,
    voltage_kv: float = 300.0,
) -> torch.Tensor:
    """
    Applies the CTF to the image.

    Args:
        image: Image tensor of shape (B, N, N).
        defocus: Defocus values in micrometers, broadcastable to (B, 1, 1).
        b_factor: B-factor values, broadcastable to (B, 1, 1).
        amp: Amplitude contrast values, broadcastable to (B, 1, 1). Must be > 0.
        pixel_size: Pixel size in Angstrom (scalar tensor or float).
        voltage_kv: Accelerating voltage in kV. Defaults to 300 kV.

    Returns:
        torch.Tensor: The image with the CTF applied.
    """

    num_batch, num_pixels, _ = image.shape
    freq_pix_1d = torch.fft.fftfreq(num_pixels, d=pixel_size, device=image.device)
    x, y = torch.meshgrid(freq_pix_1d, freq_pix_1d, indexing="ij")

    freq2_2d = x**2 + y**2
    freq2_2d = freq2_2d.expand(num_batch, -1, -1)

    lambda_e = electron_wavelength_angstrom(voltage_kv)  # Angstrom

    env = torch.exp(-b_factor * freq2_2d * 0.5)
    # Defocus is provided in micrometers; convert to Angstrom (factor 1e4) to match
    # the units of (1/q^2). The 2π factor comes from the standard CTF phase form.
    phase = defocus * torch.pi * 2.0 * 1.0e4 * lambda_e

    ctf = (
        -amp * torch.cos(phase * freq2_2d * 0.5)
        - torch.sqrt(1.0 - amp**2) * torch.sin(phase * freq2_2d * 0.5)
    )
    # The trailing /amp is intentional (see e.g. Mindell & Grigorieff 2003): it
    # removes the amplitude-contrast attenuation introduced by the cosine term.
    # Guard against amp ≈ 0 inputs that would blow up the division.
    amp_safe = torch.as_tensor(amp).clamp_min(1e-6)
    ctf = ctf * env / amp_safe

    conv_image_ctf = torch.fft.fft2(image) * ctf
    image_ctf = torch.fft.ifft2(conv_image_ctf).real

    return image_ctf
