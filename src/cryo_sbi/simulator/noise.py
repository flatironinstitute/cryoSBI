import math

import torch


# Constant: log(10) / 2. Used to convert log10 SNR into a multiplicative
# noise-power factor via a single fused exp(); avoids per-call torch.tensor /
# torch.pow scalar churn on the GPU path.
_HALF_LN10 = 0.5 * math.log(10.0)


def circular_mask(n_pixels: int, radius: int, device: str = "cpu") -> torch.Tensor:
    """
    Creates a circular mask of radius RADIUS_MASK centered in the image

    Args:
        n_pixels (int): Number of pixels along image side.
        radius (int): Radius of the mask.
        device (str, optional): Device on which to allocate the mask. Defaults to "cpu".

    Returns:
        mask (torch.Tensor): Mask of shape (n_pixels, n_pixels).
    """

    grid = torch.linspace(
        -0.5 * (n_pixels - 1), 0.5 * (n_pixels - 1), n_pixels, device=device
    )
    r_2d = grid[None, :] ** 2 + grid[:, None] ** 2
    mask = r_2d < radius**2

    return mask


def get_snr(images, snr, signal_power=None):
    """
    Per-image noise std that achieves the target SNR.

    The simulator hot path passes ``signal_power`` (shape ``(B,)``) precomputed
    from the FG region of the post-CTF image, so the noise is calibrated to the
    foreground particle alone. When ``signal_power`` is None the legacy mask +
    RMS estimator is used on ``images`` itself — kept for backwards-compat with
    direct callers (tests, ad-hoc use); not used during training/inference.

    Args:
        images (torch.Tensor): (B, N, N) noiseless images, used only when
            ``signal_power`` is None.
        snr (torch.Tensor): log10 SNR values, shape ``(B,)`` or
            broadcastable to that.
        signal_power (torch.Tensor, optional): precomputed per-image RMS of
            the FG region, shape ``(B,)``.

    Returns:
        torch.Tensor: noise std per image, shape ``(B, 1, 1)``.
    """
    if signal_power is None:
        # Legacy fallback. RMS rather than std so non-zero-mean inputs are
        # measured correctly (the old `torch.std(images[:, mask])` was wrong
        # for any image whose mean isn't zero).
        N = images.shape[-1]
        mask = circular_mask(
            n_pixels=N, radius=N // 2, device=images.device
        ).to(images.dtype)
        mask_count = mask.sum().clamp_min(1.0)
        sq_mean = (images * images * mask).sum(dim=(-2, -1)) / mask_count
        signal_power = sq_mean.clamp_min(torch.finfo(images.dtype).eps).sqrt()

    snr_flat = snr.reshape(snr.shape[0])
    return (signal_power * torch.exp(-snr_flat * _HALF_LN10)).reshape(-1, 1, 1)


def add_noise(image: torch.Tensor, snr, seed=None, signal_power=None) -> torch.Tensor:
    """
    Adds Gaussian noise to image, scaled to achieve the target SNR.

    Args:
        image (torch.Tensor): Image of shape (B, n_pixels, n_pixels).
        snr (torch.Tensor): log10 SNR values, shape (B, 1, 1) or broadcastable.
        seed (int, optional): Seed for a *local* RNG. Defaults to None
            (uses torch's global RNG without re-seeding it).
        signal_power (torch.Tensor, optional): precomputed per-image FG RMS
            of shape ``(B,)``. When given, drives the noise scaling directly;
            otherwise falls back to the legacy mask + RMS estimator on
            ``image``.

    Returns:
        image_noise (torch.Tensor): Image with noise of shape (B, n_pixels, n_pixels).
    """
    noise_power = get_snr(image, snr, signal_power=signal_power)
    if seed is None:
        noise = torch.randn(image.shape, dtype=image.dtype, device=image.device)
    else:
        # Local Generator so reproducibility doesn't reseed the global RNG
        # (which would corrupt every other stochastic call in the process).
        gen = torch.Generator(device=image.device).manual_seed(int(seed))
        noise = torch.randn(
            image.shape, dtype=image.dtype, device=image.device, generator=gen
        )

    return image + noise * noise_power
