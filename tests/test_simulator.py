import pytest
import torch
import numpy as np
import json

from cryo_sbi.wpa_simulator.cryo_em_simulator import cryo_em_simulator
from cryo_sbi.wpa_simulator.ctf import apply_ctf
from cryo_sbi.wpa_simulator.image_generation import (
    project_density,
    gen_quat,
    gen_rot_matrix,
)
from cryo_sbi.wpa_simulator.noise import add_noise, circular_mask, get_snr
from cryo_sbi.wpa_simulator.normalization import gaussian_normalize_image
from cryo_sbi.inference.priors import get_image_priors


def test_apply_ctf():
    # Create a test image
    image = torch.randn(1, 64, 64)

    # Set test parameters
    defocus = torch.tensor([1.0])
    b_factor = torch.tensor([100.0])
    amp = torch.tensor([0.5])
    pixel_size = torch.tensor(1.0)

    # Apply CTF to the test image
    image_ctf = apply_ctf(image, defocus, b_factor, amp, pixel_size)

    assert image_ctf.shape == image.shape
    assert isinstance(image_ctf, torch.Tensor)
    assert not torch.allclose(image_ctf, image)


def test_gen_rot_matrix():
    # Create a test quaternion
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    # Generate a rotation matrix from the quaternion
    rot_matrix = gen_rot_matrix(quat)

    assert rot_matrix.shape == torch.Size([1, 3, 3])
    assert isinstance(rot_matrix, torch.Tensor)
    assert torch.allclose(rot_matrix, torch.eye(3).unsqueeze(0))


def test_gen_rot_matrix_batched():
    # Create a test quaternions with batche size 3
    quat = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
    )

    # Generate a rotation matrix from the quaternion
    rot_matrix = gen_rot_matrix(quat)

    assert rot_matrix.shape == torch.Size([3, 3, 3])
    assert isinstance(rot_matrix, torch.Tensor)
    assert torch.allclose(rot_matrix, torch.eye(3).repeat(3, 1, 1))


@pytest.mark.parametrize(
    ("noise_std", "num_images"),
    [
        (torch.tensor([1.5, 1]), 2),
        (torch.tensor([1.0, 2.0, 3.0]), 3),
        (torch.tensor([0.1]), 10),
    ],
)
def test_get_snr(noise_std, num_images):
    # Create a test image
    images = noise_std.reshape(-1, 1, 1) * torch.randn(num_images, 128, 128)

    # Compute the SNR of the test image
    snr = get_snr(images, torch.tensor([0.0]))

    assert snr.shape == torch.Size([images.shape[0], 1, 1]), "SNR has wrong shape"
    assert isinstance(snr, torch.Tensor)
    assert torch.allclose(snr.flatten(), noise_std * torch.ones(images.shape[0]), atol=1e-01), "SNR is not correct"
