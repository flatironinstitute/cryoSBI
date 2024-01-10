import pytest
import torch
import numpy as np
import json

from cryo_sbi.wpa_simulator.cryo_em_simulator import cryo_em_simulator
from cryo_sbi.wpa_simulator.ctf import apply_ctf
from cryo_sbi.wpa_simulator.image_generation import project_density, gen_quat, gen_rot_matrix
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
    quat = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0]
                        ])

    # Generate a rotation matrix from the quaternion
    rot_matrix = gen_rot_matrix(quat)

    assert rot_matrix.shape == torch.Size([3, 3, 3])
    assert isinstance(rot_matrix, torch.Tensor)
    assert torch.allclose(rot_matrix, torch.eye(3).repeat(3, 1, 1))