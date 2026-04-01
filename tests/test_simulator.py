import pytest
import torch

from cryo_sbi.simulator.cryo_em_simulator import cryo_em_simulator, CryoEmSimulator
from cryo_sbi.simulator.ctf import apply_ctf
from cryo_sbi.simulator.image_generation import (
    project_density,
    gen_quat,
    gen_rot_matrix,
)
from cryo_sbi.simulator.noise import add_noise, circular_mask, get_snr
from cryo_sbi.simulator.normalization import gaussian_normalize_image
from cryo_sbi.simulator.priors import get_image_priors


def test_apply_ctf():
    image = torch.randn(1, 64, 64)
    defocus = torch.tensor([1.0])
    b_factor = torch.tensor([100.0])
    amp = torch.tensor([0.5])
    pixel_size = torch.tensor(1.0)
    image_ctf = apply_ctf(image, defocus, b_factor, amp, pixel_size)
    assert image_ctf.shape == image.shape
    assert isinstance(image_ctf, torch.Tensor)
    assert not torch.allclose(image_ctf, image)


def test_gen_rot_matrix():
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    rot_matrix = gen_rot_matrix(quat)
    assert rot_matrix.shape == torch.Size([1, 3, 3])
    assert isinstance(rot_matrix, torch.Tensor)
    assert torch.allclose(rot_matrix, torch.eye(3).unsqueeze(0))


def test_gen_rot_matrix_batched():
    quat = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
    )
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
    images = noise_std.reshape(-1, 1, 1) * torch.randn(num_images, 128, 128)
    snr = get_snr(images, torch.tensor([0.0]))
    assert snr.shape == torch.Size([images.shape[0], 1, 1]), "SNR has wrong shape"
    assert isinstance(snr, torch.Tensor)
    assert torch.allclose(
        snr.flatten(), noise_std * torch.ones(images.shape[0]), atol=1e-01
    ), "SNR is not correct"


@pytest.mark.parametrize(("num_images"), [1, 5])
def test_simulator_default_settings(num_images):
    sim = CryoEmSimulator("tests/config_files/image_params_testing.json")
    images = sim.sample_and_simulate(num_images)
    assert images.shape == torch.Size([num_images, 64, 64])


@pytest.mark.parametrize(("num_images"), [1, 5])
def test_simulator_custom_indices(num_images):
    sim = CryoEmSimulator("tests/config_files/image_params_testing.json")
    test_indices = torch.arange(num_images, dtype=torch.int64)
    images, parameters = sim.sample_and_simulate(
        num_images, indices=test_indices, return_parameters=True
    )
    assert (parameters[0] == test_indices).all().item()
    assert images.shape == torch.Size([num_images, 64, 64])
