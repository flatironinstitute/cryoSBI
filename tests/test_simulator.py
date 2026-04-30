import pytest
import torch

from tests.conftest import TESTS_DIR
from cryo_sbi.simulator.cryo_em_simulator import CryoEmSimulator
from cryo_sbi.simulator.ctf import apply_ctf
from cryo_sbi.simulator.image_generation import gen_rot_matrix
from cryo_sbi.simulator.noise import get_snr


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


def _testing_simulator_config():
    """Load tests/config_files/image_params_testing.json and absolutize model_file."""
    import json
    cfg_path = TESTS_DIR / "config_files" / "image_params_testing.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["model_file"] = str(TESTS_DIR / "models" / "hsp90_models.pt")
    return cfg


@pytest.mark.parametrize(("num_images"), [1, 5])
def test_simulator_default_settings(num_images):
    sim = CryoEmSimulator(_testing_simulator_config())
    images = sim.sample_and_simulate(num_images)
    assert images.shape == torch.Size([num_images, 64, 64])


@pytest.mark.parametrize(("num_images"), [1, 5])
def test_simulator_custom_indices(num_images):
    sim = CryoEmSimulator(_testing_simulator_config())
    test_indices = torch.arange(num_images, dtype=torch.int64)
    images, parameters = sim.sample_and_simulate(
        num_images, indices=test_indices, return_parameters=True
    )
    assert (parameters[0] == test_indices).all().item()
    assert images.shape == torch.Size([num_images, 64, 64])


# ---------------------------------------------------------------------------
# Garbage class tests
# ---------------------------------------------------------------------------

GARBAGE_CONFIG = {
    "n_pixels": 64,
    "pixel_size": 2.06,
    "sigma": [0.5, 5.0],
    "model_file": str(TESTS_DIR / "models" / "hsp90_models.pt"),
    "shift": 20.0,
    "defocus": [1.5, 3.5],
    "snr": [0.05, 0.05],
    "amp": 0.1,
    "b_factor": [1.0, 100.0],
    "n_bg_min": 0,
    "n_bg_max": 3,
    "padding_factor": 2,
    "exclusion_radius": 0.0,
    "garbage_class": True,
    "min_garbage": 2,
    "max_garbage": 5,
}


def test_garbage_prior_returns_14_tensors():
    sim = CryoEmSimulator(GARBAGE_CONFIG)
    params = sim._priors.sample((16,))
    assert len(params) == 14, f"Expected 14 tensors, got {len(params)}"
    garbage_mask = params[13]
    assert garbage_mask.dtype == torch.bool
    assert garbage_mask.shape == (16,)


def test_garbage_prior_produces_garbage_images():
    """With enough samples, at least some should be garbage."""
    sim = CryoEmSimulator(GARBAGE_CONFIG)
    params = sim._priors.sample((200,))
    garbage_mask = params[13]
    assert garbage_mask.any(), "Expected at least one garbage image in 200 samples"
    assert not garbage_mask.all(), "Expected at least one non-garbage image in 200 samples"


def test_garbage_n_slots():
    """n_slots should accommodate both n_bg_max and max_garbage - 1."""
    sim = CryoEmSimulator(GARBAGE_CONFIG)
    expected = max(GARBAGE_CONFIG["n_bg_max"], GARBAGE_CONFIG["max_garbage"] - 1)
    assert sim._priors.n_slots == expected
    assert sim._n_bg_max == expected


def test_garbage_simulate_correct_shape():
    sim = CryoEmSimulator(GARBAGE_CONFIG)
    images, params = sim.sample_and_simulate(8, return_parameters=True)
    assert images.shape == torch.Size([8, 64, 64])
    assert len(params) == 14


def test_garbage_label_override():
    """Garbage images should get label = num_models."""
    sim = CryoEmSimulator(GARBAGE_CONFIG)
    params = sim._priors.sample((200,))
    fg_indices = params[0]
    garbage_mask = params[13]

    batch_indices = fg_indices[:, 0] if fg_indices.ndim == 2 else fg_indices
    batch_indices = batch_indices.clone()
    batch_indices[garbage_mask] = sim.num_models

    assert (batch_indices[garbage_mask] == sim.num_models).all()
    assert (batch_indices[~garbage_mask] < sim.num_models).all()


def test_no_garbage_when_disabled():
    """When garbage_class is False, garbage_mask should be all-False."""
    config = {**GARBAGE_CONFIG, "garbage_class": False}
    sim = CryoEmSimulator(config)
    params = sim._priors.sample((50,))
    assert len(params) == 14
    assert not params[13].any()
