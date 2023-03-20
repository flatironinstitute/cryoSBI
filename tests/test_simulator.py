import pytest
import torch
import numpy as np
import json

from cryo_sbi.wpa_simulator.ctf import calc_ctf, apply_ctf
from cryo_sbi.wpa_simulator.image_generation import gen_img, gen_quat
from cryo_sbi.wpa_simulator.noise import add_noise
from cryo_sbi.wpa_simulator.normalization import gaussian_normalize_image
from cryo_sbi.wpa_simulator.padding import pad_image
from cryo_sbi.wpa_simulator.shift import apply_no_shift, apply_random_shift
from cryo_sbi.wpa_simulator.validate_image_config import check_params
from cryo_sbi import CryoEmSimulator


def _get_config():
    config = json.load(open("tests/image_params_testing.json"))
    check_params(config)

    return config


def test_padding():
    image_params = _get_config()
    pad_width = int(np.ceil(image_params["N_PIXELS"] * 0.1)) + 1
    image = torch.zeros((image_params["N_PIXELS"], image_params["N_PIXELS"]))
    padded_image = pad_image(image, image_params)

    for size in padded_image.shape:
        assert size == pad_width * 2 + image_params["N_PIXELS"]
    return


def test_shift_size():
    image_params = _get_config()
    image = torch.zeros((image_params["N_PIXELS"], image_params["N_PIXELS"]))
    padded_image = pad_image(image, image_params)
    shifted_image = apply_random_shift(padded_image, image_params)

    for size in shifted_image.shape:
        assert size == image_params["N_PIXELS"]
    return


def test_shift_bias():
    image_params = _get_config()

    x_0 = image_params["N_PIXELS"] // 2
    y_0 = image_params["N_PIXELS"] // 2

    image = torch.zeros((image_params["N_PIXELS"], image_params["N_PIXELS"]))
    image[x_0, y_0] = 1
    image[x_0 - 1, y_0] = 1
    image[x_0, y_0 - 1] = 1
    image[x_0 - 1, y_0 - 1] = 1

    padded_image = pad_image(image, image_params)
    shifted_image = torch.zeros_like(image)

    for _ in range(10000):
        shifted_image = shifted_image + apply_random_shift(padded_image, image_params)

    indices_x, indices_y = np.where(shifted_image >= 1)

    assert np.mean(indices_x) == image_params["N_PIXELS"] / 2 - 0.5
    assert np.mean(indices_y) == image_params["N_PIXELS"] / 2 - 0.5

    return


def test_no_shift():
    image_params = _get_config()
    image = torch.zeros((image_params["N_PIXELS"], image_params["N_PIXELS"]))
    padded_image = pad_image(image, image_params)
    shifted_image = apply_no_shift(padded_image, image_params)

    for size in shifted_image.shape:
        assert size == image_params["N_PIXELS"]

    assert torch.allclose(image, shifted_image)
    return


def test_normalization():
    image_params = _get_config()
    img_shape = (image_params["N_PIXELS"], image_params["N_PIXELS"])
    image = torch.distributions.normal.Normal(23, 1.30432).sample(img_shape)
    gnormed_image = gaussian_normalize_image(image)

    assert torch.allclose(torch.mean(gnormed_image), torch.tensor(0.0), atol=1e-3)
    assert torch.allclose(torch.std(gnormed_image), torch.tensor(1.0), atol=1e-3)
    return


def test_noise():
    image_params = _get_config()
    N = 10000
    stds = torch.zeros(N)

    for i in range(N):
        image = torch.ones((image_params["N_PIXELS"], image_params["N_PIXELS"]))
        image_noise = add_noise(image, image_params)

        stds[i] = torch.std(image_noise)

    assert torch.allclose(torch.mean(stds), torch.tensor(1.0), atol=1e-3)
    return


# def test_ctf():


def test_simulation():
    simul = CryoEmSimulator("tests/image_params_testing.json")
    image_sim = simul.simulator(index=torch.tensor(0.), seed=0)

    image_params = _get_config()
    model = np.load(image_params["MODEL_FILE"])[0, 0]
    image = gen_img(model, image_params)
    image = pad_image(image, image_params)
    ctf = calc_ctf(image_params)
    image = apply_ctf(image, ctf)
    image = add_noise(image, image_params, seed=0)
    image = apply_random_shift(image, image_params, seed=0)
    image = gaussian_normalize_image(image)
    image = image.to(dtype=torch.float32)

    assert torch.allclose(image, image_sim)
    return
