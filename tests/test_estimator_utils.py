import pytest
import os
import torch
import numpy as np
import json

from cryo_sbi.inference.models import build_models
from cryo_sbi.inference.models.estimator_models import NPEWithEmbedding
from cryo_sbi.inference.validate_train_config import check_train_params
from cryo_sbi.utils.estimator_utils import (
    sample_posterior,
    compute_latent_repr,
    evaluate_log_prob,
    load_estimator
)


@pytest.fixture
def train_params():
    config = json.load(open("tests/config_files/training_params_npe_testing.json"))
    check_train_params(config)
    return config


@pytest.fixture
def train_config_path():
    return "tests/config_files/training_params_npe_testing.json"


@pytest.mark.parametrize(
    ("num_images", "num_samples", "batch_size"),
    [(1, 1, 1), (2, 10, 2), (5, 1000, 5), (100, 2, 100)],
)
def test_sampling(train_params, num_images, num_samples, batch_size):
    estimator = build_models.build_npe_flow_model(train_params)
    estimator.eval()
    images = torch.randn((num_images, 128, 128))
    samples = sample_posterior(
        estimator, images, num_samples=num_samples, batch_size=batch_size
    )
    assert samples.shape == torch.Size(
        [num_samples, num_images]
    ), f"Failed with: num_images: {num_images}, num_samles:{num_samples}, batch_size:{batch_size}"


@pytest.mark.parametrize(
    ("num_images", "batch_size"), [(1, 1), (2, 2), (1, 5), (100, 10)]
)
def test_latent_extraction(train_params, num_images, batch_size):
    estimator = build_models.build_npe_flow_model(train_params)
    estimator.eval()

    latent_dim = train_params["OUT_DIM"]
    images = torch.randn((num_images, 128, 128))
    samples = compute_latent_repr(estimator, images, batch_size=batch_size)
    assert samples.shape == torch.Size(
        [num_images, latent_dim]
    ), f"Failed with: num_images: {num_images}, batch_size:{batch_size}"


@pytest.mark.parametrize(
    ("num_images", "num_eval", "batch_size"),
    [(1, 1, 1), (2, 10, 2), (5, 1000, 5), (100, 2, 100)],
)
def test_logprob_eval(train_params, num_images, num_eval, batch_size):
    estimator = build_models.build_npe_flow_model(train_params)
    estimator.eval()
    images = torch.randn((num_images, 128, 128))
    theta = torch.linspace(0, 25, num_eval)
    samples = evaluate_log_prob(
        estimator, images, theta, batch_size=batch_size
    )
    assert samples.shape == torch.Size(
        [num_eval, num_images]
    ), f"Failed with: num_images: {num_images}, num_eval:{num_eval}, batch_size:{batch_size}"


def test_load_estimator(train_params, train_config_path):
    estimator = build_models.build_npe_flow_model(train_params)
    torch.save(estimator.state_dict(), "tests/config_files/test_estimator.estimator")
    estimator = load_estimator(
        train_config_path, "tests/config_files/test_estimator.estimator"
    )
    assert isinstance(estimator, NPEWithEmbedding)
    os.remove("tests/config_files/test_estimator.estimator")
