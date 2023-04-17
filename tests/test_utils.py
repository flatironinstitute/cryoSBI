import pytest
import torch
import numpy as np
import json

from cryo_sbi.inference.models import build_models
from cryo_sbi.inference.validate_train_config import check_train_params
from cryo_sbi.utils.estimator_utils import sample_posterior, compute_latent_repr


def _get_config():
    config = json.load(open("tests/training_params_npe_testing.json"))
    check_train_params(config)

    return config


def test_sampling():
    estimator = build_models.build_npe_flow_model(_get_config())
    estimator.eval()

    for num_images in [1, 3, 16, 32]:
        for num_samples in [2, 10, 100]:
            for batch_size in [-1, 0, 1, 10, 1000]:
                images = torch.randn((num_images, 128, 128))
                samples = sample_posterior(
                    estimator, images, num_samples=num_samples, batch_size=batch_size
                )
                assert samples.shape == torch.Size(
                    [num_samples, num_images]
                ), f"Failed with: num_images: {num_images}, num_samles:{num_samples}, batch_size:{batch_size}"


def test_latent_extraction():
    estimator = build_models.build_npe_flow_model(_get_config())
    estimator.eval()

    latent_dim = _get_config()["OUT_DIM"]

    for num_images in [1, 3, 16, 32]:
        for batch_size in [-1, 0, 1, 10, 1000]:
            images = torch.randn((num_images, 128, 128))
            samples = compute_latent_repr(estimator, images, batch_size=batch_size)
            assert samples.shape == torch.Size(
                [num_images, latent_dim]
            ), f"Failed with: num_images: {num_images}, batch_size:{batch_size}"
