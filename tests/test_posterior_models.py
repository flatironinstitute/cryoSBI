import pytest
import json
import torch
from cryo_sbi.inference.models import build_models
from cryo_sbi.inference.models import estimator_models
from cryo_sbi.inference.validate_train_config import check_train_params


@pytest.fixture
def train_params():
    config = json.load(open("tests/config_files/training_params_npe_testing.json"))
    check_train_params(config)
    return config


def test_build_npe_model(train_params):
    posterior_model = build_models.build_npe_flow_model(train_params)
    assert isinstance(posterior_model, estimator_models.NPEWithEmbedding)


@pytest.mark.parametrize(
    ("batch_size", "sample_size"), [(1, 1), (2, 10), (5, 1000), (100, 2)]
)
def test_sample_npe_model(train_params, batch_size, sample_size):
    posterior_model = build_models.build_npe_flow_model(train_params)
    test_image = torch.randn((batch_size, 128, 128))
    samples = posterior_model.sample(test_image, shape=(sample_size,))
    assert samples.shape == torch.Size([sample_size, batch_size, 1])
