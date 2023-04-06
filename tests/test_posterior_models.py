import json
import torch
from cryo_sbi.inference.models import build_models
from cryo_sbi.inference.models import estimator_models
from cryo_sbi.inference.validate_train_config import check_train_params


def _get_config_npe():
    config = json.load(open("tests/training_params_npe_testing.json"))
    check_train_params(config)

    return config


def test_build_npe_model():
    posterior_model = build_models.build_npe_flow_model(_get_config_npe())
    assert isinstance(posterior_model, estimator_models.NPEWithEmbedding)


def test_sample_npe_model():
    posterior_model = build_models.build_npe_flow_model(_get_config_npe())
    for batch_size in [1, 2, 5]:
        for sample_size in [1, 10, 1000]:
            test_image = torch.randn((batch_size, 128, 128))
            samples = posterior_model.sample(test_image, shape=(sample_size,))
            assert samples.shape == torch.Size([sample_size, batch_size, 1])
