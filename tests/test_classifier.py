import pytest
import torch
from omegaconf import OmegaConf

from cryo_sbi.models import build_models
from cryo_sbi.models import estimator_models


@pytest.fixture(
    params=[
        "tests/config_files/training_params_mlp.json",
        "tests/config_files/training_params_proto.json",
    ]
)
def train_params(request):
    return OmegaConf.load(request.param)


def test_build_classifier_model(train_params):
    classifier = build_models.build_classifier(train_params)
    assert isinstance(classifier, estimator_models.ClassifierWithEmbedding)


@pytest.mark.parametrize(
    ("batch_size", "sample_size"), [(1, 1), (2, 10), (5, 1000), (100, 2)]
)
def test_classifier_inference(train_params, batch_size, sample_size):
    classifier = build_models.build_classifier(train_params)
    test_image = torch.randn((batch_size, 128, 128))
    logits = classifier(test_image)
    assert logits.shape == torch.Size(
        [batch_size, train_params.classifier.num_classes]
    )


def test_classifier_probs(train_params):
    classifier = build_models.build_classifier(train_params)
    test_image = torch.randn((10, 128, 128))
    logits = classifier.probs(test_image)
    assert logits.shape == torch.Size(
        [10, train_params.classifier.num_classes]
    )
    assert torch.allclose(logits.sum(dim=1), torch.ones(10))


def test_classifier_logits_embeddings(train_params):
    classifier = build_models.build_classifier(train_params)
    test_image = torch.randn((10, 128, 128))
    logits, embeddings = classifier.logits_embedding(test_image)
    assert logits.shape == torch.Size(
        [10, train_params.classifier.num_classes]
    )
    assert embeddings.shape == torch.Size([10, train_params.embedding.out_dim])
