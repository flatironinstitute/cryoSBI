import pytest
import torch
from omegaconf import OmegaConf

from conftest import TESTS_DIR
from cryo_sbi.models import build_models
from cryo_sbi.models import estimator_models


NUM_CLASSES = 44


@pytest.fixture(
    params=[
        TESTS_DIR / "config_files" / "training_params_mlp.json",
        TESTS_DIR / "config_files" / "training_params_proto.json",
    ]
)
def train_params(request):
    return OmegaConf.load(str(request.param))


def test_build_classifier_model(train_params):
    classifier = build_models.build_classifier(train_params, NUM_CLASSES)
    assert isinstance(classifier, estimator_models.ClassifierWithEmbedding)


@pytest.mark.parametrize(
    # Trimmed: dropped slow (5, 1000) and (100, 2) parametrizations from CI;
    # the small cases below cover both batch=1 and batch>1 code paths.
    ("batch_size", "sample_size"), [(1, 1), (2, 10)]
)
def test_classifier_inference(train_params, batch_size, sample_size):
    classifier = build_models.build_classifier(train_params, NUM_CLASSES)
    test_image = torch.randn((batch_size, 128, 128))
    logits = classifier(test_image)
    assert logits.shape == torch.Size([batch_size, NUM_CLASSES])


def test_classifier_probs(train_params):
    classifier = build_models.build_classifier(train_params, NUM_CLASSES)
    test_image = torch.randn((10, 128, 128))
    logits = classifier.probs(test_image)
    assert logits.shape == torch.Size([10, NUM_CLASSES])
    assert torch.allclose(logits.sum(dim=1), torch.ones(10))


def test_classifier_logits_embeddings(train_params):
    classifier = build_models.build_classifier(train_params, NUM_CLASSES)
    test_image = torch.randn((10, 128, 128))
    logits, embeddings = classifier.logits_embedding(test_image)
    assert logits.shape == torch.Size([10, NUM_CLASSES])
    assert embeddings.shape == torch.Size([10, train_params.embedding.out_dim])
