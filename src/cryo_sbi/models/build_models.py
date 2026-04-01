import torch.nn as nn
from functools import partial
from omegaconf import DictConfig, OmegaConf

from cryo_sbi.models.estimator_models import (
    CLASSIFIER,
    ClassifierWithEmbedding,
)
from cryo_sbi.models.embedding_nets import EMBEDDING_NETS


def build_classifier(config) -> nn.Module:
    """
    Builds a classifier model with an embedding network.

    Args:
        config: OmegaConf DictConfig or plain dict with 'embedding' and 'classifier' sections.

    Returns:
        nn.Module: ClassifierWithEmbedding instance.
    """
    if isinstance(config, dict):
        config = OmegaConf.create(config)

    emb_cfg = config.embedding
    emb_model = emb_cfg.model
    emb_kwargs = {k: v for k, v in OmegaConf.to_container(emb_cfg).items() if k != "model"}
    embedding = partial(EMBEDDING_NETS[emb_model], **emb_kwargs)

    clf_cfg = config.classifier
    clf_model = clf_cfg.model
    clf_kwargs = {k: v for k, v in OmegaConf.to_container(clf_cfg).items() if k != "model"}
    clf_kwargs["input_dim"] = emb_kwargs["out_dim"]
    classifier = partial(CLASSIFIER[clf_model], **clf_kwargs)

    return ClassifierWithEmbedding(
        embedding_net=embedding,
        classifier=classifier,
    )
