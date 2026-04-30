import torch.nn as nn
from functools import partial
from omegaconf import OmegaConf

from cryo_sbi.models.estimator_models import (
    CLASSIFIER,
    ClassifierWithEmbedding,
)
from cryo_sbi.models.embedding_nets import EMBEDDING_NETS


def _lookup(registry: dict, name: str) -> type:
    """Case-insensitive registry lookup: accept lowercase configs while keeping
    uppercase canonical keys for back-compat with already-trained checkpoints."""
    if name in registry:
        return registry[name]
    upper = name.upper()
    if upper in registry:
        return registry[upper]
    raise KeyError(
        f"Unknown model {name!r}. Available: {sorted(registry.keys())}"
    )


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
    emb_kwargs = {
        k: v
        for k, v in OmegaConf.to_container(emb_cfg).items()
        if k != "model" and v is not None
    }
    embedding = partial(_lookup(EMBEDDING_NETS, emb_cfg.model), **emb_kwargs)

    clf_cfg = config.classifier
    # Filter None-valued fields so MLP-only / PROTOTYPE-only kwargs in the
    # ClassifierConfig dataclass schema don't get forwarded to the constructor.
    clf_kwargs = {
        k: v
        for k, v in OmegaConf.to_container(clf_cfg).items()
        if k != "model" and v is not None
    }
    clf_kwargs["input_dim"] = emb_kwargs["out_dim"]
    classifier = partial(_lookup(CLASSIFIER, clf_cfg.model), **clf_kwargs)

    return ClassifierWithEmbedding(
        embedding_net=embedding,
        classifier=classifier,
    )
