import torch.nn as nn
from functools import partial
from cryo_sbi.inference.models.estimator_models import (
    CLASSIFIER,
    ClassifierWithEmbedding,
)
from cryo_sbi.inference.models.embedding_nets import EMBEDDING_NETS


def build_classifier(config: dict) -> nn.Module:
    """
    Builds a classifier model with an embedding network based on the provided configuration.

    Args:
        config (dict): Configuration dictionary containing settings for the embedding
                       network and classifier.
        
    Returns:
        nn.Module: An instance of ClassifierWithEmbedding combining the embedding
                     network and classifier.
    """
    
    emb_cfg = config["EMBEDDING"]
    emb_model = emb_cfg["MODEL"]
    emb_kwargs = {k.lower(): v for k, v in emb_cfg.items() if k != "MODEL"}
    embedding = partial(EMBEDDING_NETS[emb_model], **emb_kwargs)

    clf_cfg = config["CLASSIFIER"]
    clf_model = clf_cfg["MODEL"]
    clf_kwargs = {k.lower(): v for k, v in clf_cfg.items() if k != "MODEL"}
    clf_kwargs["input_dim"] = emb_kwargs["out_dim"]
    classifier = partial(CLASSIFIER[clf_model], **clf_kwargs)

    return ClassifierWithEmbedding(
        embedding_net=embedding,
        classifier=classifier,
    )
