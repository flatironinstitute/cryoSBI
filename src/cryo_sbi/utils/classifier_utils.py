import logging

import torch
from omegaconf import DictConfig

from cryo_sbi.models import build_models


def load_classifier(
    train_config: DictConfig, estimator_path: str, device: str = "cpu"
) -> torch.nn.Module:
    """
    Load a trained classifier from weights and a training config.

    Args:
        train_config: DictConfig containing model architecture (cfg.train).
        estimator_path: Path to the saved model state dict (.pt).
        device: Device string.

    Returns:
        torch.nn.Module: Loaded classifier in eval mode.
    """
    state_dict = torch.load(
        estimator_path, map_location=torch.device(device), weights_only=True
    )

    # num_classes is not in the config — infer it from the saved weights so the
    # rebuilt architecture matches whatever was trained.
    if train_config.classifier.model.upper() == "PROTOTYPE":
        source_key = "classifier.prototypes"
        num_classes = int(state_dict[source_key].shape[0])
    else:
        # MLP: ClassifierWithEmbedding.classifier (MLPClassifier) wraps an
        # nn.Sequential also named ``classifier`` whose last Linear has shape
        # (num_classes, nodes_per_layer). Pick that highest-indexed weight.
        keys = [
            k for k in state_dict
            if k.startswith("classifier.classifier.") and k.endswith(".weight")
        ]
        source_key = max(keys, key=lambda k: int(k.split(".")[2]))
        num_classes = int(state_dict[source_key].shape[0])
    logging.info(
        f"Inferred num_classes={num_classes} from {estimator_path} ({source_key})"
    )

    estimator = build_models.build_classifier(train_config, num_classes)
    estimator.load_state_dict(state_dict)
    estimator.to(device)
    estimator.eval()
    return estimator
