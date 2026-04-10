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
    estimator = build_models.build_classifier(train_config)
    estimator.load_state_dict(
        torch.load(estimator_path, map_location=torch.device(device), weights_only=True)
    )
    estimator.to(device)
    estimator.eval()
    return estimator
