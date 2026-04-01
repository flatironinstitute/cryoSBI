import torch
from omegaconf import OmegaConf

from cryo_sbi.models import build_models


def load_classifier(
    config_path: str, estimator_path: str, device: str = "cpu"
) -> torch.nn.Module:
    """
    Loads a trained classifier.

    Args:
        config_path: Path to the training config file (YAML or JSON).
        estimator_path: Path to the saved model state dict.
        device: Device string.

    Returns:
        torch.nn.Module: The loaded classifier in eval mode.
    """
    config = OmegaConf.load(config_path)
    estimator = build_models.build_classifier(config)
    estimator.load_state_dict(
        torch.load(estimator_path, map_location=torch.device(device), weights_only=True)
    )
    estimator.to(device)
    estimator.eval()
    return estimator
