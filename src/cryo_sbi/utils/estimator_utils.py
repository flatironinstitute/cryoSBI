import torch
import json
from cryo_sbi.inference.models import build_models


@torch.no_grad()
def evaluate_log_prob(
    estimator: torch.nn.Module,
    images: torch.Tensor,
    theta: torch.Tensor,
    device: str = "cpu",
) -> torch.Tensor:
    pass  # TODO implement function to evaluate log prob


@torch.no_grad()
def sample_posterior(
    estimator: torch.nn.Module,
    images: torch.Tensor,
    num_samples: int,
    batch_size: int = 100,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Samples from the posterior distribution

    Args:
        estimator (torch.nn.Module): The posterior to use for sampling.
        images (torch.Tensor): The images used to condition the posterio.
        num_samples (int): The number of samples to draw
        batch_size (int, optional): The batch size for sampling. Defaults to 100.
        device (str, optional): The device to use. Defaults to "cpu".

    Returns:
        torch.Tensor: The posterior samples
    """

    theta_samples = []

    if images.shape[0] > batch_size and batch_size > 0:
        images = torch.split(images, split_size_or_sections=batch_size, dim=0)
    else:
        batch_size = images.shape[0]
        images = [images]

    for image_batch in images:
        samples = estimator.sample(
            image_batch.to(device, non_blocking=True), shape=(num_samples,)
        ).cpu()
        theta_samples.append(samples.reshape(-1, image_batch.shape[0]))

    return torch.cat(theta_samples, dim=1)


@torch.no_grad()
def compute_latent_repr(
    estimator: torch.nn.Module,
    images: torch.Tensor,
    batch_size: int = 100,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Computes the latent representation of images.

    Args:
        estimator (torch.nn.Module): Posterior model for which to compute the latent representation.
        images (torch.Tensor): The images to compute the latent representation for.
        batch_size (int, optional): The batch size to use. Defaults to 100.
        device (str, optional): The device to use. Defaults to "cpu".

    Returns:
        torch.Tensor: The latent representation of the images.
    """

    latent_space_samples = []

    if images.shape[0] > batch_size and batch_size > 0:
        images = torch.split(images, split_size_or_sections=batch_size, dim=0)
    else:
        batch_size = images.shape[0]
        images = [images]

    for image_batch in images:
        samples = estimator.embedding(image_batch.to(device, non_blocking=True)).cpu()
        latent_space_samples.append(samples.reshape(image_batch.shape[0], -1))

    return torch.cat(latent_space_samples, dim=0)


def load_estimator(
    config_file_path: str, estimator_path: str, device: str = "cpu"
) -> torch.nn.Module:
    """
    Loads a trained estimator.

    Args:
        config_file_path (str): Path to the config file used to train the estimator.
        estimator_path (str): Path to the estimator.
        device (str, optional): The device to use. Defaults to "cpu".

    Returns:
        torch.nn.Module: The loaded estimator.
    """

    train_config = json.load(open(config_file_path))
    estimator = build_models.build_npe_flow_model(train_config)
    estimator.load_state_dict(
        torch.load(estimator_path, map_location=torch.device(device))
    )
    estimator.to(device)
    estimator.eval()

    return estimator
