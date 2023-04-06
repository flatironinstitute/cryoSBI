import torch
import json
from cryo_sbi.inference.models import build_models


def sample_posterior(estimator, images, num_samples, batch_size=100, device="cpu"):
    theta_samples = []

    if images.shape[0] > batch_size and batch_size > 0:
        images = torch.split(images, split_size_or_sections=batch_size, dim=0)
    else:
        batch_size = images.shape[0]
        images = [images]

    with torch.no_grad():
        for image_batch in images:
            samples = estimator.sample(
                image_batch.to(device, non_blocking=True), shape=(num_samples,)
            ).cpu()
            theta_samples.append(samples.reshape(-1, image_batch.shape[0]))

    return torch.cat(theta_samples, dim=1)


def compute_latent_repr(estimator, images, batch_size=100, device="cpu"):
    latent_space_samples = []

    if images.shape[0] > batch_size and batch_size > 0:
        images = torch.split(images, split_size_or_sections=batch_size, dim=0)
    else:
        batch_size = images.shape[0]
        images = [images]

    with torch.no_grad():
        for image_batch in images:
            samples = estimator.embedding(
                image_batch.to(device, non_blocking=True)
            ).cpu()
            latent_space_samples.append(samples.reshape(image_batch.shape[0], -1))

    return torch.cat(latent_space_samples, dim=0)


def load_estimator(config_file_path, estimator_path, device="cpu"):
    train_config = json.load(open(config_file_path))
    estimator = build_models.build_npe_flow_model(train_config)
    estimator.load_state_dict(torch.load(estimator_path))
    estimator.to(device)
    estimator.eval()

    return estimator
