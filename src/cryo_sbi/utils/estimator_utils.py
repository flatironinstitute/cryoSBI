import torch


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
