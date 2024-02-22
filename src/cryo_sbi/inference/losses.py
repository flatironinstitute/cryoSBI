import torch
import torch.nn as nn

class NPELoss_LatentPrior(nn.Module):
    """Creates a module that calculates the negative log-likelihood (NLL) loss for a
    NPE density estimator and the gaussian latent prior.

    Arguments:
        estimator: A log-density estimator.
    """

    def __init__(self, estimator: nn.Module, lambda_latent: float = 0.1):
        super().__init__()

        print("Training NPE with latent prior loss with lambda_latent = ", lambda_latent)
        self.estimator = estimator
        self.lambda_latent = lambda_latent

    def forward(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:

        latent_vec = self.estimator.embedding(x)
        log_p = self.estimator.npe(self.estimator.standardize(theta), latent_vec)

        return -log_p.mean() + self.lambda_latent * (latent_vec.pow(2).sum(1).mean() / 2)