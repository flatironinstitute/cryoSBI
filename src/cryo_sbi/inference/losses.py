import torch
import torch.nn as nn


def kernel_matrix(x, y, l):
    d = torch.cdist(x, y)**2

    kernel = torch.exp(-(1 / (2 * l ** 2)) * d)

    return kernel


def mmd_unweighted(x, y, lengthscale):
    """ Approximates the squared MMD between samples x_i ~ P and y_i ~ Q
    """

    m = x.shape[0]
    n = y.shape[0]

    z = torch.cat((x, y), dim=0)

    K = kernel_matrix(z, z, lengthscale)

    kxx = K[0:m, 0:m]
    kyy = K[m:(m + n), m:(m + n)]
    kxy = K[0:m, m:(m + n)]

    return (1 / m ** 2) * torch.sum(kxx) - (2 / (m * n)) * torch.sum(kxy) + (1 / n ** 2) * torch.sum(kyy)


def median_heuristic(y):
    a = torch.cdist(y, y)**2
    return torch.sqrt(torch.median(a / 2))


class NPERobustStatsLoss(nn.Module):

    def __init__(self, estimator: nn.Module, gamma: float):
        super().__init__()
        print("distance loss")
        self.estimator = estimator
        self.distance = nn.PairwiseDistance(p=10)

    def forward(self, theta: torch.Tensor, x: torch.Tensor, x_obs: torch.Tensor) -> torch.Tensor:

        latent_vecs_x = self.estimator.embedding(x)
        self.estimator.eval()
        latent_vecs_x_obs = self.estimator.embedding(x_obs)
        summary_stats_regularization = self.distance(latent_vecs_x, latent_vecs_x_obs).mean()
        self.estimator.train()

        log_p = self.estimator.npe(self.estimator.standardize(theta), latent_vecs_x)
        print(-log_p.mean().item(), summary_stats_regularization.mean().item())
        
        return -log_p.mean() + 0.0 * summary_stats_regularization