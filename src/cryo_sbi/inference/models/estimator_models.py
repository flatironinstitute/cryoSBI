import torch
import torch.nn as nn
import zuko
from lampe.inference import NPE, NRE

import sys

sys.path.insert(0, "../inference/models")


class Standardize(nn.Module):
    """Module to standardize inputs and retransform them to the original space"""

    # Code adapted from :https://github.com/mackelab/sbi/blob/main/sbi/utils/sbiutils.py
    def __init__(self, mean, std):
        super(Standardize, self).__init__()
        mean, std = map(torch.as_tensor, (mean, std))
        self.mean = mean
        self.std = std
        self.register_buffer("_mean", mean)
        self.register_buffer("_std", std)

    def forward(self, tensor):
        return (tensor - self._mean) / self._std

    def transform(self, tensor):
        return (tensor * self._std) + self._mean


class NPEWithEmbedding(nn.Module):
    def __init__(
        self,
        embedding_net,
        output_embedding_dim,
        num_transforms=4,
        num_hidden_flow=2,
        hidden_flow_dim=128,
        flow=zuko.flows.MAF,
        theta_shift=0,
        theta_scale=1,
        **kwargs
    ):
        super().__init__()

        self.npe = NPE(
            1,
            output_embedding_dim,
            transforms=num_transforms,
            build=flow,
            hidden_features=[*[hidden_flow_dim] * num_hidden_flow, 128, 64],
            **kwargs
        )

        self.embedding = embedding_net()
        self.standardize = Standardize(theta_shift, theta_scale)

    def forward(self, theta: torch.Tensor, x: torch.Tensor):
        return self.npe(self.standardize(theta), self.embedding(x))

    def flow(self, x: torch.Tensor):
        return self.npe.flow(self.embedding(x))

    def sample(self, x: torch.Tensor, shape=(1,)):
        samples_standardized = self.flow(x).sample(shape)
        return self.standardize.transform(samples_standardized)


class NREWithEmbedding(nn.Module):
    def __init__(
        self,
        embedding_net,
        output_embedding_dim,
        hidden_features,
        activation,
        network,
        theta_shift=0,
        theta_scale=1,
    ):
        super().__init__()

        self.nre = NRE(
            1,
            output_embedding_dim,
            hidden_features=hidden_features,
            activation=activation,
            build=network,
        )
        self.embedding = embedding_net
        self.standardize = Standardize(theta_shift, theta_scale)

    def forward(self, theta, x):
        return self.nre(self.standardize(theta), self.embedding(x))
