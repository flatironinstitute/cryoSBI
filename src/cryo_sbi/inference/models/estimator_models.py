import torch
import torch.nn as nn
import zuko
from lampe.inference import NPE, NRE


class Standardize(nn.Module):
    """
    Module to standardize inputs and retransform them to the original space

    Args:
        mean (torch.Tensor): mean of the data
        std (torch.Tensor): standard deviation of the data

    Returns:
        standardized (torch.Tensor): standardized data
    """

    # Code adapted from :https://github.com/mackelab/sbi/blob/main/sbi/utils/sbiutils.py
    def __init__(self, mean: float, std: float) -> None:
        super(Standardize, self).__init__()
        mean, std = map(torch.as_tensor, (mean, std))
        self.mean = mean
        self.std = std
        self.register_buffer("_mean", mean)
        self.register_buffer("_std", std)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Standardize the input tensor

        Args:
            tensor (torch.Tensor): input tensor

        Returns:
            standardized (torch.Tensor): standardized tensor
        """

        return (tensor - self._mean) / self._std

    def transform(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Transform the standardized tensor back to the original space

        Args:
            tensor (torch.Tensor): input tensor

        Returns:
            retransformed (torch.Tensor): retransformed tensor
        """

        return (tensor * self._std) + self._mean


class NPEWithEmbedding(nn.Module):
    """Neural Posterior Estimation with embedding net

    Attributes:
        npe (NPE): NPE model
        embedding (nn.Module): embedding net
        standardize (Standardize): standardization module
    """

    def __init__(
        self,
        embedding_net: nn.Module,
        output_embedding_dim: int,
        num_transforms: int = 4,
        num_hidden_flow: int = 2,
        hidden_flow_dim: int = 128,
        flow: nn.Module = zuko.flows.MAF,
        theta_shift: float = 0.0,
        theta_scale: float = 1.0,
        **kwargs,
    ) -> None:
        """
        Neural Posterior Estimation with embedding net.

        Args:
            embedding_net (nn.Module): embedding net
            output_embedding_dim (int): output embedding dimension
            num_transforms (int, optional): number of transforms. Defaults to 4.
            num_hidden_flow (int, optional): number of hidden layers in flow. Defaults to 2.
            hidden_flow_dim (int, optional): hidden dimension in flow. Defaults to 128.
            flow (nn.Module, optional): flow. Defaults to zuko.flows.MAF.
            theta_shift (float, optional): Shift of the theta for standardization. Defaults to 0.0.
            theta_scale (float, optional): Scale of the theta for standardization. Defaults to 1.0.
            kwargs: additional arguments for the flow

        Returns:
            None
        """

        super().__init__()

        self.npe = NPE(
            1,
            output_embedding_dim,
            transforms=num_transforms,
            build=flow,
            hidden_features=[*[hidden_flow_dim] * num_hidden_flow, 128, 64],
            **kwargs,
        )

        self.embedding = embedding_net()
        self.standardize = Standardize(theta_shift, theta_scale)

    def forward(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the NPE model

        Args:
            theta (torch.Tensor): Conformational parameters.
            x (torch.Tensor): Image to condition the posterior on.

        Returns:
            torch.Tensor: Log probability of the posterior.
        """

        return self.npe(self.standardize(theta), self.embedding(x))

    def flow(self, x: torch.Tensor) -> zuko.flows.FlowModule:
        """
        Conditions the posterior on an image.

        Args:
            x (torch.Tensor): Image to condition the posterior on.

        Returns:
            zuko.flows.Flow: The posterior distribution.
        """
        return self.npe.flow(self.embedding(x))

    def sample(self, x: torch.Tensor, shape=(1,)) -> torch.Tensor:
        """
        Generate samples from the posterior distribution.

        Args:
            x (torch.Tensor): Image to condition the posterior on.
            shape (tuple, optional): Shape of the samples. Defaults to (1,).

        Returns:
            torch.Tensor: Samples from the posterior distribution.
        """

        samples_standardized = self.flow(x).sample(shape)
        return self.standardize.transform(samples_standardized)
