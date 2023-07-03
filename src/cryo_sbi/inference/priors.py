import torch
import zuko
from torch.distributions.distribution import Distribution
from torch.utils.data import DataLoader, Dataset, IterableDataset


def gen_quat() -> torch.Tensor:
    """
    Generate a random quaternion.

    Returns:
        quat (np.ndarray): Random quaternion

    """
    count = 0
    while count < 1:
        quat = 2 * torch.rand(size=(4,)) - 1
        norm = torch.sqrt(torch.sum(quat**2))
        if 0.2 <= norm <= 1.0:
            quat /= norm
            count += 1

    return quat


def get_image_priors(
    max_index, image_config: dict, device="cuda"
) -> zuko.distributions.BoxUniform:
    """
    Return uniform prior in 1d from 0 to 19

    Args:
        max_index (int): max index of the 1d prior

    Returns:
        zuko.distributions.BoxUniform: prior
    """
    if isinstance(image_config["SIGMA"], list) and len(image_config["SIGMA"]) == 2:
        lower = torch.tensor(
            [[image_config["SIGMA"][0]]], dtype=torch.float32, device=device
        )
        upper = torch.tensor(
            [[image_config["SIGMA"][1]]], dtype=torch.float32, device=device
        )
        sigma = zuko.distributions.BoxUniform(lower=lower, upper=upper, ndims=1)

    if isinstance(image_config["DEFOCUS"], list) and len(image_config["DEFOCUS"]) == 2:
        lower = torch.tensor(
            [[image_config["DEFOCUS"][0]]], dtype=torch.float32, device=device
        )
        upper = torch.tensor(
            [[image_config["DEFOCUS"][1]]], dtype=torch.float32, device=device
        )
        defocus = zuko.distributions.BoxUniform(lower=lower, upper=upper, ndims=1)

    if (
        isinstance(image_config["B_FACTOR"], list)
        and len(image_config["B_FACTOR"]) == 2
    ):
        lower = torch.tensor(
            [[image_config["B_FACTOR"][0]]], dtype=torch.float32, device=device
        )
        upper = torch.tensor(
            [[image_config["B_FACTOR"][1]]], dtype=torch.float32, device=device
        )
        b_factor = zuko.distributions.BoxUniform(lower=lower, upper=upper, ndims=1)

    if isinstance(image_config["SNR"], list) and len(image_config["SNR"]) == 2:
        lower = torch.tensor(
            [[image_config["SNR"][0]]], dtype=torch.float32, device=device
        ).log10()
        upper = torch.tensor(
            [[image_config["SNR"][1]]], dtype=torch.float32, device=device
        ).log10()
        snr = zuko.distributions.BoxUniform(lower=lower, upper=upper, ndims=1)

    amp = zuko.distributions.BoxUniform(
        lower=torch.tensor([[image_config["AMP"]]], dtype=torch.float32, device=device),
        upper=torch.tensor([[image_config["AMP"]]], dtype=torch.float32, device=device),
        ndims=1,
    )

    index_prior = zuko.distributions.BoxUniform(
        lower=torch.tensor([0], dtype=torch.float32, device=device),
        upper=torch.tensor([max_index], dtype=torch.float32, device=device),
    )

    return ImagePrior(index_prior, sigma, defocus, b_factor, snr, amp, device=device)


class QuaternionPrior:
    def __init__(self, device) -> None:
        self.device = device

    def sample(self, shape) -> torch.Tensor:
        quats = torch.stack([gen_quat().to(self.device) for _ in range(shape[0])], dim=0)
        return quats


class ImagePrior:
    def __init__(
        self,
        index_prior,
        sigma_prior,
        defocus_prior,
        b_factor_prior,
        snr_prior,
        amp_prior,
        device,
    ) -> None:
        self.priors = [
            index_prior,
            QuaternionPrior(device),
            sigma_prior,
            defocus_prior,
            b_factor_prior,
            amp_prior,
            snr_prior,
        ]

    def sample(self, shape) -> torch.Tensor:
        samples = [prior.sample(shape) for prior in self.priors]
        return samples


class PriorDataset(IterableDataset):
    def __init__(
        self,
        prior: Distribution,
        batch_shape: torch.Size = (),
    ):
        super().__init__()

        self.prior = prior
        self.batch_shape = batch_shape

    def __iter__(self):
        while True:
            theta = self.prior.sample(self.batch_shape)
            yield theta


class PriorLoader(DataLoader):
    def __init__(
        self,
        prior: Distribution,
        batch_size: int = 2**8,  # 256
        **kwargs,
    ):
        super().__init__(
            PriorDataset(prior, batch_shape=(batch_size,)),
            batch_size=None,
            **kwargs,
        )
