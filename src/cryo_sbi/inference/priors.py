import torch
import zuko
from torch.distributions.distribution import Distribution
from torch.utils.data import DataLoader, IterableDataset


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


class IndexPrior:
    def __init__(
        self, num_models: int, num_representatives: int = None, device="cpu"
    ) -> None:
        self.num_models = num_models
        self.num_representatives = num_representatives
        self.device = device

        self.index_prior = torch.distributions.Categorical(
            probs=torch.tensor(
                [1 / self.num_models for _ in range(self.num_models)],
                device=device,
            )
        )

        if num_representatives is not None:
            self.representatives_prior = torch.distributions.Categorical(
                probs=torch.tensor(
                    [
                        1 / self.num_representatives
                        for _ in range(self.num_representatives)
                    ],
                    device=device,
                )
            )

    def sample(self, shape) -> torch.Tensor:
        """
        Sample indices from the prior distribution.

        Args:
            shape (tuple): Shape of the samples to be generated.

        Returns:
            torch.Tensor: If num_representatives is not None, returns a 2D tensor where
                  the first column contains samples from the index prior and
                  the second column contains samples from the representatives prior.
                  Otherwise, returns a 1D tensor with samples from the index prior.
        """
        if self.num_representatives is not None:
            return torch.stack(
                [
                    self.index_prior.sample(shape),
                    self.representatives_prior.sample(shape),
                ],
                dim=1,
            )
        else:
            return self.index_prior.sample(shape)


def get_image_priors(
    num_models: int, num_representatives: int, image_config: dict, device="cuda"
) -> zuko.distributions.BoxUniform:

    if isinstance(image_config["SIGMA"], list) and len(image_config["SIGMA"]) == 2:
        lower = torch.tensor(
            [[image_config["SIGMA"][0]]], dtype=torch.float32, device=device
        )
        upper = torch.tensor(
            [[image_config["SIGMA"][1]]], dtype=torch.float32, device=device
        )

        assert lower <= upper, "Lower bound must be smaller or equal than upper bound."

        sigma_prior = zuko.distributions.BoxUniform(lower=lower, upper=upper, ndims=1)

    shift_prior = zuko.distributions.BoxUniform(
        lower=torch.tensor(
            [-image_config["SHIFT"], -image_config["SHIFT"]],
            dtype=torch.float32,
            device=device,
        ),
        upper=torch.tensor(
            [image_config["SHIFT"], image_config["SHIFT"]],
            dtype=torch.float32,
            device=device,
        ),
        ndims=1,
    )

    if isinstance(image_config["DEFOCUS"], list) and len(image_config["DEFOCUS"]) == 2:
        lower = torch.tensor(
            [[image_config["DEFOCUS"][0]]], dtype=torch.float32, device=device
        )
        upper = torch.tensor(
            [[image_config["DEFOCUS"][1]]], dtype=torch.float32, device=device
        )

        assert lower > 0.0, "The lower bound for DEFOCUS must be positive."
        assert lower <= upper, "Lower bound must be smaller or equal than upper bound."

        defocus_prior = zuko.distributions.BoxUniform(lower=lower, upper=upper, ndims=1)

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

        #assert lower > 0.0, "The lower bound for B_FACTOR must be positive."
        assert lower <= upper, "Lower bound must be smaller or equal than upper bound."

        b_factor_prior = zuko.distributions.BoxUniform(
            lower=lower, upper=upper, ndims=1
        )

    if isinstance(image_config["SNR"], list) and len(image_config["SNR"]) == 2:
        lower = torch.tensor(
            [[image_config["SNR"][0]]], dtype=torch.float32, device=device
        ).log10()
        upper = torch.tensor(
            [[image_config["SNR"][1]]], dtype=torch.float32, device=device
        ).log10()

        assert lower <= upper, "Lower bound must be smaller or equal than upper bound."

        snr_prior = zuko.distributions.BoxUniform(lower=lower, upper=upper, ndims=1)


    if isinstance(image_config["AMP"], list) and len(image_config["AMP"]) == 2:
        assert image_config["AMP"][0] >= 0.0, "The lower bound for AMP must be non-negative."
        assert image_config["AMP"][1] >= image_config["AMP"][0], "The upper bound for AMP must be greater than or equal to the lower bound."
        lower = torch.tensor(
            [[image_config["AMP"][0]]], dtype=torch.float32, device=device
        )
        upper = torch.tensor(
            [[image_config["AMP"][1]]], dtype=torch.float32, device=device
        )

        assert lower <= upper, "Lower bound must be smaller or equal than upper bound."

        amp_prior = zuko.distributions.BoxUniform(lower=lower, upper=upper, ndims=1)
    else:
        amp_prior = zuko.distributions.BoxUniform(
            lower=torch.tensor(
                [[image_config["AMP"]]], dtype=torch.float32, device=device
            ),
            upper=torch.tensor(
                [[image_config["AMP"]]], dtype=torch.float32, device=device
            ),
            ndims=1,
        )

    index_prior = IndexPrior(num_models, num_representatives, device)
    quaternion_prior = QuaternionPrior(device)
    if (
        image_config.get("ROTATIONS")
        and isinstance(image_config["ROTATIONS"], list)
        and len(image_config["ROTATIONS"]) == 4
    ):
        test_quat = image_config["ROTATIONS"]
        quaternion_prior = QuaternionTestPrior(test_quat, device)

    return ImagePrior(
        index_prior,
        quaternion_prior,
        sigma_prior,
        shift_prior,
        defocus_prior,
        b_factor_prior,
        amp_prior,
        snr_prior,
    )


class QuaternionPrior:
    def __init__(self, device) -> None:
        self.device = device

    def sample(self, shape) -> torch.Tensor:
        quats = torch.stack(
            [gen_quat().to(self.device) for _ in range(shape[0])], dim=0
        )
        return quats


class QuaternionTestPrior:
    def __init__(self, quat, device) -> None:
        self.device = device
        self.quat = torch.tensor(quat, device=device)

    def sample(self, shape) -> torch.Tensor:
        quats = torch.stack([self.quat for _ in range(shape[0])], dim=0)
        return quats


class ImagePrior:
    def __init__(
        self,
        index_prior,
        quaternion_prior,
        sigma_prior,
        shift_prior,
        defocus_prior,
        b_factor_prior,
        amp_prior,
        snr_prior,
    ) -> None:
        self.priors = [
            index_prior,
            quaternion_prior,
            sigma_prior,
            shift_prior,
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
