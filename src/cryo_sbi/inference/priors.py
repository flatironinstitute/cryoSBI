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

        assert lower > 0.0, "The lower bound for B_FACTOR must be positive."
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

    amp_prior = zuko.distributions.BoxUniform(
        lower=torch.tensor([[image_config["AMP"]]], dtype=torch.float32, device=device),
        upper=torch.tensor([[image_config["AMP"]]], dtype=torch.float32, device=device),
        ndims=1,
    )

    # preload models to layout
    models = torch.load(image_config["MODEL_FILE"])
    if isinstance(models, list):
        print("using multiple models per bin")
        index_prior = MultiBinIndexPrior(*get_bin_layout(models), device)
    else:
        index_prior = zuko.distributions.BoxUniform(
            lower=torch.tensor([0], dtype=torch.float32, device=device),
            upper=torch.tensor([max_index], dtype=torch.float32, device=device),
        )

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
        device=device,
    )


def get_bin_layout(models):
    models_per_bin = []
    layout = []
    index_to_cv = []
    start = 0
    for i, m in enumerate(models):
        models_per_bin.append(m.shape[0]) 
        layout.append([j for j in range(start, start+m.shape[0])])
        index_to_cv += [i for _ in range(m.shape[0])]
        start  = start + m.shape[0]
    return torch.tensor(models_per_bin), layout, torch.tensor(index_to_cv, dtype=torch.float32)


class MultiBinIndexPrior:
    def __init__(self, models_per_bin, layout, index_to_cv, device) -> None:
        self.num_bins = len(models_per_bin)
        self.models_per_bin = models_per_bin
        self.index_to_cv = index_to_cv
        self.layout = layout
        self.device = device

    def sample(self, shape) -> torch.Tensor:
        samples = []
        idx_cv = torch.randint(0, self.num_bins, shape, device="cpu").flatten()
        for i in range(len(idx_cv)):
            idx = torch.randint(0, self.models_per_bin[idx_cv[i]], (1,), device="cpu")
            samples.append(self.layout[idx_cv[i]][idx])
        
        return torch.tensor(samples).reshape(-1, 1).to(torch.long).to(self.device)


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
        device,
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
