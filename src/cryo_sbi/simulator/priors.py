import torch
import zuko
from omegaconf import ListConfig
from torch.distributions.distribution import Distribution
from torch.utils.data import DataLoader, IterableDataset

from cryo_sbi.simulator.image_generation import gen_rot_matrix


def gen_quat() -> torch.Tensor:
    """
    Generate a random unit quaternion.

    Returns:
        torch.Tensor: Random unit quaternion of shape (4,)
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

        Returns:
            torch.Tensor: 2D tensor (index, representative) if num_representatives is set,
                          else 1D tensor of model indices.
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
    num_models: int, num_representatives: int, image_config, device="cuda"
) -> "ImagePrior":
    """
    Build an ImagePrior from a config. Accepts both OmegaConf DictConfig and plain dicts.
    Config keys are lowercase (e.g. image_config.sigma, image_config.shift, ...).
    """
    # Support both attribute-style (OmegaConf) and dict-style access
    def _get(cfg, key):
        if hasattr(cfg, key):
            return getattr(cfg, key)
        return cfg[key]

    sigma = _get(image_config, "sigma")
    if isinstance(sigma, (list, tuple, ListConfig)) and len(sigma) == 2:
        lower = torch.tensor([[sigma[0]]], dtype=torch.float32, device=device)
        upper = torch.tensor([[sigma[1]]], dtype=torch.float32, device=device)
        assert lower <= upper, "sigma lower bound must be <= upper bound"
        sigma_prior = zuko.distributions.BoxUniform(lower=lower, upper=upper, ndims=1)

    shift = _get(image_config, "shift")
    shift_prior = zuko.distributions.BoxUniform(
        lower=torch.tensor([-shift, -shift], dtype=torch.float32, device=device),
        upper=torch.tensor([shift, shift], dtype=torch.float32, device=device),
        ndims=1,
    )

    defocus = _get(image_config, "defocus")
    if isinstance(defocus, (list, tuple, ListConfig)) and len(defocus) == 2:
        lower = torch.tensor([[defocus[0]]], dtype=torch.float32, device=device)
        upper = torch.tensor([[defocus[1]]], dtype=torch.float32, device=device)
        assert lower > 0.0, "defocus lower bound must be positive"
        assert lower <= upper, "defocus lower bound must be <= upper bound"
        defocus_prior = zuko.distributions.BoxUniform(lower=lower, upper=upper, ndims=1)

    b_factor = _get(image_config, "b_factor")
    if isinstance(b_factor, (list, tuple, ListConfig)) and len(b_factor) == 2:
        lower = torch.tensor([[b_factor[0]]], dtype=torch.float32, device=device)
        upper = torch.tensor([[b_factor[1]]], dtype=torch.float32, device=device)
        assert lower > 0.0, "b_factor lower bound must be positive"
        assert lower <= upper, "b_factor lower bound must be <= upper bound"
        b_factor_prior = zuko.distributions.BoxUniform(lower=lower, upper=upper, ndims=1)

    snr = _get(image_config, "snr")
    if isinstance(snr, (list, tuple, ListConfig)) and len(snr) == 2:
        lower = torch.tensor([[snr[0]]], dtype=torch.float32, device=device).log10()
        upper = torch.tensor([[snr[1]]], dtype=torch.float32, device=device).log10()
        assert lower <= upper, "snr lower bound must be <= upper bound"
        snr_prior = zuko.distributions.BoxUniform(lower=lower, upper=upper, ndims=1)

    amp = _get(image_config, "amp")
    amp_prior = zuko.distributions.BoxUniform(
        lower=torch.tensor([[amp]], dtype=torch.float32, device=device),
        upper=torch.tensor([[amp]], dtype=torch.float32, device=device),
        ndims=1,
    )

    index_prior = IndexPrior(num_models, num_representatives, device)
    quaternion_prior = QuaternionPrior(device)

    rotations = None
    if hasattr(image_config, "rotations"):
        rotations = image_config.rotations
    elif isinstance(image_config, dict) and "rotations" in image_config:
        rotations = image_config["rotations"]

    if rotations and isinstance(rotations, (list, tuple)) and len(rotations) == 4:
        quaternion_prior = QuaternionTestPrior(rotations, device)

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
        return torch.stack(
            [gen_quat().to(self.device) for _ in range(shape[0])], dim=0
        )


class QuaternionTestPrior:
    def __init__(self, quat, device) -> None:
        self.device = device
        self.quat = torch.tensor(quat, device=device)

    def sample(self, shape) -> torch.Tensor:
        return torch.stack([self.quat for _ in range(shape[0])], dim=0)


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

    def sample(self, shape) -> list:
        return [prior.sample(shape) for prior in self.priors]


def fit_ellipsoids(models: torch.Tensor) -> torch.Tensor:
    """
    Fit bounding ellipsoids to each model via PCA of atom coordinates.

    For each model the 3×3 covariance matrix of its atom positions is computed;
    the square-roots of its eigenvalues give the three semi-axes of the
    best-fit ellipsoid in Angstrom.

    Args:
        models: (N, 3, n_atoms) or (N, R, 3, n_atoms). If 4D, the first
                representative ([:,0]) is used. NaN/Inf atoms are ignored.

    Returns:
        torch.Tensor: (N, 3) semi-axes in Angstrom, on CPU.
    """
    if models.ndim == 4:
        models = models[:, 0]  # use first representative

    N = models.shape[0]
    radii = []
    for i in range(N):
        coords = models[i]                                  # (3, n_atoms)
        finite = torch.isfinite(coords).all(dim=0)
        coords = coords[:, finite]                          # (3, n_valid)
        center = coords.mean(dim=1, keepdim=True)
        coords_c = coords - center                          # (3, n_valid)
        cov = (coords_c @ coords_c.T) / coords_c.shape[1]  # (3, 3)
        eigs = torch.linalg.eigvalsh(cov)                   # ascending
        radii.append(torch.sqrt(eigs.clamp(min=0)))
    return torch.stack(radii, dim=0)  # (N, 3)


class MultiParticleImagePrior:
    """
    Prior for multi-particle cryo-EM images.

    Wraps an `ImagePrior` (single-particle) and extends its `sample()` to also
    sample background particle parameters and resolve placement conflicts on the
    CPU worker, using precomputed bounding ellipsoids for fast overlap detection.

    The placement check is:
        ||c_new - c_acc|| > r_new + r_acc + exclusion_radius
    where r is the projected bounding radius of each particle's ellipsoid under
    its sampled rotation (computed analytically as sqrt of the largest eigenvalue
    of the 2×2 projected covariance).
    """

    def __init__(
        self,
        base_prior: ImagePrior,
        ellipsoid_radii: torch.Tensor,
        n_bg_min: int,
        n_bg_max: int,
        n_pixels: int,
        padding_factor: int,
        pixel_size: float,
        exclusion_radius: float,
        max_placement_attempts: int = 200,
    ) -> None:
        self.base_prior = base_prior
        self.ellipsoid_radii = ellipsoid_radii.cpu()   # (N, 3)
        self.n_bg_min = n_bg_min
        self.n_bg_max = n_bg_max
        self.n_px_pad = n_pixels * padding_factor
        self.pixel_size = pixel_size
        self.exclusion_radius = exclusion_radius
        self.max_placement_attempts = max_placement_attempts
        self._half_pad_ang = self.n_px_pad * pixel_size / 2.0

    @staticmethod
    def _projected_radius(semi_axes: torch.Tensor, quat: torch.Tensor) -> float:
        """
        Bounding circle radius of an ellipsoid projected onto the xy-plane.

        The 3D covariance Σ = R @ diag(a²,b²,c²) @ R.T; the projected 2×2
        covariance is its top-left block; the bounding radius is the sqrt of
        the largest eigenvalue of that block.

        Args:
            semi_axes: (3,) semi-axes in Angstrom.
            quat: (4,) unit quaternion.

        Returns:
            float bounding radius in Angstrom.
        """
        # Build rotation matrix (reuse gen_rot_matrix)
        R = gen_rot_matrix(quat.unsqueeze(0))[0]       # (3, 3)
        cov3 = R @ torch.diag(semi_axes ** 2) @ R.T   # (3, 3)
        cov2 = cov3[:2, :2]                            # (2, 2)
        eigs = torch.linalg.eigvalsh(cov2)             # (2,)
        return float(torch.sqrt(eigs.max().clamp(min=0)))

    def sample(self, shape) -> list:
        """
        Sample a batch of multi-particle image parameters.

        Returns a list of tensors:
            [fg_indices, fg_quats, fg_sigma, fg_shift,
             fg_defocus, fg_b_factor, fg_amp, fg_snr,
             bg_indices, bg_quats, bg_sigma, bg_centers, bg_mask]
        """
        batch_size = shape[0]
        fg_params = self.base_prior.sample(shape)
        (fg_indices, fg_quats, fg_sigma, fg_shift,
         fg_defocus, fg_b_factor, fg_amp, fg_snr) = fg_params

        # Allocate padded bg tensors
        has_reps = fg_indices.ndim == 2
        if has_reps:
            bg_indices = torch.zeros(batch_size, self.n_bg_max, 2, dtype=torch.long)
        else:
            bg_indices = torch.zeros(batch_size, self.n_bg_max, dtype=torch.long)
        bg_quats   = torch.zeros(batch_size, self.n_bg_max, 4)
        bg_sigma   = torch.ones(batch_size, self.n_bg_max, 1, 1)
        bg_centers = torch.zeros(batch_size, self.n_bg_max, 2)
        bg_mask    = torch.zeros(batch_size, self.n_bg_max, dtype=torch.bool)

        for i in range(batch_size):
            fg_idx = int(fg_indices[i, 0] if has_reps else fg_indices[i])
            fg_r   = self._projected_radius(self.ellipsoid_radii[fg_idx], fg_quats[i])

            accepted_centers = [torch.zeros(2)]
            accepted_radii   = [fg_r]

            n_bg = int(torch.randint(self.n_bg_min, self.n_bg_max + 1, (1,)).item())
            placed = 0

            for _ in range(n_bg):
                for _ in range(self.max_placement_attempts):
                    bg_p = self.base_prior.sample((1,))
                    b_idx_t, b_quat, b_sig, *_ = bg_p
                    b_idx = int(b_idx_t[0, 0] if b_idx_t.ndim == 2 else b_idx_t[0])
                    b_r   = self._projected_radius(self.ellipsoid_radii[b_idx], b_quat[0])

                    center = (torch.rand(2) * 2 - 1) * self._half_pad_ang

                    no_overlap = all(
                        torch.norm(center - c).item() > b_r + r_acc + self.exclusion_radius
                        for c, r_acc in zip(accepted_centers, accepted_radii)
                    )
                    if no_overlap:
                        if has_reps:
                            bg_indices[i, placed] = b_idx_t[0]
                        else:
                            bg_indices[i, placed] = b_idx
                        bg_quats[i, placed]   = b_quat[0]
                        bg_sigma[i, placed]   = b_sig[0]
                        bg_centers[i, placed] = center
                        bg_mask[i, placed]    = True
                        accepted_centers.append(center)
                        accepted_radii.append(b_r)
                        placed += 1
                        break

        return [fg_indices, fg_quats, fg_sigma, fg_shift,
                fg_defocus, fg_b_factor, fg_amp, fg_snr,
                bg_indices, bg_quats, bg_sigma, bg_centers, bg_mask]


class PriorDataset(IterableDataset):
    def __init__(self, prior: ImagePrior, batch_shape: torch.Size = ()):
        super().__init__()
        self.prior = prior
        self.batch_shape = batch_shape

    def __iter__(self):
        while True:
            yield self.prior.sample(self.batch_shape)


class PriorLoader(DataLoader):
    def __init__(self, prior: ImagePrior, batch_size: int = 256, **kwargs):
        super().__init__(
            PriorDataset(prior, batch_shape=(batch_size,)),
            batch_size=None,
            **kwargs,
        )
