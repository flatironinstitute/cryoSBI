from __future__ import annotations

import torch
import zuko
from torch.utils.data import DataLoader, IterableDataset

from cryo_sbi.simulator.image_generation import gen_quat, gen_rot_matrix


def _box_uniform(lo: float, hi: float, device: str) -> zuko.distributions.BoxUniform:
    """Scalar BoxUniform with shape (1, 1) bounds for correct sample shape (B, 1, 1)."""
    t = lambda v: torch.tensor([[v]], dtype=torch.float32, device=device)
    return zuko.distributions.BoxUniform(lower=t(lo), upper=t(hi), ndims=1)


# ---------------------------------------------------------------------------
# Priors
# ---------------------------------------------------------------------------

class IndexPrior:
    """Uniform categorical prior over model indices, optionally with representatives."""

    def __init__(self, num_models: int, num_representatives: int | None = None, device: str = "cpu"):
        self._model_dist = torch.distributions.Categorical(
            probs=torch.full((num_models,), 1.0 / num_models, device=device)
        )
        self._rep_dist = (
            torch.distributions.Categorical(
                probs=torch.full((num_representatives,), 1.0 / num_representatives, device=device)
            )
            if num_representatives is not None
            else None
        )

    def sample(self, shape: tuple) -> torch.Tensor:
        idx = self._model_dist.sample(shape)
        if self._rep_dist is not None:
            return torch.stack([idx, self._rep_dist.sample(shape)], dim=1)
        return idx


class QuaternionPrior:
    """
    Uniform random rotation prior. If fixed_quat is given, always returns that quaternion
    (useful for testing/debugging).
    """

    def __init__(self, device: str, fixed_quat: list | None = None):
        self._device = device
        self._fixed = torch.tensor(fixed_quat, dtype=torch.float32, device=device) if fixed_quat else None

    def sample(self, shape: tuple) -> torch.Tensor:
        if self._fixed is not None:
            return self._fixed.unsqueeze(0).expand(shape[0], -1)
        return torch.stack([gen_quat().to(self._device) for _ in range(shape[0])])


class ImagePrior:
    """Joint prior over all single-particle image parameters."""

    def __init__(
        self,
        index_prior: IndexPrior,
        quaternion_prior: QuaternionPrior,
        sigma_prior,
        shift_prior,
        defocus_prior,
        b_factor_prior,
        amp_prior,
        snr_prior,
    ):
        self._priors = [
            index_prior, quaternion_prior,
            sigma_prior, shift_prior,
            defocus_prior, b_factor_prior,
            amp_prior, snr_prior,
        ]

    def sample(self, shape: tuple) -> list[torch.Tensor]:
        return [p.sample(shape) for p in self._priors]

    @classmethod
    def from_config(
        cls,
        num_models: int,
        num_representatives: int | None,
        image_config,
        device: str = "cuda",
    ) -> "ImagePrior":
        """Build an ImagePrior from an OmegaConf DictConfig or plain dict."""
        sigma    = list(image_config.sigma)
        shift    = float(image_config.shift)
        defocus  = list(image_config.defocus)
        b_factor = list(image_config.b_factor)
        snr      = list(image_config.snr)
        amp      = float(image_config.amp)

        shift_prior = zuko.distributions.BoxUniform(
            lower=torch.tensor([-shift, -shift], dtype=torch.float32, device=device),
            upper=torch.tensor([ shift,  shift], dtype=torch.float32, device=device),
            ndims=1,
        )

        rotations = getattr(image_config, "rotations", None)

        return cls(
            index_prior=IndexPrior(num_models, num_representatives, device),
            quaternion_prior=QuaternionPrior(
                device,
                fixed_quat=rotations if rotations and len(rotations) == 4 else None,
            ),
            sigma_prior    = _box_uniform(sigma[0], sigma[1], device),
            shift_prior    = shift_prior,
            defocus_prior  = _box_uniform(defocus[0], defocus[1], device),
            b_factor_prior = _box_uniform(b_factor[0], b_factor[1], device),
            amp_prior      = _box_uniform(amp, amp, device),
            snr_prior      = _box_uniform(
                torch.tensor(snr[0]).log10().item(),
                torch.tensor(snr[1]).log10().item(),
                device,
            ),
        )


def get_image_priors(
    num_models: int, num_representatives: int | None, image_config, device: str = "cuda"
) -> ImagePrior:
    """Thin wrapper around ImagePrior.from_config for backward compatibility."""
    return ImagePrior.from_config(num_models, num_representatives, image_config, device)


# ---------------------------------------------------------------------------
# Ellipsoid fitting
# ---------------------------------------------------------------------------

def fit_ellipsoids(models: torch.Tensor) -> torch.Tensor:
    """
    Fit a bounding ellipsoid to each model via PCA of its atom coordinates.

    Args:
        models: (N, 3, n_atoms) or (N, R, 3, n_atoms). If 4-D, the first
                representative is used. NaN / Inf atoms are excluded.

    Returns:
        torch.Tensor: (N, 3) ellipsoid semi-axes in Angstrom, on CPU.
    """
    if models.ndim == 4:
        models = models[:, 0]

    radii = []
    for coords in models:                                  # (3, n_atoms)
        valid = coords[:, torch.isfinite(coords).all(dim=0)]
        centered = valid - valid.mean(dim=1, keepdim=True)
        cov = centered @ centered.T / centered.shape[1]   # (3, 3)
        radii.append(torch.linalg.eigvalsh(cov).clamp(min=0).sqrt())
    return torch.stack(radii)                              # (N, 3)


# ---------------------------------------------------------------------------
# Multi-particle prior
# ---------------------------------------------------------------------------

class MultiParticleImagePrior:
    """
    Extends ImagePrior with background particle sampling and placement.

    All work runs on CPU prior-loader workers. Overlap detection uses the
    Alfano-Greer criterion with precomputed ellipsoid semi-axes: a candidate
    is accepted when its projected bounding radius clears all previously
    placed particles by at least exclusion_radius.
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
        garbage_class: bool = False,
        min_garbage: int = 2,
        max_garbage: int = 10,
        num_models: int = 1,
    ):
        self.base_prior = base_prior
        self.ellipsoid_radii = ellipsoid_radii.cpu()
        self.n_bg_min = n_bg_min
        self.n_bg_max = n_bg_max
        self.exclusion_radius = exclusion_radius
        self.max_placement_attempts = max_placement_attempts
        self._half_pad_ang = n_pixels * padding_factor * pixel_size / 2.0

        self.garbage_class = garbage_class
        self.min_garbage = min_garbage
        self.max_garbage = max_garbage
        self.n_slots = max(n_bg_max, max_garbage - 1) if garbage_class else n_bg_max
        self.p_garbage = 1.0 / (num_models + 1) if garbage_class else 0.0

    @staticmethod
    def _projected_radius(semi_axes: torch.Tensor, quat: torch.Tensor) -> float:
        """
        Bounding circle radius of the ellipsoid projected onto the xy-plane.

        Constructs the 3-D covariance Σ = R diag(a²,b²,c²) Rᵀ, takes its
        2×2 xy-block, and returns sqrt(λ_max).
        """
        R = gen_rot_matrix(quat.unsqueeze(0))[0]
        cov2 = (R @ torch.diag(semi_axes ** 2) @ R.T)[:2, :2]
        return float(torch.linalg.eigvalsh(cov2).max().clamp(min=0).sqrt())

    def _place_background(
        self, i, n_to_place, has_reps, accepted_centers, accepted_radii,
        bg_indices, bg_quats, bg_sigma, bg_centers, bg_mask, placed_start=0,
    ) -> int:
        """Place n_to_place background particles with collision detection. Returns number placed."""
        placed = placed_start
        for _ in range(n_to_place):
            for _ in range(self.max_placement_attempts):
                b_idx_t, b_quat, b_sig, *_ = self.base_prior.sample((1,))
                b_idx = int(b_idx_t[0, 0] if b_idx_t.ndim == 2 else b_idx_t[0])
                b_r   = self._projected_radius(self.ellipsoid_radii[b_idx], b_quat[0])
                center = (torch.rand(2) * 2 - 1) * self._half_pad_ang

                if all(
                    torch.norm(center - c).item() > b_r + r + self.exclusion_radius
                    for c, r in zip(accepted_centers, accepted_radii)
                ):
                    bg_indices[i, placed] = b_idx_t[0] if has_reps else b_idx
                    bg_quats[i, placed]   = b_quat[0]
                    bg_sigma[i, placed]   = b_sig[0]
                    bg_centers[i, placed] = center
                    bg_mask[i, placed]    = True
                    accepted_centers.append(center)
                    accepted_radii.append(b_r)
                    placed += 1
                    break
        return placed

    def sample(self, shape: tuple) -> list[torch.Tensor]:
        """
        Sample a batch of multi-particle image parameters.

        Returns 14 tensors:
            fg_indices, fg_quats, fg_sigma, fg_shift,
            fg_defocus, fg_b_factor, fg_amp, fg_snr,
            bg_indices, bg_quats, bg_sigma, bg_centers, bg_mask,
            garbage_mask
        """
        B = shape[0]
        fg_indices, fg_quats, fg_sigma, fg_shift, \
            fg_defocus, fg_b_factor, fg_amp, fg_snr = self.base_prior.sample(shape)

        has_reps  = fg_indices.ndim == 2
        idx_shape = (B, self.n_slots, 2) if has_reps else (B, self.n_slots)
        bg_indices = torch.zeros(idx_shape, dtype=torch.long)
        bg_quats   = torch.zeros(B, self.n_slots, 4)
        bg_sigma   = torch.ones(B, self.n_slots, 1, 1)   # ones: masked slots must not cause div-by-0
        bg_centers = torch.zeros(B, self.n_slots, 2)
        bg_mask    = torch.zeros(B, self.n_slots, dtype=torch.bool)
        garbage_mask = torch.zeros(B, dtype=torch.bool)

        for i in range(B):
            fg_idx = int(fg_indices[i, 0] if has_reps else fg_indices[i])
            accepted_centers = [torch.zeros(2)]
            accepted_radii   = [self._projected_radius(self.ellipsoid_radii[fg_idx], fg_quats[i])]

            is_garbage = self.garbage_class and (torch.rand(1).item() < self.p_garbage)

            if is_garbage:
                garbage_mask[i] = True
                n_garbage = int(torch.randint(self.min_garbage, self.max_garbage + 1, (1,)).item())
                # fg slot already holds 1 random structure; place n_garbage - 1 more in bg slots
                n_bg_to_place = max(0, n_garbage - 1)
                self._place_background(
                    i, n_bg_to_place, has_reps, accepted_centers, accepted_radii,
                    bg_indices, bg_quats, bg_sigma, bg_centers, bg_mask,
                )
            else:
                n_bg = int(torch.randint(self.n_bg_min, self.n_bg_max + 1, (1,)).item())
                self._place_background(
                    i, n_bg, has_reps, accepted_centers, accepted_radii,
                    bg_indices, bg_quats, bg_sigma, bg_centers, bg_mask,
                )

        return [fg_indices, fg_quats, fg_sigma, fg_shift,
                fg_defocus, fg_b_factor, fg_amp, fg_snr,
                bg_indices, bg_quats, bg_sigma, bg_centers, bg_mask,
                garbage_mask]


# ---------------------------------------------------------------------------
# DataLoader wrappers
# ---------------------------------------------------------------------------

class PriorDataset(IterableDataset):
    def __init__(self, prior: ImagePrior | MultiParticleImagePrior, batch_shape: tuple):
        super().__init__()
        self.prior = prior
        self.batch_shape = batch_shape

    def __iter__(self):
        while True:
            yield self.prior.sample(self.batch_shape)


class PriorLoader(DataLoader):
    def __init__(self, prior: ImagePrior | MultiParticleImagePrior, batch_size: int = 256, **kwargs):
        super().__init__(
            PriorDataset(prior, batch_shape=(batch_size,)),
            batch_size=None,
            **kwargs,
        )
