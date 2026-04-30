from __future__ import annotations

import warnings

import torch
import zuko
from torch.utils.data import DataLoader, IterableDataset

from cryo_sbi.simulator.image_generation import gen_quats, gen_rot_matrix


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
        if fixed_quat is None:
            self._fixed = None
        else:
            q = torch.tensor(fixed_quat, dtype=torch.float32, device=device)
            norm = q.norm()
            if norm == 0:
                raise ValueError("fixed_quat must have non-zero norm.")
            self._fixed = q / norm

    def sample(self, shape: tuple) -> torch.Tensor:
        if self._fixed is not None:
            return self._fixed.unsqueeze(0).expand(shape[0], -1)
        return gen_quats(shape[0], device=self._device)


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
            upper=torch.tensor([shift, shift], dtype=torch.float32, device=device),
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
    Fit a PCA-aligned bounding shape to each model.

    The principal axes come from PCA of the centered atom positions, and each
    semi-axis is set to the maximum |projection| of any atom along that axis.
    These are the half-extents of the bounding *box* in the PCA frame — every
    atom is inside the box, but atoms near the box corners can still lie
    outside the inscribed ellipsoid `(p₁/a)² + (p₂/b)² + (p₃/c)² ≤ 1`.

    A ``UserWarning`` is issued for any model where more than 15% of the atom
    centers fall outside the implied ellipsoid; in that case the FG SNR mask
    will under-cover the molecule and ``simulation.snr_mask_radius_angstrom``
    should be set explicitly.

    Args:
        models: (N, 3, n_atoms) or (N, R, 3, n_atoms). If 4-D, the first
                representative is used. NaN / Inf atoms are excluded.

    Returns:
        torch.Tensor: (N, 3) PCA-frame semi-axes in Angstrom, on CPU.
    """
    if models.ndim == 4:
        models = models[:, 0]

    radii = []
    for i, coords in enumerate(models):                         # (3, n_atoms)
        valid = coords[:, torch.isfinite(coords).all(dim=0)]    # drop NaN/Inf
        centered = valid - valid.mean(dim=1, keepdim=True)
        # Principal axes via PCA of the atom-coordinate covariance.
        # eigh returns columns-as-eigenvectors; eigvecs.T projects onto them.
        _, eigvecs = torch.linalg.eigh(centered @ centered.T / centered.shape[1])
        projected = eigvecs.T @ centered                        # (3, n_atoms)
        # Per-axis semi-axis = max |projection| along that axis (box half-extent).
        half_extents = projected.abs().max(dim=1).values        # (3,)

        # Sanity: how many atoms fall outside the implied ellipsoid?
        # An atom at projected coords p is inside iff Σ_k (p_k / s_k)² ≤ 1.
        # Clamp the divisor: a degenerate model (single atom, collinear, etc.)
        # has half_extents == 0 along some axis and would otherwise produce
        # 0/0 = NaN, silently suppressing the >15% warning.
        eps = torch.finfo(half_extents.dtype).eps
        ellipsoid_metric = (
            (projected / half_extents.clamp_min(eps).unsqueeze(-1)) ** 2
        ).sum(dim=0)
        n_total = projected.shape[1]
        n_outside = int((ellipsoid_metric > 1.0).sum().item())
        if (half_extents == 0).any():
            warnings.warn(
                f"fit_ellipsoids: model {i} is degenerate (single atom or "
                "collinear/coplanar) — at least one principal-axis extent is "
                "0, so collision detection will not exclude overlap along "
                "that axis.",
                stacklevel=2,
            )
        elif n_total > 0 and n_outside / n_total > 0.15:
            warnings.warn(
                f"fit_ellipsoids: model {i} has {100 * n_outside / n_total:.1f}% "
                f"of atoms outside its bounding ellipsoid (>15% threshold). "
                "The FG SNR mask derived from these radii may under-cover the "
                "molecule; consider setting simulation.snr_mask_radius_angstrom "
                "explicitly.",
                stacklevel=2,
            )
        radii.append(half_extents)
    return torch.stack(radii)                                   # (N, 3)


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
        n_pixels_padded: int,
        pixel_size: float,
        exclusion_radius: float,
        max_placement_attempts: int = 200,
        garbage_class: bool = False,
        min_garbage: int = 2,
        max_garbage: int = 10,
        num_models: int = 1,
    ):
        # MultiParticleImagePrior is CPU-only by design: it runs inside CPU
        # DataLoader workers, and ellipsoid_radii is forced to CPU below. If
        # base_prior was built on CUDA, indexing ellipsoid_radii with a CUDA
        # `fg_idx_flat` later would raise a cryptic device-mismatch error.
        # Fail fast at construction with a clear message instead.
        base_device = base_prior._priors[0]._model_dist.probs.device
        if base_device.type != "cpu":
            raise ValueError(
                f"MultiParticleImagePrior requires a CPU base_prior, got "
                f"device={base_device}. Construct ImagePrior with device='cpu'."
            )
        self.base_prior = base_prior
        self.ellipsoid_radii = ellipsoid_radii.cpu()
        self.n_bg_min = n_bg_min
        self.n_bg_max = n_bg_max
        self.exclusion_radius = exclusion_radius
        self.max_placement_attempts = max_placement_attempts
        self._half_pad_ang = n_pixels_padded * pixel_size / 2.0

        self.garbage_class = garbage_class
        self.min_garbage = min_garbage
        self.max_garbage = max_garbage
        # n_slots reserves enough background-particle storage for either real
        # background images (n_bg_max slots) or garbage images (which use
        # max_garbage-1 slots, since the foreground occupies one of them).
        self.n_slots = max(n_bg_max, max_garbage - 1) if garbage_class else n_bg_max
        # Make the garbage class as likely as any of the num_models real classes:
        # together they form a uniform (num_models + 1)-way distribution.
        self.p_garbage = 1.0 / (num_models + 1) if garbage_class else 0.0

    @staticmethod
    def _projected_radii(semi_axes: torch.Tensor, quats: torch.Tensor) -> torch.Tensor:
        """
        Batched projected bounding-circle radii on the xy-plane.

        Args:
            semi_axes: (N, 3) ellipsoid semi-axes.
            quats: (N, 4) quaternions.

        Returns:
            (N,) projected radii.
        """
        R = gen_rot_matrix(quats)                             # (N, 3, 3)
        diag = torch.diag_embed(semi_axes ** 2)               # (N, 3, 3)
        cov3 = R @ diag @ R.transpose(-1, -2)                 # (N, 3, 3)
        cov2 = cov3[:, :2, :2]                                # (N, 2, 2)
        return torch.linalg.eigvalsh(cov2).max(dim=-1).values.clamp(min=0).sqrt()

    def _sample_pool(self, pool_size: int, has_reps: bool):
        """Pre-sample a pool of background candidates with projected radii."""
        # NB: `_priors` is positional; index 0 = IndexPrior, index 2 = sigma.
        # Order is set in `ImagePrior.__init__`.
        index_prior = self.base_prior._priors[0]
        sigma_prior = self.base_prior._priors[2]

        p_idx_raw = index_prior.sample((pool_size,))
        p_quats = gen_quats(pool_size)
        p_sigma = sigma_prior.sample((pool_size,))
        p_idx_flat = p_idx_raw[:, 0].long() if has_reps else p_idx_raw.long()
        p_radii = self._projected_radii(self.ellipsoid_radii[p_idx_flat], p_quats)
        p_centers = (torch.rand(pool_size, 2) * 2 - 1) * self._half_pad_ang
        return p_idx_raw, p_quats, p_sigma, p_idx_flat, p_radii, p_centers

    def sample(self, shape: tuple) -> list[torch.Tensor]:
        """
        Sample a batch of multi-particle image parameters.

        Returns 14 tensors:
            fg_indices, fg_quats, fg_sigma, fg_shift,
            fg_defocus, fg_b_factor, fg_amp, fg_snr,
            bg_indices, bg_quats, bg_sigma, bg_centers, bg_mask,
            garbage_mask

        Class-label contract (when garbage_class is True):
            ``fg_indices`` plays two roles. As a *model selector* it is always
            a real index in ``[0, num_models)`` — the simulator uses it to pull
            atom coords. As a *class label* it is correct only on non-garbage
            rows; on garbage rows the true label is ``num_models``. Consumers
            that need labels must merge with ``garbage_mask``::

                labels = fg_indices[..., 0].clone() if has_reps else fg_indices.clone()
                labels[garbage_mask] = num_models

            See ``cryo_sbi.training.training`` for the canonical merge.
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

        # Vectorized foreground projected radii
        fg_idx_flat = fg_indices[:, 0].long() if has_reps else fg_indices.long()
        fg_proj_radii = self._projected_radii(
            self.ellipsoid_radii[fg_idx_flat], fg_quats
        )

        # Vectorized garbage / bg-count decisions
        if self.garbage_class:
            garbage_mask = torch.rand(B) < self.p_garbage
            n_bg_counts = torch.randint(self.n_bg_min, self.n_bg_max + 1, (B,))
            n_garbage_counts = torch.randint(self.min_garbage, self.max_garbage + 1, (B,))
            n_to_place = torch.where(
                garbage_mask, (n_garbage_counts - 1).clamp(min=0), n_bg_counts
            )
        else:
            garbage_mask = torch.zeros(B, dtype=torch.bool)
            n_to_place = torch.randint(self.n_bg_min, self.n_bg_max + 1, (B,))

        # Pre-sample candidate pool
        pool_size = max(B * self.n_slots * 5, 1024)
        p_idx_raw, p_quats, p_sigma, p_idx_flat, p_radii, p_centers = \
            self._sample_pool(pool_size, has_reps)
        pool_ptr = 0

        # Pre-allocate per-sample collision buffers
        max_accepted = 1 + self.n_slots
        acc_centers = torch.zeros(max_accepted, 2)
        acc_radii = torch.zeros(max_accepted)

        for i in range(B):
            n = int(n_to_place[i].item())
            if n == 0:
                continue

            # Reset collision state: foreground at origin
            acc_centers[0] = 0.0
            acc_radii[0] = fg_proj_radii[i]
            n_acc = 1

            placed = 0
            for _ in range(n):
                for _ in range(self.max_placement_attempts):
                    # Refill pool if exhausted
                    if pool_ptr >= pool_size:
                        p_idx_raw, p_quats, p_sigma, p_idx_flat, p_radii, p_centers = \
                            self._sample_pool(pool_size, has_reps)
                        pool_ptr = 0

                    p = pool_ptr
                    pool_ptr += 1

                    # Vectorized distance check against all accepted particles
                    dists = torch.norm(acc_centers[:n_acc] - p_centers[p], dim=1)
                    if (dists > p_radii[p] + acc_radii[:n_acc] + self.exclusion_radius).all():
                        bg_indices[i, placed] = p_idx_raw[p] if has_reps else p_idx_flat[p]
                        bg_quats[i, placed]   = p_quats[p]
                        bg_sigma[i, placed]   = p_sigma[p]
                        bg_centers[i, placed] = p_centers[p]
                        bg_mask[i, placed]    = True
                        acc_centers[n_acc] = p_centers[p]
                        acc_radii[n_acc] = p_radii[p]
                        n_acc += 1
                        placed += 1
                        break

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
