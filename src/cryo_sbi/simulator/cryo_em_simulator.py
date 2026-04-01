import numpy as np
import torch
from omegaconf import OmegaConf, DictConfig

from cryo_sbi.simulator.ctf import apply_ctf
from cryo_sbi.simulator.image_generation import project_density
from cryo_sbi.simulator.noise import add_noise
from cryo_sbi.simulator.normalization import gaussian_normalize_image
from cryo_sbi.simulator.priors import (
    fit_ellipsoids,
    get_image_priors,
    MultiParticleImagePrior,
)


def cryo_em_simulator(
    models,
    index,
    quaternion,
    sigma,
    shift,
    defocus,
    b_factor,
    amp,
    snr,
    num_pixels,
    pixel_size,
):
    """
    Low-level functional simulator. Generates a batch of cryo-EM images from
    pre-selected model coordinates and pre-sampled imaging parameters.

    Args:
        models (torch.Tensor): Coarse-grained models (num_models, 3, num_beads).
        index (torch.Tensor): 1D or 2D index tensor selecting models (and representatives).
        quaternion (torch.Tensor): Rotation quaternions, shape (batch, 4).
        sigma (torch.Tensor): Gaussian width per image, shape (batch, 1).
        shift (torch.Tensor): In-plane shifts per image, shape (batch, 2).
        defocus (torch.Tensor): CTF defocus per image, shape (batch, 1).
        b_factor (torch.Tensor): CTF B-factor per image, shape (batch, 1).
        amp (torch.Tensor): Amplitude contrast per image, shape (batch, 1).
        snr (torch.Tensor): log10 SNR per image, shape (batch, 1).
        num_pixels (torch.Tensor): Scalar — image side length in pixels.
        pixel_size (torch.Tensor): Scalar — pixel size in Angstrom.

    Returns:
        torch.Tensor: Normalized cryo-EM images, shape (batch, n_pixels, n_pixels).
    """
    if index.ndim == 2:
        models_selected = models[index[:, 0], index[:, 1]]
    else:
        models_selected = models[index]

    image = project_density(models_selected, quaternion, sigma, shift, num_pixels, pixel_size)
    image = apply_ctf(image, defocus, b_factor, amp, pixel_size)
    image = add_noise(image, snr)
    image = gaussian_normalize_image(image)
    return image


class CryoEmSimulator:
    """
    Single-particle cryo-EM image simulator.

    Two simulation methods:
    - simulate(*parameters): hot path for training — accepts pre-sampled parameter
      tensors from PriorLoader workers; runs image formation on self._device.
    - sample_and_simulate(): eval/interactive path — samples parameters internally
      then calls simulate().
    """

    def __init__(self, config, device: str = "cpu"):
        """
        Args:
            config: Path to a JSON/YAML config file (str) or an OmegaConf DictConfig.
            device: PyTorch device string.
        """
        self._device = device
        self._load_params(config)
        self._load_models()
        self._priors = get_image_priors(
            self.num_models, self.num_representatives, self._config, device="cpu"
        )
        self._num_pixels = torch.tensor(
            self._config.n_pixels, dtype=torch.float32, device=device
        )
        self._pixel_size = torch.tensor(
            self._config.pixel_size, dtype=torch.float32, device=device
        )

    def _load_params(self, config) -> None:
        if isinstance(config, DictConfig):
            self._config = config
        elif isinstance(config, dict):
            self._config = OmegaConf.create(config)
        elif isinstance(config, str):
            self._config = OmegaConf.load(config)
        else:
            raise TypeError(
                f"config must be a path (str), dict, or DictConfig, got {type(config)}"
            )

    def _load_models(self) -> None:
        model_file = self._config.model_file
        if model_file.endswith("npy"):
            models = (
                torch.from_numpy(np.load(model_file))
                .to(self._device)
                .to(torch.float32)
            )
        elif model_file.endswith("pt"):
            models = (
                torch.load(model_file, weights_only=True)
                .to(self._device)
                .to(torch.float32)
            )
        else:
            raise NotImplementedError("Model file must be .npy or .pt")

        self._models = models
        # CPU copy kept for operations that must run on CPU (e.g. ellipsoid fitting)
        self._models_cpu = models.cpu()

        if self._models.ndim == 3:
            self.num_models = self._models.shape[0]
            self.num_representatives = None
        elif self._models.ndim == 4:
            self.num_models = self._models.shape[0]
            self.num_representatives = self._models.shape[1]
        else:
            raise ValueError(
                "Models must have shape (models, 3, atoms) or "
                "(models, representatives, 3, atoms)."
            )
        assert self._models.shape[-2] == 3, "Models must have shape (..., 3, atoms)."

    def _select_models(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.ndim == 2:
            return self._models[indices[:, 0], indices[:, 1]]
        return self._models[indices]

    def simulate(self, *parameters) -> torch.Tensor:
        """
        Hot path: generate images from pre-sampled parameters (from ImagePrior.sample()).
        All tensors are moved to self._device internally.

        Parameters (positional):
            indices:    (batch,) or (batch, 2)
            quaternions: (batch, 4)
            sigma:      (batch, 1)
            shift:      (batch, 2)
            defocus:    (batch, 1)
            b_factor:   (batch, 1)
            amp:        (batch, 1)
            snr:        (batch, 1)

        Returns:
            torch.Tensor: Images on self._device, shape (batch, n_pixels, n_pixels).
        """
        indices, quaternions, sigma, shift, defocus, b_factor, amp, snr = parameters[:8]
        return cryo_em_simulator(
            self._models,
            indices.to(self._device, non_blocking=True),
            quaternions.to(self._device, non_blocking=True),
            sigma.to(self._device, non_blocking=True),
            shift.to(self._device, non_blocking=True),
            defocus.to(self._device, non_blocking=True),
            b_factor.to(self._device, non_blocking=True),
            amp.to(self._device, non_blocking=True),
            snr.to(self._device, non_blocking=True),
            self._num_pixels,
            self._pixel_size,
        )

    def sample_and_simulate(
        self,
        num_sim: int,
        indices=None,
        return_parameters: bool = False,
        batch_size: int = None,
    ):
        """
        Self-contained eval path: sample parameters then simulate.

        Args:
            num_sim: Number of images to generate.
            indices: Optional fixed index tensor; if None, sampled from prior.
            return_parameters: If True, also return the sampled parameters.
            batch_size: Process in sub-batches to manage memory. Defaults to num_sim.

        Returns:
            images (torch.Tensor on CPU), optionally (images, parameters).
        """
        parameters = self._priors.sample((num_sim,))
        if indices is not None:
            assert isinstance(indices, torch.Tensor), "indices must be a torch.Tensor"
            parameters[0] = indices

        if batch_size is None:
            batch_size = num_sim

        images = []
        for i in range(0, num_sim, batch_size):
            batch_params = [p[i : i + batch_size] for p in parameters]
            batch_images = self.simulate(*batch_params)
            images.append(batch_images.cpu())

        images = torch.cat(images, dim=0)

        if return_parameters:
            return images, parameters
        return images


class MultiParticleCryoEmSimulator(CryoEmSimulator):
    """
    Multi-particle cryo-EM simulator.

    Background particle placement and overlap checking happen entirely on CPU
    PriorLoader workers via `MultiParticleImagePrior`. Bounding ellipsoids for
    each model are precomputed once at init and passed to the prior so workers
    need no model coordinates. The `simulate()` method receives ready-made
    padded parameter tensors and performs pure batched GPU image formation.

    Extra config keys (beyond single-particle):
        n_bg_min (int): Minimum number of background particles per image.
        n_bg_max (int): Maximum number of background particles per image.
        padding_factor (int): Padded box side = n_pixels * padding_factor.
        exclusion_radius (float): Minimum centre-to-centre gap in Angstrom.
        max_placement_attempts (int, optional): Retry limit per bg particle (default 200).
    """

    def __init__(self, config, device: str = "cpu"):
        # Loads models, sets self._models / self._models_cpu / self._priors (ImagePrior)
        super().__init__(config, device)

        # Precompute bounding ellipsoid semi-axes for every model (CPU, once)
        ellipsoid_radii = fit_ellipsoids(self._models_cpu)  # (N, 3)

        # Replace the base single-particle prior with the multi-particle one
        self._priors = MultiParticleImagePrior(
            base_prior=get_image_priors(
                self.num_models, self.num_representatives,
                self._config, device="cpu"
            ),
            ellipsoid_radii=ellipsoid_radii,
            n_bg_min=int(self._config.n_bg_min),
            n_bg_max=int(self._config.n_bg_max),
            n_pixels=int(self._config.n_pixels),
            padding_factor=int(self._config.padding_factor),
            pixel_size=float(self._config.pixel_size),
            exclusion_radius=float(self._config.exclusion_radius),
            max_placement_attempts=int(
                getattr(self._config, "max_placement_attempts", 200)
            ),
        )

        self._n_bg_max = int(self._config.n_bg_max)
        self._n_px_pad = int(self._config.n_pixels) * int(self._config.padding_factor)
        self._num_pixels_padded = torch.tensor(
            self._n_px_pad, dtype=torch.float32, device=device
        )
        self._pad_start = (self._n_px_pad - int(self._config.n_pixels)) // 2
        self._pad_end = self._pad_start + int(self._config.n_pixels)

    def simulate(self, *parameters) -> torch.Tensor:
        """
        Generate a batch of multi-particle cryo-EM images from pre-sampled parameters.
        All tensors arrive on CPU from PriorLoader workers and are moved to device here.

        Returns:
            torch.Tensor: Images on self._device, shape (B, n_pixels, n_pixels).
        """
        dev = self._device
        B = parameters[0].shape[0]
        n_flat = B * self._n_bg_max

        (fg_indices, fg_quats, fg_sigma, fg_shift,
         fg_defocus, fg_b_factor, fg_amp, fg_snr,
         bg_indices, bg_quats, bg_sigma, bg_centers, bg_mask) = (
            t.to(dev, non_blocking=True) for t in parameters
        )

        # Project foreground into padded box: (B, P, P)
        fg_density = project_density(
            self._select_models(fg_indices), fg_quats, fg_sigma, fg_shift,
            self._num_pixels_padded, self._pixel_size,
        )

        # Flatten and project all background particles at once: (B*n_bg_max, P, P)
        bg_density_flat = project_density(
            self._select_models(bg_indices.reshape(n_flat, *bg_indices.shape[2:])),
            bg_quats.reshape(n_flat, 4),
            bg_sigma.reshape(n_flat, 1, 1),
            bg_centers.reshape(n_flat, 2),
            self._num_pixels_padded, self._pixel_size,
        )

        # Mask invalid slots, sum over bg dimension: (B, P, P)
        bg_density = (
            bg_density_flat.reshape(B, self._n_bg_max, self._n_px_pad, self._n_px_pad)
            * bg_mask.unsqueeze(-1).unsqueeze(-1)
        ).sum(dim=1)

        # CTF in padded box (periodic BCs), crop, noise, normalize
        image = apply_ctf(fg_density + bg_density, fg_defocus, fg_b_factor, fg_amp, self._pixel_size)
        image = image[:, self._pad_start:self._pad_end, self._pad_start:self._pad_end]
        image = add_noise(image, fg_snr)
        return gaussian_normalize_image(image)
