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

def image_formation(
    fg_models,
    fg_quats,
    fg_sigma,
    fg_shift,
    fg_defocus,
    fg_b_factor,
    fg_amp,
    fg_snr,
    bg_models,
    bg_quats,
    bg_sigma,
    bg_centers,
    bg_mask,
    num_pixels_padded,
    pixel_size,
    pad_start: int,
    pad_end: int,
) -> torch.Tensor:
    """
    Unified image formation pipeline for single- and multi-particle images.
    All model selection is done by the caller — this function is pure tensor math.

    When n_bg_max=0, bg_models has shape (0, 3, n_atoms): the background
    project_density call is a no-op and bg_density is an all-zero tensor.

    Args:
        fg_models (torch.Tensor): Foreground model coords, shape (B, 3, n_atoms).
        fg_quats (torch.Tensor): Foreground rotations, shape (B, 4).
        fg_sigma (torch.Tensor): Foreground Gaussian width, shape (B, 1, 1).
        fg_shift (torch.Tensor): Foreground in-plane shifts, shape (B, 2).
        fg_defocus (torch.Tensor): CTF defocus, shape (B, 1).
        fg_b_factor (torch.Tensor): CTF B-factor, shape (B, 1).
        fg_amp (torch.Tensor): Amplitude contrast, shape (B, 1).
        fg_snr (torch.Tensor): log10 SNR, shape (B, 1).
        bg_models (torch.Tensor): Background model coords, shape (B*n_bg_max, 3, n_atoms).
            Shape (0, 3, n_atoms) when n_bg_max=0.
        bg_quats (torch.Tensor): Background rotations, shape (B*n_bg_max, 4).
        bg_sigma (torch.Tensor): Background Gaussian width, shape (B*n_bg_max, 1, 1).
        bg_centers (torch.Tensor): Background in-plane shifts, shape (B*n_bg_max, 2).
        bg_mask (torch.Tensor): Valid-slot mask, shape (B, n_bg_max), dtype bool.
        num_pixels_padded (torch.Tensor): Scalar — padded canvas side length.
        pixel_size (torch.Tensor): Scalar — pixel size in Angstrom.
        pad_start (int): Start index for center-crop.
        pad_end (int): End index for center-crop.

    Returns:
        torch.Tensor: Normalized images, shape (B, pad_end-pad_start, pad_end-pad_start).
    """
    B = fg_models.shape[0]
    n_bg_max = bg_mask.shape[1]

    fg_density = project_density(
        fg_models, fg_quats, fg_sigma, fg_shift, num_pixels_padded, pixel_size
    )

    bg_density_flat = project_density(
        bg_models, bg_quats, bg_sigma, bg_centers, num_pixels_padded, pixel_size
    )

    n_px_pad = bg_density_flat.shape[-1]
    bg_density = (
        bg_density_flat.reshape(B, n_bg_max, n_px_pad, n_px_pad)
        * bg_mask.unsqueeze(-1).unsqueeze(-1)
    ).sum(dim=1)

    image = apply_ctf(
        fg_density + bg_density, fg_defocus, fg_b_factor, fg_amp, pixel_size
    )
    image = image[:, pad_start:pad_end, pad_start:pad_end]
    image = add_noise(image, fg_snr)
    return gaussian_normalize_image(image)



class CryoEmSimulator:
    """
    Cryo-EM image simulator supporting both single- and multi-particle images.

    Single-particle mode: set n_bg_max=0 and padding_factor=1 in the config
    (or omit them — these are the defaults). Multi-particle mode: set n_bg_max>0
    and padding_factor>1.

    Two simulation entry points:
    - simulate(*parameters): hot training path — accepts pre-sampled parameter
      tensors from PriorLoader workers; all image formation runs on self._device.
    - sample_and_simulate(): interactive/eval path — samples parameters internally
      then calls simulate().
    """

    def __init__(self, config, device: str = "cpu"):
        """
        Args:
            config: Path to a YAML/JSON config file (str) or an OmegaConf DictConfig.
            device: PyTorch device string.
        """
        self._device = device
        self._load_params(config)
        self._load_models()

        self._n_bg_max  = int(getattr(self._config, "n_bg_max", 0))
        padding_factor  = int(getattr(self._config, "padding_factor", 1))
        self._n_px_pad  = int(self._config.n_pixels) * padding_factor
        self._pad_start = (self._n_px_pad - int(self._config.n_pixels)) // 2
        self._pad_end   = self._pad_start + int(self._config.n_pixels)

        self._num_pixels = torch.tensor(
            self._config.n_pixels, dtype=torch.float32, device=device
        )
        self._num_pixels_padded = torch.tensor(
            self._n_px_pad, dtype=torch.float32, device=device
        )
        self._pixel_size = torch.tensor(
            self._config.pixel_size, dtype=torch.float32, device=device
        )

        self.garbage_class = bool(getattr(self._config, "garbage_class", False))

        ellipsoid_radii = fit_ellipsoids(self._models_cpu)
        self._priors = MultiParticleImagePrior(
            base_prior=get_image_priors(
                self.num_models, self.num_representatives, self._config, device="cpu"
            ),
            ellipsoid_radii=ellipsoid_radii,
            n_bg_min=int(getattr(self._config, "n_bg_min", 0)),
            n_bg_max=self._n_bg_max,
            n_pixels=int(self._config.n_pixels),
            padding_factor=padding_factor,
            pixel_size=float(self._config.pixel_size),
            exclusion_radius=float(getattr(self._config, "exclusion_radius", 0.0)),
            max_placement_attempts=int(
                getattr(self._config, "max_placement_attempts", 200)
            ),
            garbage_class=self.garbage_class,
            min_garbage=int(getattr(self._config, "min_garbage", 2)),
            max_garbage=int(getattr(self._config, "max_garbage", 10)),
            num_models=self.num_models,
        )
        self._n_bg_max = self._priors.n_slots

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
        # Unwrap top-level 'simulation' key if present (YAML files may nest config under it)
        if "simulation" in self._config and len(self._config) == 1:
            self._config = self._config.simulation

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
        Hot path: generate images from pre-sampled parameters (from MultiParticleImagePrior.sample()).
        Moves all tensors to self._device, selects model coordinates, then calls image_formation().

        Parameters (positional, 14 tensors from MultiParticleImagePrior.sample()):
            fg_indices, fg_quats, fg_sigma, fg_shift,
            fg_defocus, fg_b_factor, fg_amp, fg_snr,
            bg_indices, bg_quats, bg_sigma, bg_centers, bg_mask,
            garbage_mask

        Returns:
            torch.Tensor: Images on self._device, shape (B, n_pixels, n_pixels).
        """
        dev = self._device
        B = parameters[0].shape[0]
        n_flat = B * self._n_bg_max

        (fg_indices, fg_quats, fg_sigma, fg_shift,
         fg_defocus, fg_b_factor, fg_amp, fg_snr,
         bg_indices, bg_quats, bg_sigma, bg_centers, bg_mask,
         _garbage_mask) = (
            t.to(dev, non_blocking=True) for t in parameters
        )

        fg_models = self._select_models(fg_indices)
        bg_models = self._select_models(bg_indices.reshape(n_flat, *bg_indices.shape[2:]))

        return image_formation(
            fg_models, fg_quats, fg_sigma, fg_shift,
            fg_defocus, fg_b_factor, fg_amp, fg_snr,
            bg_models,
            bg_quats.reshape(n_flat, 4),
            bg_sigma.reshape(n_flat, 1, 1),
            bg_centers.reshape(n_flat, 2),
            bg_mask,
            self._num_pixels_padded, self._pixel_size,
            self._pad_start, self._pad_end,
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
            indices: Optional fixed fg index tensor; if None, sampled from prior.
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
