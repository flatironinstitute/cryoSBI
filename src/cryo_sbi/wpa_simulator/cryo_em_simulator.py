from typing import Union, Callable
import json
import numpy as np
import torch

from cryo_sbi.wpa_simulator.ctf import apply_ctf
from cryo_sbi.wpa_simulator.image_generation import project_density
from cryo_sbi.wpa_simulator.noise import add_noise, get_snr
from cryo_sbi.wpa_simulator.normalization import gaussian_normalize_image
from cryo_sbi.inference.priors import get_image_priors
from cryo_sbi.wpa_simulator.validate_image_config import check_image_params
from cryo_sbi.utils.micrograph_utils import RandomMicrographPatches
from cryo_sbi.utils.image_utils import NormalizeIndividual


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
    noise=True,
    ctf=True,
    ):
    """
    Simulates a bacth of cryo-electron microscopy (cryo-EM) images of a set of given coars-grained models.

    Args:
        models (torch.Tensor): A tensor of coars grained models (num_models, 3, num_beads).
        index (torch.Tensor): A tensor of indices to select the models to simulate.
        quaternion (torch.Tensor): A tensor of quaternions to rotate the models.
        sigma (float): The standard deviation of the Gaussian kernel used to project the density.
        shift (torch.Tensor): A tensor of shifts to apply to the models.
        defocus (float): The defocus value of the contrast transfer function (CTF).
        b_factor (float): The B-factor of the CTF.
        amp (float): The amplitude contrast of the CTF.
        snr (float): The signal-to-noise ratio of the simulated image.
        num_pixels (int): The number of pixels in the simulated image.
        pixel_size (float): The size of each pixel in the simulated image.

    Returns:
        torch.Tensor: A tensor of the simulated cryo-EM image.
    """
    models_selected = models[index.round().long().flatten()]
    image = project_density(
        models_selected,
        quaternion,
        sigma,
        shift,
        num_pixels,
        pixel_size,
    )
    if ctf:
        image = apply_ctf(image, defocus, b_factor, amp, pixel_size)
    if noise:
        image = add_noise(image, snr)
    image = gaussian_normalize_image(image)
    return image


class CryoEmSimulator:
    def __init__(self, config_fname: str, device: str = "cpu"):
        self._device = device
        self._load_params(config_fname)
        self._load_models()
        self._priors = get_image_priors(self.max_index, self._config, device=device)
        self._num_pixels = torch.tensor(
            self._config["N_PIXELS"], dtype=torch.float32, device=device
        )
        self._pixel_size = torch.tensor(
            self._config["PIXEL_SIZE"], dtype=torch.float32, device=device
        )
        self._micrograph_loader = None
    
        
    def _init_micrograph_loader(self, micrographs, patch_size, num_noise_samples):
        if self._micrograph_loader is None:
            self._micrograph_loader = RandomMicrographPatches(
                micro_graphs=micrographs,
                patch_size=patch_size,
                transform=NormalizeIndividual(),
                max_iter=num_noise_samples,
            )
        else:
            self._micrograph_loader._max_iter = num_noise_samples
        

    def _load_params(self, config_fname: str) -> None:
        """
        Loads the parameters from the config file into a dictionary.

        Args:
            config_fname (str): Path to the configuration file.

        Returns:
            None
        """

        config = json.load(open(config_fname))
        check_image_params(config)
        self._config = config

    def _load_models(self) -> None:
        """
        Loads the models from the model file specified in the config file.

        Returns:
            None

        """
        if self._config["MODEL_FILE"].endswith("npy"):
            models = (
                torch.from_numpy(
                    np.load(self._config["MODEL_FILE"]),
                )
                .to(self._device)
                .to(torch.float32)
            )
        elif self._config["MODEL_FILE"].endswith("pt"):
            models = (
                torch.load(self._config["MODEL_FILE"])
                .to(self._device)
                .to(torch.float32)
            )

        else:
            raise NotImplementedError(
                "Model file format not supported. Please use .npy or .pt."
            )

        self._models = models

        assert self._models.ndim == 3, "Models are not of shape (models, 3, atoms)."
        assert self._models.shape[1] == 3, "Models are not of shape (models, 3, atoms)."

    @property
    def max_index(self) -> int:
        """
        Returns the maximum index of the model file.

        Returns:
            int: Maximum index of the model file.
        """
        return len(self._models) - 1

    def simulate(self, num_sim, indices=None, return_parameters=False, batch_size=None, noise=True, ctf=True):
        """
        Simulate cryo-EM images using the specified models and prior distributions.

        Args:
            num_sim (int): The number of images to simulate.
            indices (torch.Tensor, optional): The indices of the images to simulate. If None, all images are simulated.
            return_parameters (bool, optional): Whether to return the sampled parameters used for simulation.
            batch_size (int, optional): The batch size to use for simulation. If None, all images are simulated in a single batch.

        Returns:
            torch.Tensor or tuple: The simulated images as a tensor of shape (num_sim, num_pixels, num_pixels),
            and optionally the sampled parameters as a tuple of tensors.
        """

        parameters = self._priors.sample((num_sim,))
        indices = parameters[0] if indices is None else indices
        if indices is not None:
            assert isinstance(
                indices, torch.Tensor
            ), "Indices are not a torch.tensor, converting to torch.tensor."
            assert (
                indices.dtype == torch.float32
            ), "Indices are not a torch.float32, converting to torch.float32."
            assert (
                indices.ndim == 2
            ), "Indices are not a 2D tensor, converting to 2D tensor. With shape (batch_size, 1)."
            parameters[0] = indices

        images = []
        if batch_size is None:
            batch_size = num_sim
        for i in range(0, num_sim, batch_size):
            batch_indices = indices[i : i + batch_size]
            batch_parameters = [param[i : i + batch_size] for param in parameters[1:]]
            batch_images = cryo_em_simulator(
                self._models,
                batch_indices,
                *batch_parameters,
                self._num_pixels,
                self._pixel_size,
                noise=noise,
                ctf=ctf,
            )
            images.append(batch_images.cpu())

        images = torch.cat(images, dim=0)

        if return_parameters:
            return images.cpu(), parameters
        else:
            return images.cpu()

    def simulate_with_micrograph_noise(self, num_sim, micrographs, indices=None, return_parameters=False, batch_size=None, ctf=True, snr=0.0001):
            self._init_micrograph_loader(micrographs, self._config["N_PIXELS"], num_noise_samples=num_sim)
            images_and_maybe_params = self.simulate(
                num_sim=num_sim,
                indices=indices,
                return_parameters=return_parameters,
                batch_size=batch_size,
                noise=False,
                ctf=ctf
            )
            if return_parameters:
                images, parameters = images_and_maybe_params
            else:
                images = images_and_maybe_params
            print("finished simulating images, drawing noise samples...")
            noise_samples = []
            for noise_sample in self._micrograph_loader:
                noise_samples.append(noise_sample)
            noise_samples = torch.cat(noise_samples, dim=0)

            print("finished drawing noise samples, adding noise to images...")
            noise_samples = noise_samples / snr

            images = images + noise_samples
            images = gaussian_normalize_image(images)


            return images
        

