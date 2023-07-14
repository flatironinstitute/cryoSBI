from typing import Union, Callable
import json
import numpy as np
import torch

from cryo_sbi.wpa_simulator.ctf import apply_ctf
from cryo_sbi.wpa_simulator.image_generation import project_density
from cryo_sbi.wpa_simulator.noise import add_noise
from cryo_sbi.wpa_simulator.normalization import gaussian_normalize_image
from cryo_sbi.inference.priors import get_image_priors
from cryo_sbi.wpa_simulator.validate_image_config import check_image_params


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
    models_selected = models[index.round().long().flatten()]
    image = project_density(
        models_selected,
        quaternion,
        sigma,
        shift,
        num_pixels,
        pixel_size,
    )
    image = apply_ctf(image, defocus, b_factor, amp, pixel_size)
    image = add_noise(image, snr)
    image = gaussian_normalize_image(image)
    return image


class CryoEmSimulator:
    def __init__(self, config_fname: str, device: str = "cpu"):
        self._device = device
        self._load_params(config_fname)
        self._load_models()
        self._priors = get_image_priors(self.max_index, self._config, device=device)
        self._num_pixels = torch.tensor(self._config["N_PIXELS"], dtype=torch.float32, device=device)
        self._pixel_size = torch.tensor(self._config["PIXEL_SIZE"], dtype=torch.float32, device=device)

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
                ).to(self._device).to(torch.float32)
            )
        else:
            models = torch.load(
                self._config["MODEL_FILE"]
            ).to(self._device).to(torch.float32)

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

    def simulate(self, num_sim, indices=None, return_parameters=False):
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
            indices = torch.tensor(indices, dtype=torch.float32)

        images = cryo_em_simulator(
            self._models,
            indices,
            parameters[1],
            parameters[2],
            parameters[3],
            parameters[4],
            parameters[5],
            parameters[6],
            parameters[7],
            self._num_pixels,
            self._pixel_size,
        )

        if return_parameters:
            return images.cpu(), parameters
        else:
            return images.cpu()
