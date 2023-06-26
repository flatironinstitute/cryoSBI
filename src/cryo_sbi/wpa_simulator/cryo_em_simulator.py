from typing import Union, Callable
import torch
import numpy as np
import json
from scipy.spatial.transform import Rotation

from cryo_sbi.wpa_simulator.ctf import calc_ctf, apply_ctf
from cryo_sbi.wpa_simulator.image_generation import gen_img
from cryo_sbi.wpa_simulator.noise import add_noise
from cryo_sbi.wpa_simulator.normalization import gaussian_normalize_image
from cryo_sbi.wpa_simulator.padding import pad_image
from cryo_sbi.wpa_simulator.shift import apply_no_shift, apply_random_shift
from cryo_sbi.wpa_simulator.validate_image_config import check_params
from cryo_sbi.wpa_simulator.implicit_water import add_noise_field


class CryoEmSimulator:
    """Simulator for cryo-EM images.

    Args:
        config_fname (str): Path to the configuration file.

    Attributes:
        config (dict): Configuration parameters.
        models (np.ndarray): The models to use for image generation.
        rot_mode (str): The rotation mode to use. Can be "random", "list" or None.
        quaternions (np.ndarray): The quaternions to use for image generation.
        add_noise (bool): function which adds noise to images. Defaults to Gaussian noise.
    """

    def __init__(self, config_fname: str, add_noise: Callable = add_noise):
        self._load_params(config_fname)
        self._load_models()
        self._pad_width = int(np.ceil(self.config["N_PIXELS"] * 0.1)) + 1
        self.add_noise = add_noise

    def _load_params(self, config_fname: str) -> None:
        """
        Loads the parameters from the config file into a dictionary.

        Args:
            config_fname (str): Path to the configuration file.

        Returns:
            None
        """

        config = json.load(open(config_fname))
        check_params(config)
        self.config = config

    def _load_models(self) -> None:
        """
        Loads the models from the model file specified in the config file.

        Returns:
            None

        """

        self.models = np.load(self.config["MODEL_FILE"])
        if self.models.ndim == 3:
            self.model = self.models[0]
        
        if self.models.ndim == 4:
            self.model = self.models[0, 0]

        print(self.config["MODEL_FILE"])


    def _simulator_with_quat(
        self, quaternion: np.ndarray, seed: Union[None, int] = None
    ) -> torch.Tensor:
        """
        Simulates an image with a given quaternion.

        Args:
            index (torch.Tensor): Index of the model to use.
            quaternion (np.ndarray): Quaternion to rotate structure.
            seed (Union[None, int], optional): Seed for random number generator. Defaults to None.

        Returns:
            torch.Tensor: Simulated image.
        """

        coord = np.copy(self.model)

        rot_mat = Rotation.from_quat(quaternion).as_matrix()
        coord = np.matmul(rot_mat, coord)

        image = gen_img(coord, self.config)
        image = pad_image(image, self.config)

        if self.config["CTF"]:
            image = apply_ctf(image, calc_ctf(self.config))

        if self.config["NOISE"]:
            image = self.add_noise(image, self.config, seed)

        if self.config["SHIFT"]:
            image = apply_random_shift(image, self.config, seed)
        else:
            image = apply_no_shift(image, self.config)

        image = gaussian_normalize_image(image)

        return image.to(dtype=torch.float)

    def simulator(
        self, quat: torch.Tensor, seed: Union[None, int] = None
    ) -> torch.Tensor:
        """
        Simulates an image with parameters specified in the config file.

        Args:
            index (torch.Tensor): Index of the model to use.
            seed (Union[None, int], optional): Seed for random number generator. Defaults to None.

        Returns:
            torch.Tensor: Simulated image.
        """

        #print(quat)
        quat = torch.reshape(quat, (4,)).numpy()
        image = self._simulator_with_quat(quat, seed)

        return image
