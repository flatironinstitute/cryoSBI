import torch
import numpy as np
import json
from scipy.spatial.transform import Rotation

from cryo_sbi.wpa_simulator.ctf import calc_ctf, apply_ctf
from cryo_sbi.wpa_simulator.image_generation import gen_img, gen_quat
from cryo_sbi.wpa_simulator.noise import add_noise
from cryo_sbi.wpa_simulator.normalization import gaussian_normalize_image
from cryo_sbi.wpa_simulator.padding import pad_image
from cryo_sbi.wpa_simulator.shift import apply_no_shift, apply_random_shift
from cryo_sbi.wpa_simulator.validate_image_config import check_params


class CryoEmSimulator:
    def __init__(self, config_fname):
        self._load_params(config_fname)
        self._load_models()
        self.rot_mode = None
        self.quaternions = None
        self._config_rotations()
        self._pad_width = int(np.ceil(self.config["N_PIXELS"] * 0.1)) + 1


    def _load_params(self, config_fname):
        config = json.load(open(config_fname))
        check_params(config)
        self.config = config


    def _load_models(self):
        if "hsp90" in self.config["MODEL_FILE"]:
            self.models = np.load(self.config["MODEL_FILE"])[:, 0]

        elif "6wxb" in self.config["MODEL_FILE"]:
            self.models = np.load(self.config["MODEL_FILE"])

        elif "square" in self.config["MODEL_FILE"]:
            self.models = np.transpose(
                np.load(self.config["MODEL_FILE"]).diagonal(), [2, 0, 1]
            )
        print(self.config["MODEL_FILE"])

    def _config_rotations(self):
        if isinstance(self.config["ROTATIONS"], bool):
            if self.config["ROTATIONS"]:
                self.rot_mode = "random"

        elif isinstance(self.config["ROTATIONS"], str):
            self.rot_mode = "list"
            self.quaternions = np.loadtxt(self.config["ROTATIONS"], skiprows=1)

            assert (
                self.quaternions.shape[1] == 4
            ), "Quaternion shape is not 4. Corrupted file?"


    @property
    def max_index(self):
        return len(self.models) - 1


    def _simulator_with_quat(self, index, quaternion, seed):
        index = int(torch.round(index))

        coord = np.copy(self.models[index])

        if quaternion is not None:
            rot_mat = Rotation.from_quat(quaternion).as_matrix()
            coord = np.matmul(rot_mat, coord)

        image = gen_img(coord, self.config)
        image = pad_image(image, self.config)

        if self.config["CTF"]:
            image = apply_ctf(image, calc_ctf(self.config))

        if self.config["NOISE"]:
            image = add_noise(image, self.config, seed)

        if self.config["SHIFT"]:
            image = apply_random_shift(image, self.config, seed)
        else:
            image = apply_no_shift(image, self.config)

        image = gaussian_normalize_image(image)

        return image.to(dtype=torch.float)


    def simulator(self, index, seed=None):

        if self.rot_mode == "random":
            quat = gen_quat()
        elif self.rot_mode == "list":
            quat = self.quaternions[np.random.randint(0, self.quaternions.shape[0])]
        else:
            quat = None

        image = self._simulator_with_quat(index, quat, seed)

        return image