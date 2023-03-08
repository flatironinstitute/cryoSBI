import torch
import numpy as np
import json

from scipy.spatial.transform import Rotation
import image_generation
import normalization
import validate_image_config
import shift
import ctf
import padding
import noise


class CryoEmSimulator:
    def __init__(self, config_fname):

        self._load_params(config_fname)
        self._load_models()
        self.rot_mode = None
        self.quaternions = None
        self._config_rotations()
        self._ctf = ctf.calc_ctf(self.config)
        self._pad_width = int(np.ceil(self.config["N_PIXELS"] * 0.1)) + 1


    def _load_params(self, config_fname):

        config = json.load(open(config_fname))
        validate_image_config.check_params(config)
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
            self.quaternions = np.loadtxt(
                self.config["ROTATIONS"], skiprows=1
            )

            assert (
                self.quaternions.shape[1] == 4
            ), "Quaternion shape is not 4. Corrupted file?"


    def get_max_index(self):
        return len(self.models) - 1


    def simulator(self, index):

        index = int(torch.round(index))
        coord = self.models[index]

        if self.rot_mode == "random":
            quat = image_generation.gen_quat()
            rot_mat = Rotation.from_quat(quat).as_matrix()
            coord = np.matmul(rot_mat, coord)

        elif self.rot_mode == "list":
            quat = self.quaternions[np.random.randint(0, self.quaternions.shape[0])]
            rot_mat = Rotation.from_quat(quat).as_matrix()
            coord = np.matmul(rot_mat, coord)

        image = image_generation.gen_img(coord, self.config)
        image = padding.pad_image(image, self.config)

        if self.config["CTF"]:
            image = ctf.apply_ctf(image, self._ctf)

        if self.config["NOISE"]:
            image = noise.add_noise(image, self.config)

        if self.config["SHIFT"]:
            image = shift.apply_random_shift(image, self.config)
        else:
            image = shift.apply_no_shift(image, self.config)

        image = normalization.gaussian_normalize_image(image)

        return image.to(dtype=torch.float)


    def _simulator_with_quat(self, index_quat):

        index = int(torch.round(index_quat[0]))
        coord = self.models[index]
        rot_mat = Rotation.from_quat(index_quat[1:]).as_matrix()
        coord = np.matmul(rot_mat, coord)

        image = image_generation.gen_img(coord, self.config)

        image = padding.pad_image(image, self.config)

        if self.config["CTF"]:
            image = ctf.apply_ctf(image, self._ctf)

        if self.config["NOISE"]:
            image = noise.add_noise(image, self.config)

        if self.config["SHIFT"]:
            image = shift.apply_random_shift(image, self.config)
        else:
            image = shift.apply_no_shift(image, self.config)

        image = normalization.gaussian_normalize_image(image)

        return image.to(dtype=torch.float)