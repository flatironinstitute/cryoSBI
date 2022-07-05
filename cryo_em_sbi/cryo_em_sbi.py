import cryo_em_sbi.image_generation as image_gen

import json
import torch
import numpy as np
from scipy.spatial.transform import Rotation
from sbi import utils as utils
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi


class CryoEmSbi:
    def __init__(self, config_fname):

        self.config = self.load_params(config_fname)
        self.check_params()
        self.load_models()

        self.set_prior_and_simulator()

    def load_params(self, config_fname):

        config = json.load(open(config_fname))

        self.image_params = self.config["IMAGES"]
        self.simulation_params = self.config["SIMULATION"]
        self.training_params = self.config["TRAINING"]

        return config, image_params, simulation

    def load_models(self):

        if "hsp90" in self.simulation_params["MODEL_FILE"]:
            self.models = np.load(self.simulation_params["MODEL_FILE"])[:, 0]

        if "square" in self.simulation_params["MODEL_FILE"]:
            self.models = np.transpose(
                np.load(self.simulation_params["MODEL_FILE"]).diagonal(), [2, 0, 1]
            )

    def _simulator_sim(self, index):

        index = int(torch.round(index))

        coord = self.models[index]

        quat = image_gen.gen_quat()
        rot_mat = Rotation.from_quat(quat).as_matrix()
        coord = np.matmul(rot_mat, coord)

        image = image_gen.gen_img(coord)
        image = image_gen.add_noise(image)
        image = image_gen.gaussian_normalization(image)

        image = torch.tensor(
            image.reshape(-1, 1), device=self.simulation_params["DEVICE"]
        )

        return image

    def _simulator_train(self, index):

        index = int(torch.round(index))

        coord = self.models[index]

        quat = image_gen.gen_quat()
        rot_mat = Rotation.from_quat(quat).as_matrix()
        coord = np.matmul(rot_mat, coord)

        image = image_gen.gen_img(coord)
        image = image_gen.add_noise(image)
        image = image_gen.gaussian_normalization(image)

        image = torch.tensor(
            image.reshape(-1, 1), device=self.simulation_params["DEVICE"]
        )

        return image

    def _prep_prior(self):

        self._prior_simulate = utils.BoxUniform(
            low=0 * torch.ones(1, device=self.simulation_params["DEVICE"]),
            high=19 * torch.ones(1, device=self.simulation_params["DEVICE"]),
            device=self.simulation_params["DEVICE"],
        )

        self._prior_train = utils.BoxUniform(
            low=0 * torch.ones(1, device=self.training_params["DEVICE"]),
            high=19 * torch.ones(1, device=self.training_params["DEVICE"]),
            device=self.training_params["DEVICE"],
        )

    def set_prior_and_simulator(self):

        self.simulator_sim, self.prior_sim = prepare_for_sbi(
            self._simulator_sim, self._prior_simulate
        )
        self.simulator_train, self.prior_train = prepare_for_sbi(
            self._simulator_train, self._prior_train
        )
