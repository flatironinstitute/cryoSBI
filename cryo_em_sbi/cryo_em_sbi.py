from cryo_em_sbi.simulating import image_generation
from cryo_em_sbi.utils import validate_config
from cryo_em_sbi.preprocessing import preprocessing

import json
import pickle
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from sbi import utils as utils
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
from sbi.utils.get_nn_models import posterior_nn


class CryoEmSbi:
    def __init__(self, config_fname):

        self._load_params(config_fname)
        self._load_models()

        self._set_prior_and_simulator_simulation()
        self._set_prior_and_simulator_analysis()

    def _load_params(self, config_fname):

        config = json.load(open(config_fname))
        validate_config.check_params(config)
        self.config = config

        return

    def _load_models(self):

        if "hsp90" in self.config["SIMULATION"]["MODEL_FILE"]:
            self.models = np.load(self.config["SIMULATION"]["MODEL_FILE"])[:, 0]

        if "square" in self.config["SIMULATION"]["MODEL_FILE"]:
            self.models = np.transpose(
                np.load(self.config["SIMULATION"]["MODEL_FILE"]).diagonal(), [2, 0, 1]
            )

        return

    def _simulator(self, index):

        index = int(torch.round(index))

        coord = self.models[index]

        if self.config["SIMULATION"]["ROTATIONS"]:

            quat = image_generation.gen_quat()
            rot_mat = Rotation.from_quat(quat).as_matrix()
            coord = np.matmul(rot_mat, coord)

        image = image_generation.gen_img(coord, self.config["IMAGES"])

        image = torch.tensor(
            image.reshape(-1, 1), device=self.config["SIMULATION"]["DEVICE"]
        )

        return image

    def _analysis_simulator(self, index):

        index = int(torch.round(index))

        coord = self.models[index]

        if self.config["SIMULATION"]["ROTATIONS"]:

            quat = image_generation.gen_quat()
            rot_mat = Rotation.from_quat(quat).as_matrix()
            coord = np.matmul(rot_mat, coord)

        image = image_generation.gen_img(coord, self.config["IMAGES"])

        if self.config["PREPROCESSING"]["SHIFT"]:
            image = preprocessing.pad_image(image, self.config["IMAGES"])

        if self.config["PREPROCESSING"]["CTF"]:
            image = preprocessing.apply_ctf(
                image, self.config["IMAGES"], self.config["PREPROCESSING"]
            )

        if self.config["PREPROCESSING"]["SHIFT"]:
            image = preprocessing.apply_random_shift(image, self.config["IMAGES"])

        if self.config["PREPROCESSING"]["NOISE"]:
            image = preprocessing.add_noise(image, self.config["PREPROCESSING"])

        image = preprocessing.gaussian_normalize_image(image)

        image = torch.tensor(
            image.reshape(-1, 1), device=self.config["TRAINING"]["DEVICE"]
        )

        return image

    def _get_prior(self):

        prior = utils.BoxUniform(
            low=0 * torch.ones(1, device=self.config["SIMULATION"]["DEVICE"]),
            high=19 * torch.ones(1, device=self.config["SIMULATION"]["DEVICE"]),
            device=self.config["SIMULATION"]["DEVICE"],
        )

        return prior

    def _set_prior_and_simulator_simulation(self):

        prior = utils.BoxUniform(
            low=0 * torch.ones(1, device=self.config["SIMULATION"]["DEVICE"]),
            high=19 * torch.ones(1, device=self.config["SIMULATION"]["DEVICE"]),
            device=self.config["SIMULATION"]["DEVICE"],
        )

        self.simulator, self.prior = prepare_for_sbi(self._simulator, prior)

        return

    def _set_prior_and_simulator_analysis(self):

        prior = utils.BoxUniform(
            low=0 * torch.ones(1, device=self.config["TRAINING"]["DEVICE"]),
            high=19 * torch.ones(1, device=self.config["TRAINING"]["DEVICE"]),
            device=self.config["TRAINING"]["DEVICE"],
        )

        self.simulator_analysis, self.prior_analysis = prepare_for_sbi(
            self._analysis_simulator, prior
        )

        return

    def update_config(self, config):

        validate_config.check_params(config)
        self.config = config

        self._load_models()

        self._set_prior_and_simulator_simulation()

        return

    def simulate(
        self, num_workers, fname_indices="indices.pt", fname_images="images.pt"
    ):

        indices, images = simulate_for_sbi(
            self.simulator,
            proposal=self.prior,
            num_simulations=self.config["SIMULATION"]["N_SIMULATIONS"],
            num_workers=num_workers,
        )

        torch.save(indices, fname_indices)
        torch.save(images, fname_images)

        return indices, images

    def preprocess(
        self,
        indices,
        images,
        fname_output_indices="indices_training.pt",
        fname_output_images="images_training.pt",
    ):

        indices = indices
        images = images

        if self.config["PREPROCESSING"]["SHIFT"]:
            images = preprocessing.pad_dataset(
                images, self.config["SIMULATION"], self.config["IMAGES"]
            )

        if self.config["PREPROCESSING"]["CTF"]:
            images = preprocessing.apply_ctf_to_dataset(
                images,
                self.config["SIMULATION"],
                self.config["IMAGES"],
                self.config["PREPROCESSING"],
            )

        if self.config["PREPROCESSING"]["SHIFT"]:
            images = preprocessing.shift_dataset(
                images, self.config["SIMULATION"], self.config["IMAGES"]
            )

        if self.config["PREPROCESSING"]["NOISE"]:
            images = preprocessing.add_noise_to_dataset(
                images, self.config["SIMULATION"], self.config["PREPROCESSING"]
            )

        images = preprocessing.normalize_dataset(images, self.config["SIMULATION"])

        indices = indices.to(self.config["TRAINING"]["DEVICE"])
        images = images.to(self.config["TRAINING"]["DEVICE"])

        torch.save(indices, fname_output_indices)
        torch.save(images, fname_output_images)

        return indices, images

    def train_posterior(
        self,
        num_workers,
        fname_indices="indices_training.pt",
        fname_images="images_training.pt",
    ):

        torch.set_num_threads(num_workers)

        indices = torch.load(fname_indices)
        images = torch.load(fname_images)

        density_estimator_build_fun = posterior_nn(
            model=self.config["TRAINING"]["MODEL"],
            hidden_features=self.config["TRAINING"]["HIDDEN_FEATURES"],
            num_transforms=self.config["TRAINING"]["NUM_TRANSFORMS"],
        )
        inference = SNPE(
            prior=self.prior_analysis,
            density_estimator=density_estimator_build_fun,
            device=self.config["TRAINING"]["DEVICE"],
        )

        inference = inference.append_simulations(indices, images)

        density_estimator = inference.train()
        posterior = inference.build_posterior(density_estimator)

        with open(self.config["TRAINING"]["POSTERIOR_NAME"], "wb") as handle:
            pickle.dump(posterior, handle)

        return posterior
