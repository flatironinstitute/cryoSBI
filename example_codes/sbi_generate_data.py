# Numerical libraries
import numpy as np
from scipy.spatial.transform import Rotation
import sys
import json

# MD Stuff
import MDAnalysis as mda

# SBI
import torch
from sbi import utils as utils
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi

import configparser

def gen_quat():
    # Sonya's code
    # Generates a single quaternion

    count = 0
    while count < 1:

        quat = np.random.uniform(
            -1, 1, 4
        )  # note this is a half-open interval, so 1 is not included but -1 is
        norm = 1 / np.sqrt(np.sum(quat**2))

        if 0.2 <= norm <= 1.0:
            quat *= norm
            count += 1

    return quat


def gen_img(coord):

    n_atoms = coord.shape[1]
    norm = 1 / (2 * np.pi * image_params["SIGMA"] ** 2 * n_atoms)

    grid_min = -image_params["PIXEL_SIZE"] * (image_params["N_PIXELS"] - 1) * 0.5
    grid_max = (
        image_params["PIXEL_SIZE"] * (image_params["N_PIXELS"] - 1) * 0.5
        + image_params["PIXEL_SIZE"]
    )

    grid = np.arange(grid_min, grid_max, image_params["PIXEL_SIZE"])

    gauss = np.exp(
        -0.5 * (((grid[:, None] - coord[0, :]) / image_params["SIGMA"]) ** 2)
    )[:, None] * np.exp(
        -0.5 * (((grid[:, None] - coord[1, :]) / image_params["SIGMA"]) ** 2)
    )

    image = gauss.sum(axis=2) * norm

    return image


def add_noise(img):

    # mean_image = np.mean(img)
    std_image = np.std(img)

    mask = np.abs(img) > 0.5 * std_image

    signal_mean = np.mean(img[mask])
    signal_std = np.std(img[mask])

    noise_std = signal_std / np.sqrt(image_params["SNR"])
    noise = np.random.normal(loc=signal_mean, scale=noise_std, size=img.shape)

    img_noise = img + noise

    img_noise -= np.mean(img_noise)
    img_noise /= np.std(img_noise)

    # img_noise = np.zeros_like(img)
    # img_noise[mask] = img[mask]

    return img_noise


def simulator(index):

    index = int(np.round(index))

    coord = models[index]

    quat = gen_quat()
    rot_mat = Rotation.from_quat(quat).as_matrix()
    coord = np.matmul(rot_mat, coord)

    image = gen_img(coord)
    image = add_noise(image)

    return image

def main(argv):

    prior_indices = utils.BoxUniform(
        low=1 * torch.ones(1), high=20 * torch.ones(1)
    )
    simulator_sbi, prior_sbi = prepare_for_sbi(simulator, prior_indices)

    n_simulations = simulation_params["N_SIMULATIONS"]
    indices, images = simulate_for_sbi(
        simulator_sbi,
        proposal=prior_sbi,
        num_simulations=n_simulations,
        num_workers=int(argv[1]),
    )

    torch.save(indices, "indices.pt")
    torch.save(images, "images.pt")

def check_inputs():
    
    for section in ["IMAGES", "SIMULATION"]:
        assert section in config.keys(), f"Please provide section {section} in config.ini"

    for key in  ["N_PIXELS", "PIXEL_SIZE", "SNR", "SIGMA"]:
        assert key in image_params.keys(), f"Please provide a value for {key}"

    for key in ["N_SIMULATIONS"]:
        assert key in simulation_params.keys(), f"Please provide a value for {key}"

    return
        

if __name__ == "__main__":

    global config, image_params, simulation_params, models

    config = json.load(open("config.json"))

    image_params = dict(config["IMAGES"])
    simulation_params = config["SIMULATION"]

    check_inputs()

    models = np.load("../all_models.npy")[:, 0]

    main(sys.argv)
