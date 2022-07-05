import json
import argparse
import numpy as np
from scipy.spatial.transform import Rotation

# SBI
import torch
from sbi import utils as utils
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi


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


def gen_img(coord, image_params):

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


def simulator(index):

    index = int(torch.round(index))

    coord = models[index]

    quat = gen_quat()
    rot_mat = Rotation.from_quat(quat).as_matrix()
    coord = np.matmul(rot_mat, coord)

    image = gen_img(coord)

    image = torch.tensor(image.reshape(-1, 1), device=simulation_params["DEVICE"])

    return image


def main(num_workers):

    prior_indices = utils.BoxUniform(
        low=0 * torch.ones(1, device=simulation_params["DEVICE"]),
        high=19 * torch.ones(1, device=simulation_params["DEVICE"]),
        device=simulation_params["DEVICE"],
    )

    simulator_sbi, prior_sbi = prepare_for_sbi(simulator, prior_indices)

    n_simulations = simulation_params["N_SIMULATIONS"]
    indices, images = simulate_for_sbi(
        simulator_sbi,
        proposal=prior_sbi,
        num_simulations=n_simulations,
        num_workers=num_workers,
    )

    torch.save(indices, "indices.pt")
    torch.save(images, "images.pt")


def check_inputs():

    for section in ["IMAGES", "SIMULATION"]:
        assert (
            section in config.keys()
        ), f"Please provide section {section} in config.ini"

    for key in ["N_PIXELS", "PIXEL_SIZE", "SIGMA"]:
        assert key in image_params.keys(), f"Please provide a value for {key}"

    for key in ["N_SIMULATIONS", "MODEL_FILE", "DEVICE"]:
        assert key in simulation_params.keys(), f"Please provide a value for {key}"

    return


if __name__ == "__main__":

    global config, image_params, simulation_params, models

    parser = argparse.ArgumentParser(
        description="Input file and number of workers",
    )
    parser.add_argument(
        "--num_workers",
        dest="num_workers",
        type=int,
        help="Number of processes for SBI",
        required=True,
    )
    parser.add_argument(
        "--config",
        dest="config_fname",
        type=str,
        help="Name of the config file",
        required=True,
    )
    args = parser.parse_args()

    config = json.load(open(args.config_fname))

    image_params = config["IMAGES"])
    simulation_params = config["SIMULATION"]

    check_inputs()

    if "hsp90" in simulation_params["MODEL_FILE"]:
        models = np.load(simulation_params["MODEL_FILE"])[:, 0]

    if "square" in simulation_params["MODEL_FILE"]:
        models = np.transpose(
            np.load(simulation_params["MODEL_FILE"]).diagonal(), [2, 0, 1]
        )

    main(args.num_workers)
