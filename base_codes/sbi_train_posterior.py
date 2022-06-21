import pickle
import argparse
import json
import torch
import numpy as np

from sbi import utils as utils
from sbi.inference import SNPE
from sbi.utils.get_nn_models import posterior_nn


def main(num_workers):

    torch.set_num_threads(num_workers)

    images = torch.load("images.pt")
    indices = torch.load("indices.pt")

    prior_indices = utils.BoxUniform(
        low=1 * torch.ones(1, device=training_params["DEVICE"]),
        high=20 * torch.ones(1, device=training_params["DEVICE"]),
        device=training_params["DEVICE"],
    )

    density_estimator_build_fun = posterior_nn(
        model="maf",
        hidden_features=training_params["HIDDEN_FEATURES"],
        num_transforms=training_params["NUM_TRANSFORMS"],
    )
    inference = SNPE(
        prior=prior_indices,
        density_estimator=density_estimator_build_fun,
        device=training_params["DEVICE"],
    )

    # Train multiple posteriors for different noise levels

    inference = inference.append_simulations(indices, images)

    density_estimator = inference.train()
    posterior = inference.build_posterior(density_estimator)

    with open(training_params["POSTERIOR_NAME"], "wb") as handle:
        pickle.dump(posterior, handle)


def check_inputs():

    for section in ["TRAINING"]:
        assert (
            section in config.keys()
        ), f"Please provide section {section} in config.ini"

    for key in ["HIDDEN_FEATURES", "NUM_TRANSFORMS", "DEVICE"]:
        assert key in training_params.keys(), f"Please provide a value for {key}"

    if "POSTERIOR_NAME" not in training_params.keys():
        training_params["POSTERIOR_NAME"] = "posterior.pkl"

    if "cuda" in training_params["DEVICE"]:
        assert (
            torch.cuda.is_available()
        ), "Your device is cuda but there is no GPU available"

    return


if __name__ == "__main__":

    global config, training_params

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

    training_params = dict(config["TRAINING"])

    check_inputs()

    main(args.num_workers)
