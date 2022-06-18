import pickle
import sys
import torch
from sbi import utils as utils
from sbi.inference import SNPE
from sbi.utils.get_nn_models import posterior_nn


def main(argv):

    torch.set_num_threads(int(argv[1]))

    images = torch.load("images_1e6_128_noise_01.pt")
    indices = torch.load("indices_1e6_128_noise_01.pt")

    sufix_posterior = "_128_noise_01_100_8"

    prior_indices = utils.BoxUniform(
        low=1 * torch.ones(1), high=20 * torch.ones(1)
    )

    density_estimator_build_fun = posterior_nn(
        model="maf", hidden_features=100, num_transforms=8
    )
    inference = SNPE(
        prior=prior_indices, density_estimator=density_estimator_build_fun
    )

    # Train multiple posteriors for different noise levels

    inference = inference.append_simulations(indices, images)

    density_estimator = inference.train()
    posterior = inference.build_posterior(density_estimator)

    with open("posterior" + sufix_posterior + ".pkl", "wb") as handle:
        pickle.dump(posterior, handle)


if __name__ == "__main__":

    main(sys.argv)
