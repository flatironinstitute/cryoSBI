import matplotlib.pyplot as plt
import numpy as np
import torch
import json
from multiprocessing import Pool
from lampe.data import JointLoader
from itertools import islice
from tqdm import tqdm
from lampe.diagnostics import expected_coverage_mc
from lampe.plots import coverage_plot

from cryo_sbi.inference.models import build_models
from cryo_sbi import CryoEmSimulator
from cryo_sbi.inference import priors


file_name = "23_03_21_final_posterior"  # File name
data_dir = "../experiments/benchmark_hsp90/results/raw_data/"
plot_dir = "../experiments/benchmark_hsp90/results/plots/"
config_dir = "../experiments/benchmark_hsp90/"
num_samples_stats = 20000  # Number of simulations for computing posterior stats
num_samples_SBC = 10000  # Number of simulations for SBC
num_posterior_samples_SBC = 4096  # Number of posterior samples for each SBC simulation
num_samples_posterior = 50000  # Number of samples to draw from posterior
batch_size_sampling = 100  # Batch size for sampling posterior
num_workers = 24  # Number of CPU cores
device = "cuda"  # Device for computations
save_data = True
save_figures = True

from cryo_sbi.wpa_simulator.noise import add_colored_noise, add_gradient_snr, add_noise

noise = [add_colored_noise, add_gradient_snr, add_noise]
noise_type = ["colored", "gradient", "gaussian"]


for noise_name, noise_func in zip(noise_type, noise):
    cryosbi = CryoEmSimulator(f"{config_dir}image_params_snr=0.07_defocus=1.5.json")
    cryosbi.add_noise = noise_func
    train_config = json.load(open(config_dir + "resnet18_encoder.json"))
    estimator = build_models.build_npe_flow_model(train_config)
    estimator.load_state_dict(torch.load(config_dir + f"posterior_hsp90.estimator"))
    estimator.cuda()
    estimator.eval()

    loader = JointLoader(
        priors.get_uniform_prior_1d(cryosbi.max_index),
        cryosbi.simulator,
        vectorized=False,
        batch_size=1,
        num_workers=num_workers,
        prefetch_factor=1,
    )

    levels, coverages = expected_coverage_mc(
        estimator.flow,
        (
            (estimator.standardize(theta.cuda()), x.cuda())
            for theta, x in islice(loader, num_samples_SBC)
        ),
        n=num_posterior_samples_SBC,
    )

    torch.save(
        {
            "levels": levels,
            "coverages": coverages,
        },
        f"{data_dir}{file_name}_{noise_name}_SBC.pt",
    )

    print("SBC finished")

    indices = priors.get_uniform_prior_1d(cryosbi.max_index).sample(
        (num_samples_stats,)
    )
    images = torch.stack([cryosbi.simulator(index) for index in indices], dim=0)

    print("Simulation finished")

    theta_samples = []
    with torch.no_grad():
        for batched_images in torch.split(
            images, split_size_or_sections=batch_size_sampling, dim=0
        ):
            samples = estimator.sample(
                batched_images.cuda(non_blocking=True), shape=(num_samples_posterior,)
            ).cpu()
            theta_samples.append(samples.reshape(-1, batch_size_sampling))
        samples = torch.cat(theta_samples, dim=1)

    mean_distance = (samples.mean(dim=0) - indices.reshape(-1)).numpy()
    posterior_quantiles = np.quantile(samples.numpy(), [0.025, 0.975], axis=0)
    confidence_widths = (posterior_quantiles[1] - posterior_quantiles[0]).flatten()

    torch.save(
        {"indices": indices, "images": images, "posterior_samples": samples},
        f"{data_dir}{file_name}_{noise_name}_stats.pt",
    )
