# Numerical libraries
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

# Plotting
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

# MD Stuff
import MDAnalysis as mda

# Utils
from tqdm import tqdm
import pickle

# SBI
import torch
from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference.base import infer
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
from sbi.utils.get_nn_models import posterior_nn


def gen_quat(size):
    #Sonya's code
    
    #np.random.seed(0)
    quaternions = np.zeros((size, 4))
    count = 0

    while count < size:

        quat = np.random.uniform(-1,1,4) #note this is a half-open interval, so 1 is not included but -1 is
        norm = np.sqrt(np.sum(quat**2))

        if ( 0.2 <= norm <= 1.0 ):
            quaternions[count] = quat/norm
            count += 1

    return quaternions

def gen_img(coord, args_dict):
    
    n_atoms = coord.shape[1]
    norm = 1 / (2 * np.pi * args_dict["SIGMA"]**2 * n_atoms)

    grid_min = -args_dict["PIXEL_SIZE"] * (args_dict["N_PIXELS"] - 1)*0.5
    grid_max = args_dict["PIXEL_SIZE"] * (args_dict["N_PIXELS"] - 1)*0.5 + args_dict["PIXEL_SIZE"]

    grid = np.arange(grid_min, grid_max, args_dict["PIXEL_SIZE"])

    gauss = np.exp( -0.5 * ( ((grid[:,None] - coord[0,:]) / args_dict["SIGMA"])**2) )[:,None] * \
            np.exp( -0.5 * ( ((grid[:,None] - coord[1,:]) / args_dict["SIGMA"])**2) )

    image = gauss.sum(axis=2) * norm

    return image

def load_model(fname, filter = "name CA"):

    mda_model = mda.Universe(fname)

    # Center model
    mda_model.atoms.translate(-mda_model.select_atoms('all').center_of_mass())

    # Extract coordinates
    coordinates = mda_model.select_atoms(filter).positions.T

    return coordinates

def add_noise(img, n_pixels, pixel_size, snr):

    img_noise = np.asarray(img).reshape(n_pixels, n_pixels)
    
    rad_sq = (pixel_size * (n_pixels + 1)*0.5)**2

    grid_min = -pixel_size * (n_pixels - 1)*0.5
    grid_max = pixel_size * (n_pixels - 1)*0.5 + pixel_size

    grid = np.arange(grid_min, grid_max, pixel_size)

    mask = grid[None,:]**2 + grid[:,None]**2 < rad_sq

    noise_std = np.std(img[mask]) / snr
    noise = np.random.normal(loc=0.0, scale = noise_std, size=img.shape)

    img_noise = img + noise

    return img_noise

def simulator(index):

    index1 = int(np.round(index))

    coord = load_model(f"models/state_1_{index1}.pdb")

    quat = gen_quat(1)[0]
    rot_mat = Rotation.from_quat(quat).as_matrix()
    coord = np.matmul(rot_mat, coord)

    image = gen_img(coord, args_dict)
    image = add_noise(image, args_dict["N_PIXELS"], args_dict["PIXEL_SIZE"], 0.1)

    return image


args_dict = {"PIXEL_SIZE": 4,
            "N_PIXELS": 32,
            "SIGMA": 1.0
            }


def main():
    
    prior_indices = utils.BoxUniform(low=1*torch.ones(1), high=20*torch.ones(1))

    simulator_sbi, prior_sbi = prepare_for_sbi(simulator, prior_indices)

    indices, images = simulate_for_sbi(
        simulator_sbi,
        proposal=prior_sbi,
        num_simulations=100000,
        num_workers=32
    )

    torch.save(images, "images_rot_noise_01.pt")
    torch.save(indices, "indices_rot_noise_01.pt")

    density_estimator_build_fun = posterior_nn(model='maf', hidden_features=100, num_transforms=8)
    inference = SNPE(prior=prior_indices, density_estimator=density_estimator_build_fun)

    inference = inference.append_simulations(indices, images)

    density_estimator = inference.train()
    posterior = inference.build_posterior(density_estimator)

    with open("posteriors_rot.pkl", "wb") as handle:
        pickle.dump(posterior, handle)

if __name__ == "__main__":

    main()