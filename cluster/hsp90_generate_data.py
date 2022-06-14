# Numerical libraries
import numpy as np
from scipy.spatial.transform import Rotation
import sys
# MD Stuff
import MDAnalysis as mda

# SBI
import torch
from sbi import utils as utils
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi


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

def add_noise(img, snr):

    #mean_image = np.mean(img)
    std_image = np.std(img)

    mask = np.abs(img) > 0.5 * std_image

    signal_mean = np.mean(img[mask])
    signal_std = np.std(img[mask])

    noise_std = signal_std / np.sqrt(snr)
    noise = np.random.normal(loc=signal_mean, scale=noise_std, size=img.shape)

    img_noise = img + noise

    img_noise -= np.mean(img_noise)
    img_noise /= np.std(img_noise)

    # img_noise = np.zeros_like(img)
    # img_noise[mask] = img[mask]
    
    return img_noise

def load_model(fname, filter = "name CA"):

    mda_model = mda.Universe(fname)

    # Center model
    mda_model.atoms.translate(-mda_model.select_atoms('all').center_of_mass())

    # Extract coordinates
    coordinates = mda_model.select_atoms(filter).positions.T

    return coordinates

prior_indices = utils.BoxUniform(low=1*torch.ones(1), high=20*torch.ones(1))

args_dict = {"PIXEL_SIZE": 1,
             "N_PIXELS": 128,
             "SIGMA": 1.0
             }

def simulator(index):

    index1 = int(np.round(index))

    coord = load_model(f"../models/state_1_{index1}.pdb")

    quat = gen_quat(1)[0]
    rot_mat = Rotation.from_quat(quat).as_matrix()
    coord = np.matmul(rot_mat, coord)

    image = gen_img(coord, args_dict)
    image = add_noise(image, 0.1)

    return image

simulator_sbi, prior_sbi = prepare_for_sbi(simulator, prior_indices)

def main(argv):


    prior_indices = utils.BoxUniform(low=1*torch.ones(1), high=20*torch.ones(1))

    simulator_sbi, prior_sbi = prepare_for_sbi(simulator, prior_indices)

    indices, images = simulate_for_sbi(
        simulator_sbi,
        proposal=prior_sbi,
        num_simulations=1000000,
        num_workers=int(argv[1])
    )

    torch.save(indices, "indices.pt")
    torch.save(images, "images.pt")


if __name__ == "__main__":

    main(sys.argv)