import numpy as np
import pickle
import os

# SBI
import torch
from sbi import utils as utils
from sbi.inference import SNPE
from sbi.utils.get_nn_models import posterior_nn



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

def add_noise_to_dataset(dataset, args_dict, snr):

    images_with_noise = torch.empty_like(dataset)

    for i in range(dataset.shape[0]):

        image_with_noise = add_noise(
            dataset[i].reshape(args_dict["N_PIXELS"], args_dict["N_PIXELS"]).numpy(),
            args_dict["N_PIXELS"],
            args_dict["PIXEL_SIZE"],
            snr)

        images_with_noise[i] = torch.tensor(image_with_noise.reshape(args_dict["N_PIXELS"]**2))

    return images_with_noise

def normalize_dataset(dataset):

    norm_images = torch.empty_like(dataset)

    for i in range(dataset.shape[0]):

        mu = torch.mean(dataset[i])
        sigma = torch.std(dataset[i])

        norm_images[i] = (dataset[i] - mu) / sigma

    return norm_images

def main():

    args_dict = {"PIXEL_SIZE": 4,
             "N_PIXELS": 32,
             "SIGMA": 1.0
             }

    images = torch.load("images_no_rot_clean.pt")
    indices = torch.load("indices_no_rot_clean.pt")

    prior_indices = utils.BoxUniform(low=1*torch.ones(1), high=20*torch.ones(1))

    density_estimator_build_fun = posterior_nn(model='maf', hidden_features=10, num_transforms=4)
    inference = SNPE(prior=prior_indices, density_estimator=density_estimator_build_fun)

    # Calculate and save posterior
    posteriors = {}

    # Train multiple posteriors for different noise levels
    snr_training = [0.1]

    for snr in snr_training:
        
        ##### Post processing images #####
        images_with_noise = add_noise_to_dataset(images, args_dict, snr) 
        images_with_noise = normalize_dataset(images_with_noise)
        ##### Post processing images #####
        
        inference = inference.append_simulations(indices, images_with_noise)

        density_estimator = inference.train()
        posteriors[f"snr_{snr}"] = inference.build_posterior(density_estimator)

    with open("posteriors_no_rot.pkl", "wb") as handle:
        pickle.dump(posteriors, handle)


if __name__ == "__main__":

    main()
