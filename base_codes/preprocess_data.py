import argparse
import json
import numpy as np
from sklearn import preprocessing
import torch


#### Preprocessing functions for single images ####
def pad_image(image):

    pad_width = int(np.ceil(image_params["N_PIXELS"] * 0.1)) + 1
    padded_image = np.pad(image, pad_width=pad_width)

    return padded_image


def apply_ctf(image):
    def calc_ctf(n_pixels, amp, phase, b_factor):

        ctf = np.zeros((n_pixels, n_pixels), dtype=np.complex128)

        freq_pix_1d = np.fft.fftfreq(n_pixels, d=image_params["PIXEL_SIZE"])

        x, y = np.meshgrid(freq_pix_1d, freq_pix_1d)

        freq2_2d = x**2 + y**2
        imag = np.zeros_like(freq2_2d) * 1j

        env = np.exp(-b_factor * freq2_2d * 0.5)
        ctf = (
            amp * np.cos(phase * freq2_2d * 0.5)
            - np.sqrt(1 - amp**2) * np.sin(phase * freq2_2d * 0.5)
            + imag
        )

        return ctf * env / amp

    b_factor = 0.0  # no
    amp = 0.1  # no

    elecwavel = 0.019866
    phase = preproc_params["DEFOCUS"] * np.pi * 2.0 * 10000 * elecwavel

    ctf = calc_ctf(image.shape[0], amp, phase, b_factor)

    conv_image_ctf = np.fft.fft2(image) * ctf

    image_ctf = np.fft.ifft2(conv_image_ctf).real

    return image_ctf


def apply_random_shift(padded_image):

    shift_x = int(np.ceil(image_params["N_PIXELS"] * 0.1 * (2 * np.random.rand() - 1)))
    shift_y = int(np.ceil(image_params["N_PIXELS"] * 0.1 * (2 * np.random.rand() - 1)))

    pad_width = int(np.ceil(image_params["N_PIXELS"] * 0.1)) + 1

    low_ind_x = pad_width - shift_x
    high_ind_x = padded_image.shape[0] - pad_width - shift_x

    low_ind_y = pad_width - shift_y
    high_ind_y = padded_image.shape[0] - pad_width - shift_y

    shifted_image = padded_image[low_ind_x:high_ind_x, low_ind_y:high_ind_y]

    return shifted_image


def add_noise(img):

    # mean_image = np.mean(img)
    std_image = np.std(img)

    mask = np.abs(img) > 0.5 * std_image

    signal_mean = np.mean(img[mask])
    signal_std = np.std(img[mask])

    noise_std = signal_std / np.sqrt(preproc_params["SNR"])
    noise = np.random.normal(loc=signal_mean, scale=noise_std, size=img.shape)

    img_noise = img + noise

    return img_noise


def gaussian_normalize_image(image):

    mean_img = np.mean(image)
    std_img = np.std(image)

    return (image - mean_img) / std_img


### Preprocessing functions for datasets ###
def pad_dataset(dataset):

    padded_images = torch.empty_like(dataset, device=training_params["DEVICE"])
    n_pixels = int(np.sqrt(dataset.shape[1]))

    for i in range(dataset.shape[0]):

        tmp_image = pad_image(dataset[i].reshape(n_pixels, n_pixels).cpu().numpy())

        padded_images[i] = torch.tensor(
            tmp_image.reshape(1, -1), device=training_params["DEVICE"]
        )

    return padded_images


def apply_ctf_to_dataset(dataset):

    ctf_images = torch.empty_like(dataset, device=training_params["DEVICE"])
    n_pixels = int(np.sqrt(dataset.shape[1]))

    for i in range(dataset.shape[0]):

        tmp_image = apply_ctf(dataset[i].reshape(n_pixels, n_pixels).cpu().numpy())

        ctf_images[i] = torch.tensor(
            tmp_image.reshape(1, -1), device=training_params["DEVICE"]
        )

    return ctf_images


def shift_dataset(dataset):

    shifted_images = torch.empty_like(dataset, device=training_params["DEVICE"])
    n_pixels = int(np.sqrt(dataset.shape[1]))

    for i in range(dataset.shape[0]):

        tmp_image = apply_random_shift(
            dataset[i].reshape(n_pixels, n_pixels).cpu().numpy()
        )

        shifted_images[i] = torch.tensor(
            tmp_image.reshape(1, -1), device=training_params["DEVICE"]
        )

    return shifted_images


def add_noise_to_dataset(dataset):

    images_with_noise = torch.empty_like(dataset, device=training_params["DEVICE"])
    n_pixels = int(np.sqrt(dataset.shape[1]))

    for i in range(dataset.shape[0]):

        tmp_image = add_noise(dataset[i].reshape(n_pixels, n_pixels).cpu().numpy())

        images_with_noise[i] = torch.tensor(
            tmp_image.reshape(1, -1), device=training_params["DEVICE"]
        )

    return images_with_noise


def normalize_dataset(dataset):

    norm_images = torch.empty_like(dataset, device=training_params["DEVICE"])
    n_pixels = int(np.sqrt(dataset.shape[1]))

    for i in range(dataset.shape[0]):

        tmp_image = gaussian_normalize_image(
            dataset[i].reshape(n_pixels, n_pixels).cpu().numpy()
        )

        norm_images[i] = torch.tensor(
            tmp_image.reshape(1, -1), device=training_params["DEVICE"]
        )

    return norm_images


def main():

    indices = torch.load("indices.pt").to(training_params["DEVICE"])
    images = torch.load("images.pt").to(training_params["DEVICE"])

    images = pad_dataset(images)
    images = apply_ctf_to_dataset(images)
    images = shift_dataset(images)
    images = add_noise_to_dataset(images)
    images = normalize_dataset(images)

    torch.save(indices, "indices_training.pt")
    torch.save(images, "images_training.pt")

    return


def check_inputs():

    for section in ["IMAGES", "SIMULATION", "PREPROCESSING", "TRAINING"]:
        assert (
            section in config.keys()
        ), f"Please provide section {section} in config.ini"

    for key in ["N_PIXELS", "PIXEL_SIZE", "SIGMA"]:
        assert key in image_params.keys(), f"Please provide a value for {key}"

    for key in ["N_SIMULATIONS", "MODEL_FILE", "DEVICE"]:
        assert key in simulation_params.keys(), f"Please provide a value for {key}"

    for key in ["SNR", "DEFOCUS"]:
        assert key in preproc_params.keys(), f"Please provide a value for {key}"

    for key in ["DEVICE"]:
        assert (
            key in training_params.keys()
        ), f"Please include the DEVICE you will use for training"

    if "cuda" in training_params["DEVICE"]:
        assert (
            torch.cuda.is_available()
        ), "Your device is cuda but there is no GPU available"

    return


if __name__ == "__main__":

    global config, image_params, simulation_params, preproc_params, training_params, models

    parser = argparse.ArgumentParser(
        description="Input file",
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

    image_params = dict(config["IMAGES"])
    simulation_params = config["SIMULATION"]
    preproc_params = config["PREPROCESSING"]

    check_inputs()

    main()
