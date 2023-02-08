import numpy as np
import torch


def apply_ctf(image, image_params, preproc_params):
    def calc_ctf(n_pixels, amp, phase, b_factor):
        ctf = torch.zeros(n_pixels, n_pixels, dtype=torch.complex64)

        freq_pix_1d = torch.fft.fftfreq(n_pixels, d=image_params["PIXEL_SIZE"])

        x, y = torch.meshgrid(freq_pix_1d, freq_pix_1d)

        freq2_2d = x**2 + y**2
        imag = torch.zeros_like(freq2_2d) * 1j

        env = torch.exp(torch.tensor(-b_factor * freq2_2d * 0.5))
        ctf = (
            amp * torch.cos(torch.tensor(phase * freq2_2d * 0.5))
            - torch.sqrt(torch.tensor(1 - amp**2))
            * torch.sin(torch.tensor(phase * freq2_2d * 0.5))
            + imag
        )

        return ctf * env / amp

    b_factor = 0.0  # no
    amp = 0.1  # no

    elec_wav_param1 = float(12.264259661581491)
    elec_wav_param2 = float(0.9784755917869367)

    elecwavel = elec_wav_param1 / np.sqrt(
        preproc_params["VOLTAGE"] * 1e3
        + elec_wav_param2 * preproc_params["VOLTAGE"] ** 2
    )

    if len(preproc_params["DEFOCUS"]) == 1:
        phase = preproc_params["DEFOCUS"] * np.pi * 2.0 * 10000 * elecwavel

    else:
        defocus = (
            np.random.rand()
            * (preproc_params["DEFOCUS"][1] - preproc_params["DEFOCUS"][0])
            + preproc_params["DEFOCUS"][0]
        )

        phase = defocus * np.pi * 2.0 * 10000 * elecwavel

    ctf = calc_ctf(image.shape[0], amp, phase, b_factor)

    conv_image_ctf = torch.fft.fft2(image) * ctf

    image_ctf = torch.fft.ifft2(conv_image_ctf).real

    return image_ctf


def apply_ctf_to_dataset(dataset, image_params, preproc_params):
    ctf_images = torch.empty_like(dataset, device=preproc_params["DEVICE"])
    n_pixels = int(np.sqrt(dataset.shape[1]))

    for i in range(dataset.shape[0]):
        tmp_image = apply_ctf(
            dataset[i].reshape(n_pixels, n_pixels),
            image_params,
            preproc_params,
        )

        ctf_images[i] = tmp_image.reshape(1, -1).to(preproc_params["DEVICE"])

    return ctf_images
