import torch
import torchvision.transforms as transforms
import mrcfile
from cryo_sbi.wpa_simulator.noise import circular_mask


class LowPassFilter:
    def __init__(self, frequency_cutoff, image_size):
        self.mask = circular_mask(image_size, frequency_cutoff)

    def __call__(self, image):
        fft_image = torch.fft.fft2(image)

        if len(image.shape) == 2:
            fft_image[self.mask] = 0 + 0j
        elif len(image.shape) == 3:
            fft_image[:, self.mask] = 0 + 0j
        else:
            raise NotImplementedError

        reconstructed = torch.fft.ifft2(fft_image).real
        return reconstructed


class NormalizeIndividual:
    def __init__(self) -> None:
        pass

    def __call__(self, images):
        mean = images.mean(dim=[1, 2])
        std = images.std(dim=[1, 2])
        return transforms.functional.normalize(images, mean=mean, std=std)


class MRCtoTensor:
    def __init__(self) -> None:
        pass

    def __call__(self, image_path):
        assert isinstance(image_path, str), "image path needs to be a string"
        with mrcfile.open(image_path) as mrc:
            image = mrc.data
        return torch.from_numpy(image)
