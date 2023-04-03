import torch
import torchvision.transforms as transforms
import mrcfile


def circular_mask(n_pixels, radius, inside=True):
    grid = torch.linspace(-0.5 * (n_pixels - 1), 0.5 * (n_pixels - 1), n_pixels)
    r_2d = grid[None, :] ** 2 + grid[:, None] ** 2

    if inside is True:
        mask = r_2d < radius**2
    else:
        mask = r_2d > radius**2

    return mask


class Mask:
    def __init__(self, image_size, radius, inside=False):
        self.image_size = image_size
        self.n_pixels = radius
        self.mask = circular_mask(image_size, radius, inside=inside)

    def __call__(self, image):
        if len(image.shape) == 2:
            image[self.mask] = 0
        elif len(image.shape) == 3:
            image[:, self.mask] = 0
        else:
            raise NotImplementedError

        return image


def fourier_down_sample(image, image_size, n_pixels):
    fft_image = torch.fft.fft2(image)
    fft_image = torch.fft.fftshift(fft_image)

    if len(image.shape) == 2:
        fft_image = fft_image[
            n_pixels : image_size - n_pixels,
            n_pixels : image_size - n_pixels,
        ]
    elif len(image.shape) == 3:
        fft_image = fft_image[
            :,
            n_pixels : image_size - n_pixels,
            n_pixels : image_size - n_pixels,
        ]
    else:
        raise NotImplementedError

    fft_image = torch.fft.fftshift(fft_image)
    reconstructed = torch.fft.ifft2(fft_image).real
    return reconstructed


class FourierDownSample:
    def __init__(self, image_size, down_sampled_size):
        self._image_size = image_size
        self._n_pixels = (image_size - down_sampled_size) // 2

    def __call__(self, image):
        down_sampled = fourier_down_sample(
            image, image_size=self._image_size, n_pixels=self._n_pixels
        )

        return down_sampled


class LowPassFilter:
    def __init__(self, image_size, frequency_cutoff):
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
