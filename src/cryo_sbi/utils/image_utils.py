import torch
import torchvision.transforms as transforms
import torch.distributions as d
import mrcfile


def circular_mask(n_pixels, radius, inside=True):
    """Create a circular mask for a given image size and radius.

    Args:
        n_pixels (int): Side length of the image in pixels.
        radius (int): Radius of the circle.
        inside (bool, optional): If True, the mask will be True inside the circle. Defaults to True.

    Returns:
        mask (torch.Tensor): Mask of shape (n_pixels, n_pixels).
    """

    grid = torch.linspace(-0.5 * (n_pixels - 1), 0.5 * (n_pixels - 1), n_pixels)
    r_2d = grid[None, :] ** 2 + grid[:, None] ** 2

    if inside is True:
        mask = r_2d < radius**2
    else:
        mask = r_2d > radius**2

    return mask


class Mask:
    """Mask a circular region in an image.

    Args:
        image_size (int): Number of pixels in the image.
        radius (int): Radius of the circle.
        inside (bool, optional): If True, the mask will be True inside the circle. Defaults to True.

    Returns:
        image (torch.Tensor): Masked image.
    """

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
    """Downsample an image by removing the outer frequencies.

    Args:
        image (torch.Tensor): Image of shape (n_pixels, n_pixels) or (n_channels, n_pixels, n_pixels).
        image_size (int): Side length of the image in pixels.
        n_pixels (int): Number of pixels to remove from each side.

    Returns:
        reconstructed (torch.Tensor): Downsampled image.
    """

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
    """Downsample an image by removing the outer frequencies.

    Args:
        image_size (int): Side length of the image in pixels.
        down_sampled_size (int): Number of pixels in the downsampled image.

    Returns:
        down_sampled (torch.Tensor): Downsampled image.
    """

    def __init__(self, image_size, down_sampled_size):
        self._image_size = image_size
        self._n_pixels = (image_size - down_sampled_size) // 2

    def __call__(self, image):
        down_sampled = fourier_down_sample(
            image, image_size=self._image_size, n_pixels=self._n_pixels
        )

        return down_sampled


class LowPassFilter:
    """Low pass filter an image by removing the outer frequencies.

    Args:
        image_size (int): Side length of the image in pixels.
        frequency_cutoff (int): Frequency cutoff.

    Returns:
        reconstructed (torch.Tensor): Low pass filtered image.
    """

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
    """Normalize an image by subtracting the mean and dividing by the standard deviation.

    Args:
        images (torch.Tensor): Image of shape (n_channels, n_pixels, n_pixels).

    Returns:
        normalized (torch.Tensor): Normalized image.
    """

    def __init__(self) -> None:
        pass

    def __call__(self, images):
        mean = images.mean(dim=[1, 2])
        std = images.std(dim=[1, 2])
        return transforms.functional.normalize(images, mean=mean, std=std)


class MRCtoTensor:
    """Convert an MRC file to a tensor.

    Args:
        image_path (str): Path to the MRC file.

    Returns:
        image (torch.Tensor): Image of shape (n_pixels, n_pixels).
    """

    def __init__(self) -> None:
        pass

    def __call__(self, image_path):
        assert isinstance(image_path, str), "image path needs to be a string"
        with mrcfile.open(image_path) as mrc:
            image = mrc.data
        return torch.from_numpy(image)


class AddLowFrequencyNoise:
    """Adding low frequency noise, also serving as a low pass filter.

    Args:
        image_size (int): Side length of the image in pixels.
        min_frequency (int): Minimum frequency to keep.
        amplitude (float or list): Amplitude of the noise. If a float, the noise will have the same amplitude everywhere. If a list, the noise will have a uniform distribution between the two values.
        device (str, optional): Device to use. Defaults to "cpu".

    Returns:
        image (torch.Tensor): Image with added noise.
    """

    def __init__(self, image_size, min_frequency, amplitude, device="cpu") -> None:
        self.device = device
        self._imge_size = image_size
        self._mask = circular_mask(image_size, min_frequency, inside=False).to(
            self.device
        )
        self._num_frequencies = self._mask.sum()

        if isinstance(amplitude, (float, int)):
            self._amplitude = d.uniform.Uniform(
                low=torch.tensor(float(amplitude), device=device),
                high=torch.tensor(float(amplitude), device=device),
            )
        elif isinstance(amplitude, (list, tuple)) and len(amplitude) == 2:
            self._amplitude = d.uniform.Uniform(
                low=torch.tensor(float(amplitude[0]), device=device),
                high=torch.tensor(float(amplitude[1]), device=device),
            )

    def __call__(self, image):
        fft_image = torch.fft.fft2(image)

        if len(image.shape) == 2:
            fft_image[self._mask] += self._amplitude.sample() * torch.randn(
                (self._num_frequencies,), dtype=torch.complex64, device=self.device
            )
            fft_image[self._mask == False] = 0 + 0j
        elif len(image.shape) == 3:
            fft_image[:, self._mask] += self._amplitude.sample(
                (image.shape[0], 1)
            ) * torch.randn(
                (image.shape[0], self._num_frequencies),
                dtype=torch.complex64,
                device=self.device,
            )
            fft_image[:, self._mask == False] = 0 + 0j
        else:
            raise NotImplementedError

        return torch.fft.ifft2(fft_image).real
