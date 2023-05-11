import torch
import torchvision.transforms as transforms
import torch.distributions as d
import mrcfile


def circular_mask(n_pixels: int, radius: int, inside: bool = True) -> torch.Tensor:
    """
    Create a circular mask for a given image size and radius.

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
    """
    Mask a circular region in an image.

    Args:
        image_size (int): Number of pixels in the image.
        radius (int): Radius of the circle.
        inside (bool, optional): If True, the mask will be True inside the circle. Defaults to True.
    """

    def __init__(self, image_size: int, radius: int, inside: bool = False) -> None:
        self.image_size = image_size
        self.n_pixels = radius
        self.mask = circular_mask(image_size, radius, inside=inside)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Mask a circular region in an image.

        Args:
            image (torch.Tensor): Image of shape (n_pixels, n_pixels) or (n_channels, n_pixels, n_pixels).

        Returns:
            image (torch.Tensor): Image with masked region equal to zero.
        """

        if len(image.shape) == 2:
            image[self.mask] = 0
        elif len(image.shape) == 3:
            image[:, self.mask] = 0
        else:
            raise NotImplementedError

        return image


def fourier_down_sample(
    image: torch.Tensor, image_size: int, n_pixels: int
) -> torch.Tensor:
    """
    Downsample an image by removing the outer frequencies.

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
    """
    Downsample an image by removing the outer frequencies.

    Args:
        image_size (int): Size of image in pixels.
        down_sampled_size (int): Size of downsampled image in pixels.

    Returns:
        down_sampled (torch.Tensor): Downsampled image.
    """

    def __init__(self, image_size: int, down_sampled_size: int) -> None:
        self._image_size = image_size
        self._n_pixels = (image_size - down_sampled_size) // 2

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Downsample an image by removing the outer frequencies.

        Args:
            image (torch.Tensor): Image of shape (n_pixels, n_pixels) or (n_channels, n_pixels, n_pixels).

        Returns:
            down_sampled (torch.Tensor): Downsampled image.
        """

        down_sampled = fourier_down_sample(
            image, image_size=self._image_size, n_pixels=self._n_pixels
        )

        return down_sampled


class LowPassFilter:
    """
    Low pass filter an image by removing the outer frequencies.

    Args:
        image_size (int): Side length of the image in pixels.
        frequency_cutoff (int): Frequency cutoff.

    Returns:
        reconstructed (torch.Tensor): Low pass filtered image.
    """

    def __init__(self, image_size: int, frequency_cutoff: int) -> None:
        self.mask = circular_mask(image_size, frequency_cutoff)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Low pass filter an image by removing the outer frequencies.

        Args:
            image (torch.Tensor): Image of shape (n_pixels, n_pixels) or (n_channels, n_pixels, n_pixels).

        Returns:
            reconstructed (torch.Tensor): Low pass filtered image.
        """
        fft_image = torch.fft.fft2(image)
        fft_image = torch.fft.fftshift(fft_image)

        if len(image.shape) == 2:
            fft_image[self.mask] = 0 + 0j
        elif len(image.shape) == 3:
            fft_image[:, self.mask] = 0 + 0j
        else:
            raise NotImplementedError

        fft_image = torch.fft.fftshift(fft_image)
        reconstructed = torch.fft.ifft2(fft_image).real
        return reconstructed


class NormalizeIndividual:
    """
    Normalize an image by subtracting the mean and dividing by the standard deviation.

    Args:
        images (torch.Tensor): Image of shape (n_channels, n_pixels, n_pixels).

    Returns:
        normalized (torch.Tensor): Normalized image.
    """

    def __init__(self) -> None:
        pass

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """
        Normalize an image by subtracting the mean and dividing by the standard deviation.

        Args:
            images (torch.Tensor): Image of shape (n_channels, n_pixels, n_pixels).

        Returns:
            normalized (torch.Tensor): Normalized image.
        """
        mean = images.mean(dim=[1, 2])
        std = images.std(dim=[1, 2])
        return transforms.functional.normalize(images, mean=mean, std=std)


class MRCtoTensor:
    """
    Convert an MRC file to a tensor.
    """

    def __init__(self) -> None:
        pass

    def __call__(self, image_path: str) -> torch.Tensor:
        """
        Convert an MRC file to a tensor.

        Args:
            image_path (str): Path to the MRC file.

        Returns:
            image (torch.Tensor): Image of shape (n_pixels, n_pixels).
        """

        assert isinstance(image_path, str), "image_path should be a string"
        with mrcfile.open(image_path) as mrc:
            image = mrc.data
        return torch.from_numpy(image)
