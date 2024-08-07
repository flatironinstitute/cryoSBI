import math
from typing import List, Union
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.distributions as d
import mrcfile
from tqdm import tqdm


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
    """

    def __init__(self, image_size: int, frequency_cutoff: int):
        self.mask = circular_mask(image_size, frequency_cutoff, inside=False)

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


class GaussianLowPassFilter:
    """
    Low pass filter by dampening the outer frequencies with a Gaussian.
    """

    def __init__(self, image_size: int, sigma: int):
        self._image_size = image_size
        self._sigma = sigma
        self._grid = torch.linspace(
            -0.5 * (image_size - 1), 0.5 * (image_size - 1), image_size
        )
        self._r_2d = self._grid[None, :] ** 2 + self._grid[:, None] ** 2
        self._mask = torch.exp(-self._r_2d / (2 * sigma**2))

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Low pass filter an image by dampening the outer frequencies with a Gaussian.

        Args:
            image (torch.Tensor): Image of shape (n_pixels, n_pixels) or (n_channels, n_pixels, n_pixels).

        Returns:
            reconstructed (torch.Tensor): Low pass filtered image.
        """

        fft_image = torch.fft.fft2(image)
        fft_image = torch.fft.fftshift(fft_image)

        if len(image.shape) == 2:
            fft_image = fft_image * self._mask
        elif len(image.shape) == 3:
            fft_image = fft_image * self._mask.unsqueeze(0)
        else:
            raise NotImplementedError

        fft_image = torch.fft.fftshift(fft_image)
        reconstructed = torch.fft.ifft2(fft_image).real
        return reconstructed


class NormalizeIndividual:
    """
    Normalize an image by subtracting the mean and dividing by the standard deviation.
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
        if len(images.shape) == 2:
            mean = images.mean()
            std = images.std()
            images = images.unsqueeze(0)
        elif len(images.shape) == 3:
            mean = images.mean(dim=[1, 2])
            std = images.std(dim=[1, 2])
        else:
            raise NotImplementedError

        return transforms.functional.normalize(images, mean=mean, std=std)


def mrc_to_tensor(image_path: str) -> torch.Tensor:
    """
    Convert an MRC file to a tensor.

    Args:
        image_path (str): Path to the MRC file.

    Returns:
        image (torch.Tensor): Image of shape (n_pixels, n_pixels).
    """

    assert isinstance(image_path, str), "image path needs to be a string"
    with mrcfile.open(image_path) as mrc:
        image = mrc.data
    return torch.from_numpy(image)


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

        return mrc_to_tensor(image_path)


def estimate_noise_psd(images: torch.Tensor, image_size: int, mask_radius : Union[int, None] = None) -> torch.Tensor:
    """
    Estimates the power spectral density (PSD) of the noise in a set of images.

    Args:
        images (torch.Tensor): A tensor containing the input images. The shape of the tensor should be (N, H, W),
                               where N is the number of images, H is the height, and W is the width.

    Returns:
        torch.Tensor: A tensor containing the estimated PSD of the noise. The shape of the tensor is (H, W), where H is the height
                      and W is the width of the images.

    """
    if mask_radius is  None:
        mask_radius = image_size // 2
    mask = circular_mask(image_size, mask_radius, inside=False)
    denominator = mask.sum() * images.shape[0]
    images_masked = images * mask
    mean_est = images_masked.sum() / denominator
    image_masked_fft = torch.fft.fft2(images_masked)
    noise_psd_est = torch.sum(torch.abs(image_masked_fft)**2, dim=[0]) / denominator
    noise_psd_est[image_size // 2, image_size // 2] -= mean_est

    return noise_psd_est


class WhitenImage:
    """
    Whiten an image by dividing by the square root of the noise PSD.
    
    Args:
        image_size (int): Size of image in pixels.
        mask_radius (int, optional): Radius of the mask. Defaults to None.
    
    """
    
    def __init__(self, image_size: int, mask_radius: Union[int, None] = None) -> None:
        self.image_size = image_size
        self.mask_radius = mask_radius

    def _estimate_noise_psd(self, images: torch.Tensor) -> torch.Tensor:
        """
        Estimates the power spectral density (PSD) of the noise in a set of images.
        """
        noise_psd = estimate_noise_psd(images, self.image_size, self.mask_radius)
        return noise_psd
    
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """
        Whiten an image by dividing by the square root of the noise PSD.
        
        Args:
            image (torch.Tensor): Image of shape (n_pixels, n_pixels).
        
        Returns:
            image (torch.Tensor): Whitened image.
        """

        assert images.ndim == 3, "Image should have shape (num_images , n_pixels, n_pixels)"
        noise_psd = self._estimate_noise_psd(images) ** -0.5
        images_fft = torch.fft.fft2(images)
        images_fft = images_fft * noise_psd
        images = torch.fft.ifft2(images_fft).real
        return images


class MRCdataset:
    """
    Creates a dataset of MRC files.
    Each MRC file is converted to a tensor and has a unique index.

    Args:
        image_paths (list[str]): List of paths to MRC files.

    Methods:
        build_index_map: Builds a map of indices to file paths and file indices.
        getitem: Returns a at the given global index.
        __getitem__: Returns tensor of the MRC file at the given index.
    """

    def __init__(self, image_paths: List[str]):
        super().__init__()
        self.paths = image_paths
        self._num_paths = len(image_paths)
        self._index_map = None

    def __len__(self):
        return self._num_paths

    def __getitem__(self, idx):
        return idx, mrc_to_tensor(self.paths[idx])

    def _extract_num_particles(self, path):
        future_mrc = mrcfile.open_async(path)
        mrc = future_mrc.result()
        data_shape = mrc.data.shape
        # img_stack = mrc.is_image_stack()
        num_images = data_shape[0] if len(data_shape) > 2 else 1
        return num_images

    def build_index_map(self):
        """
        Builds a map of image indices to file paths and file indices.
        """
        if self._index_map is not None:
            print("Index map already built.")
            return

        self._path_index = []
        self._file_index = []
        print("Initalizing indexing...")
        for idx, path in tqdm(enumerate(self.paths), total=self._num_paths):
            num_images = self._extract_num_particles(path)
            self._path_index += [idx] * num_images
            self._file_index += list(range(num_images))
        self._index_map = True

    def save_index_map(self, path: str):
        """
        Saves the index map to a file.

        Args:
            path (str): Path to save the index map.
        """
        assert (
            self._index_map is not None
        ), "Index map not built. First call build_index_map()"
        np.savez(
            path,
            path_index=self._path_index,
            file_index=self._file_index,
            paths=self.paths,
        )

    def load_index_map(self, path: str):
        """
        Loads the index map from a file.

        Args:
            path (str): Path to load the index map.
        """
        index_map = np.load(path)
        assert len(self.paths) == len(index_map["paths"]), "Number of paths do not match the index map."
        for path1, path2 in zip(self.paths, index_map["paths"]):
            assert path1 == path2, "Paths do not match the index map."
        self._path_index = index_map["path_index"]
        self._file_index = index_map["file_index"]
        self._index_map = True

    def get_image(self, idx: Union[int, list]):
        """
        Returns the image at the given global index.

        Args:
            idx (int, List): Global index of the image.
        """
        assert (
            self._index_map is not None
        ), "Index map not built. First call build_index_map() or load_index_map()"
        if isinstance(idx, int):
            image = mrc_to_tensor(self.paths[self._path_index[idx]])
            if image.ndim > 2:
                return image[self._file_index[idx]]
        if isinstance(idx, (list, np.ndarray, torch.Tensor)):
            return [
                mrc_to_tensor(self.paths[self._path_index[i]])[self._file_index[i]]
                for i in idx
            ]


class MRCloader(torch.utils.data.DataLoader):
    """
    Creates a dataloader of MRC files.

    Args:
        image_paths (list[str]): List of paths to MRC files.
        **kwargs: Keyword arguments passed to torch.utils.data.DataLoader.
    """

    def __init__(self, image_paths: List[str], **kwargs):
        super().__init__(MRCdataset(image_paths), batch_size=None, **kwargs)
