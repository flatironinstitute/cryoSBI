import logging
from typing import List, Union
from functools import lru_cache
import numpy as np
import torch
import torchvision.transforms as transforms
import mrcfile
from tqdm import tqdm

_log = logging.getLogger(__name__)


def circular_mask(
    n_pixels: int, radius: int, inside: bool = True, device="cpu"
) -> torch.Tensor:
    """
    Create a circular mask for a given image size and radius.

    Args:
        n_pixels (int): Side length of the image in pixels.
        radius (int): Radius of the circle.
        inside (bool, optional): If True, the mask will be True inside the circle. Defaults to True.
        device (str, optional): Device on which to allocate the mask. Defaults to "cpu".

    Returns:
        mask (torch.Tensor): Mask of shape (n_pixels, n_pixels).
    """

    grid = torch.linspace(
        -0.5 * (n_pixels - 1), 0.5 * (n_pixels - 1), n_pixels, device=device
    )
    r_2d = grid[None, :] ** 2 + grid[:, None] ** 2

    if inside is True:
        mask = r_2d <= radius**2
    else:
        mask = r_2d > radius**2

    return mask


class Mask:
    """
    Mask a circular region in an image.

    Args:
        image_size (int): Number of pixels in the image.
        radius (int): Radius of the circle.
        inside (bool, optional): If True, the mask will be True inside the circle. Defaults to False.
    """

    def __init__(self, image_size: int, radius: int, inside: bool = False) -> None:
        self.image_size = image_size
        self.radius = radius
        self.mask = circular_mask(image_size, radius, inside=inside)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Mask a circular region in an image.

        Args:
            image (torch.Tensor): Image of shape (n_pixels, n_pixels) or (n_channels, n_pixels, n_pixels).

        Returns:
            image (torch.Tensor): Image with masked region equal to zero.
        """

        image = image.clone()
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

    fft_image = torch.fft.ifftshift(fft_image)
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

        fft_image = torch.fft.ifftshift(fft_image)
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

        fft_image = torch.fft.ifftshift(fft_image)
        reconstructed = torch.fft.ifft2(fft_image).real
        return reconstructed


class Identity:
    """
    Identity transform that returns the input image unchanged.
    """

    def __init__(self) -> None:
        pass

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """
        Returns the input image unchanged.

        Args:
            images (torch.Tensor): Image of shape (n_channels, n_pixels, n_pixels).

        Returns:
            images (torch.Tensor): Unchanged image.
        """
        return images


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


def mrc_to_tensor(image_path: str, copy: bool = True) -> torch.Tensor:
    """
    Convert an MRC file to a tensor.

    Args:
        image_path (str): Path to the MRC file.
        copy (bool, optional): Retained for API compatibility; the underlying
            memory-mapped data is always copied because the mmap is closed
            when this function returns and a non-copied tensor would point at
            unmapped memory.

    Returns:
        image (torch.Tensor): Image of shape (n_pixels, n_pixels).
    """

    assert isinstance(image_path, str), "image path needs to be a string"

    with mrcfile.mmap(image_path, permissive=True) as mrc:
        data = np.copy(mrc.data)

    return torch.from_numpy(data)


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


def estimate_noise_psd(
    images: torch.Tensor, image_size: int, mask_radius: Union[int, None] = None
) -> torch.Tensor:
    """
    Estimates the power spectral density (PSD) of the noise in a set of images.

    Args:
        images (torch.Tensor): A tensor containing the input images. The shape of the tensor should be (N, H, W),
                               where N is the number of images, H is the height, and W is the width.
        image_size (int): Side length of the images in pixels.
        mask_radius (int, optional): Radius of the circular mask used to isolate the noise region. Defaults to image_size // 2 when None.

    Returns:
        torch.Tensor: A tensor containing the estimated PSD of the noise. The shape of the tensor is (H, W), where H is the height
                      and W is the width of the images.

    """
    if mask_radius is None:
        mask_radius = image_size // 2
    mask = circular_mask(image_size, mask_radius, inside=False, device=images.device)
    denominator = mask.sum() * images.shape[0]
    images_masked = images * mask
    mean_est = images_masked.sum() / denominator
    image_masked_fft = torch.fft.fft2(images_masked)
    noise_psd_est = torch.sum(torch.abs(image_masked_fft) ** 2, dim=[0]) / denominator
    # DC component of an unshifted FFT lives at [0, 0], not [N//2, N//2].
    noise_psd_est[0, 0] -= mean_est

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
            images (torch.Tensor): Image of shape (n_pixels, n_pixels) or
                (num_images, n_pixels, n_pixels).

        Returns:
            torch.Tensor: Whitened image, same shape as input.
        """
        squeeze_back = False
        if images.ndim == 2:
            images = images.unsqueeze(0)
            squeeze_back = True
        elif images.ndim != 3:
            raise ValueError(
                f"Expected 2D (n_pixels, n_pixels) or 3D (B, n_pixels, n_pixels) "
                f"image; got shape {tuple(images.shape)}"
            )
        psd = self._estimate_noise_psd(images)
        # Clamp before sqrt to avoid div-by-zero where the masked-region PSD has
        # exact zeros (constant inputs, fully masked regions, etc.).
        eps = torch.finfo(psd.dtype).eps
        noise_psd = psd.clamp_min(eps) ** -0.5
        images_fft = torch.fft.fft2(images)
        images_fft = images_fft * noise_psd
        images = torch.fft.ifft2(images_fft).real
        return images.squeeze(0) if squeeze_back else images


class GaussianSpatialMask:
    """
    Applies a soft 2D Gaussian mask in the image domain to suppress edges and emphasize the center.
    """

    def __init__(self, image_size: int, sigma: float):
        self._image_size = image_size
        self._sigma = sigma

        grid = torch.linspace(
            -0.5 * (image_size - 1), 0.5 * (image_size - 1), image_size
        )
        r_2d = grid[None, :] ** 2 + grid[:, None] ** 2
        self._mask = torch.exp(-r_2d / (2 * sigma**2))

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Applies the Gaussian spatial mask to an image.

        Args:
            image (torch.Tensor): Image of shape (n_pixels, n_pixels) or (n_channels, n_pixels, n_pixels).

        Returns:
            torch.Tensor: Masked image.
        """
        if len(image.shape) == 2:
            return image * self._mask
        elif len(image.shape) == 3:
            return image * self._mask.unsqueeze(0)
        else:
            raise NotImplementedError("Image must be 2D or 3D tensor")


class MRCdataset:
    """
    Creates a dataset of MRC files.
    Each MRC file is converted to a tensor and has a unique index.

    Args:
        image_paths (list[str]): List of paths to MRC files.
        cache_size (int, optional): LRU cache size for the MRC-to-tensor loader. Set to 0 to disable caching. Defaults to 16.

    Methods:
        build_index_map: Builds a map of indices to file paths and file indices.
        getitem: Returns a at the given global index.
        __getitem__: Returns tensor of the MRC file at the given index.
    """

    def __init__(self, image_paths: List[str], cache_size: int = 16):
        super().__init__()
        self.paths = image_paths
        self._num_paths = len(image_paths)
        self._index_map = None
        self._mrc_to_tensor_cached = (
            lru_cache(maxsize=cache_size)(mrc_to_tensor)
            if cache_size > 0
            else mrc_to_tensor
        )

    def __len__(self):
        return self._num_paths

    def __getitem__(self, idx):
        """Returns tensor of the MRC file at the given index."""
        return idx, self._mrc_to_tensor_cached(self.paths[idx], copy=True)

    @staticmethod
    def _extract_num_particles(path):
        with mrcfile.open(path, permissive=True, header_only=True) as mrc:
            num_images = int(mrc.header.nz)
            return num_images if num_images > 0 else 1

    def build_index_map(self, method: str = "mrc"):
        """
        Builds a map of image indices to file paths and file indices.
        """
        if self._index_map is not None:
            _log.info("Index map already built.")
            return

        if method == "mrc":
            self._build_index_map_by_loading_mrc()
        elif method == "star":
            raise NotImplementedError("STAR file parsing not implemented yet.")
        else:
            raise ValueError("Method must be 'mrc' or 'star'.")

    def _build_index_map_by_loading_mrc(self):
        self._path_index = []
        self._file_index = []
        _log.info("Initializing indexing...")
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
        assert len(self.paths) == len(
            index_map["paths"]
        ), "Number of paths do not match the index map."
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
            image = self._mrc_to_tensor_cached(self.paths[self._path_index[idx]])
            if image.ndim > 2:
                return image[self._file_index[idx]]
            return image
        if isinstance(idx, (list, np.ndarray, torch.Tensor)):
            return [
                self._mrc_to_tensor_cached(self.paths[self._path_index[i]])[
                    self._file_index[i]
                ]
                for i in idx
            ]
        raise TypeError(f"idx must be int, list, np.ndarray, or torch.Tensor; got {type(idx)}")

    def get_path(self, idx: Union[int, list]) -> Union[str, List[str]]:
        """
        Returns the path of the image at the given global index.

        Args:
            idx (int, List): Global index of the image.
        """
        assert (
            self._index_map is not None
        ), "Index map not built. First call build_index_map() or load_index_map()"
        if isinstance(idx, int):
            return self.paths[self._path_index[idx]], self._file_index[idx]
        if isinstance(idx, (list, np.ndarray, torch.Tensor)):
            paths = [self.paths[self._path_index[i]] for i in idx]
            file_indices = [self._file_index[i] for i in idx]
            return paths, file_indices
        raise TypeError(f"idx must be int, list, np.ndarray, or torch.Tensor; got {type(idx)}")


class MRCloader(torch.utils.data.DataLoader):
    """
    Creates a dataloader of MRC files.

    Args:
        image_paths (list[str]): List of paths to MRC files.
        **kwargs: Keyword arguments passed to torch.utils.data.DataLoader.
    """

    def __init__(self, image_paths: List[str], **kwargs):
        super().__init__(
            MRCdataset(image_paths, cache_size=0), batch_size=None, **kwargs
        )
