import random
from typing import Optional, Union, List
from cryo_sbi.utils.image_utils import mrc_to_tensor
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


class RandomMicrographPatches:
    """
    Iterator that returns random patches from a list of micrographs.

    Args:
        micro_graphs (List[Union[str, torch.Tensor]]): List of micrographs.
        transform (Union[None, transforms.Compose]): Transform to apply to the patches.
        patch_size (int): Size of the patches.
        batch_size (int, optional): Batch size. Defaults to 1.
    """

    def __init__(
        self,
        micro_graphs: List[Union[str, torch.Tensor]],
        transform: Union[None, transforms.Compose],
        patch_size: int,
        max_iter: Optional[int] = 1000,
    ) -> None:
        if all(map(isinstance, micro_graphs, [str] * len(micro_graphs))):
            self._micro_graphs = [mrc_to_tensor(path) for path in micro_graphs]
        else:
            self._micro_graphs = micro_graphs

        self._transform = transform
        self._patch_size = patch_size
        self._max_iter = max_iter
        self._current_iter = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._current_iter == self._max_iter:
            self._current_iter = 0
            raise StopIteration
        random_micrograph = random.choice(self._micro_graphs)
        assert random_micrograph.ndim == 2, "Micrograph should be 2D"
        x = random.randint(
            self._patch_size, random_micrograph.shape[0] - self._patch_size
        )
        y = random.randint(
            self._patch_size, random_micrograph.shape[1] - self._patch_size
        )
        patch = TF.crop(
            random_micrograph,
            top=y,
            left=x,
            height=self._patch_size,
            width=self._patch_size,
        )
        if self._transform is not None:
            patch = self._transform(patch)
        self._current_iter += 1
        return patch

    def __len__(self):
        return self._max_iter

    @property
    def shape(self):
        """
        Shape of the transformed patches.

        Returns:
            torch.Size: Shape of the transformed patches.
        """
        return self.__next__().shape


def compute_average_psd(images: Union[torch.Tensor, RandomMicrographPatches]):
    """
    Compute the average PSD of a set of images.

    Args:
        images (Union[torch.Tensor, RandomMicrographPatches]): Images to compute the average PSD of.

    Returns:
        avg_psd (torch.Tensor): Average PSD of the images.
    """

    if isinstance(images, RandomMicrographPatches):
        avg_psd = torch.zeros(images.shape[1:])
        for image in images:
            fft_image = torch.fft.fft2(image[0])
            psd = torch.abs(fft_image) ** 2
            avg_psd += psd / len(images)
    elif isinstance(images, torch.Tensor):
        fft_images = torch.fft.fft2(images, dim=(-2, -1))
        avg_psd = torch.mean(torch.abs(fft_images) ** 2, dim=0)
    return avg_psd
