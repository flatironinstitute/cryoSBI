import pytest
import torch
import torchvision.transforms as transforms
from cryo_sbi.utils.image_utils import NormalizeIndividual
from cryo_sbi.utils import micrograph_utils as mu


def test_random_micrograph_patches_fail():
    micrograph = torch.randn(2, 128, 128)
    random_patches = mu.RandomMicrographPatches(
        micro_graphs=[micrograph], patch_size=10, transform=None
    )
    with pytest.raises(AssertionError):
        patch = next(random_patches)


@pytest.mark.parametrize(
    ("micrograph_size", "patch_size", "max_iter"), [(128, 12, 100), (128, 90, 1000)]
)
def test_random_micrograph_patches(micrograph_size, patch_size, max_iter):
    micrograph = torch.randn(micrograph_size, micrograph_size)
    random_patches = mu.RandomMicrographPatches(
        micro_graphs=[micrograph],
        patch_size=patch_size,
        transform=None,
        max_iter=max_iter,
    )
    patch = next(random_patches)
    assert patch.shape == torch.Size([1, patch_size, patch_size])
    assert len(random_patches) == max_iter


def test_compute_average_psd():
    micrograph = torch.randn(128, 128)
    transform = transforms.Compose([NormalizeIndividual()])
    random_patches = mu.RandomMicrographPatches(
        micro_graphs=[micrograph], patch_size=10, transform=transform, max_iter=100
    )
    avg_psd = mu.compute_average_psd(random_patches)
    assert avg_psd.shape == torch.Size([10, 10])
