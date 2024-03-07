import pytest
import torch
from cryo_sbi.utils import image_utils as iu


def test_circular_mask():
    n_pixels = 100
    radius = 30
    inside_mask = iu.circular_mask(n_pixels, radius, inside=True)
    outside_mask = iu.circular_mask(n_pixels, radius, inside=False)

    assert inside_mask.shape == (n_pixels, n_pixels)
    assert outside_mask.shape == (n_pixels, n_pixels)
    assert inside_mask.sum().item() == pytest.approx(radius**2 * 3.14159, abs=10)
    assert outside_mask.sum().item() == pytest.approx(
        n_pixels**2 - radius**2 * 3.14159, abs=10
    )


def test_mask_class():
    image_size = 100
    radius = 30
    inside = True
    mask = iu.Mask(image_size, radius, inside=inside)
    image = torch.ones((image_size, image_size))

    masked_image = mask(image)
    assert masked_image.shape == (image_size, image_size)
    assert masked_image[inside].sum().item() == pytest.approx(
        image_size**2 - radius**2 * 3.14159, abs=10
    )


def test_fourier_down_sample():
    image_size = 100
    n_pixels = 30
    image = torch.ones((image_size, image_size))

    downsampled_image = iu.fourier_down_sample(image, image_size, n_pixels)
    assert downsampled_image.shape == (
        image_size - 2 * n_pixels,
        image_size - 2 * n_pixels,
    )


def test_fourier_down_sample_class():
    image_size = 100
    down_sampled_size = 40
    down_sampler = iu.FourierDownSample(image_size, down_sampled_size)
    image = torch.ones((image_size, image_size))

    down_sampled_image = down_sampler(image)
    assert down_sampled_image.shape == (
        image_size - 2 * down_sampler._n_pixels,
        image_size - 2 * down_sampler._n_pixels,
    )


def test_low_pass_filter():
    image_size = 100
    frequency_cutoff = 30
    low_pass_filter = iu.LowPassFilter(image_size, frequency_cutoff)
    image = torch.ones((image_size, image_size))

    filtered_image = low_pass_filter(image)
    assert filtered_image.shape == (image_size, image_size)


def test_gaussian_low_pass_filter():
    image_size = 100
    frequency_cutoff = 30
    low_pass_filter = iu.GaussianLowPassFilter(image_size, frequency_cutoff)
    image = torch.ones((image_size, image_size))

    filtered_image = low_pass_filter(image)
    assert filtered_image.shape == (image_size, image_size)


def test_normalize_individual():
    normalize_individual = iu.NormalizeIndividual()
    image = torch.randn((3, 100, 100))

    normalized_image = normalize_individual(image)
    assert normalized_image.shape == (3, 100, 100)
    assert normalized_image.mean().item() == pytest.approx(0.0, abs=1e-1)
    assert normalized_image.std().item() == pytest.approx(1.0, abs=1e-1)


def test_mrc_to_tensor():
    image_path = "tests/data/test.mrc"
    image = iu.mrc_to_tensor(image_path)

    assert isinstance(image, torch.Tensor)
    assert image.shape == (5, 5)


def test_image_whithening():
    whitening_transform = iu.WhitenImage(torch.randn((100, 100)))
    images = torch.randn((100, 100))
    images_whitened = whitening_transform(images)
    assert images_whitened.shape == (100, 100)


def test_image_whithening_batched():
    whitening_transform = iu.WhitenImage(torch.randn((100, 100)))
    images = torch.randn((10, 100, 100))
    images_whitened = whitening_transform(images)
    assert images_whitened.shape == (10, 100, 100)
