import pytest
import torch
from itertools import product

from cryo_sbi.inference.models.embedding_nets import EMBEDDING_NETS

embedding_networks = list(EMBEDDING_NETS.keys())
num_images_to_test = [1, 5]
out_dims_to_test = [10, 100]
cases_to_test = list(product(embedding_networks, num_images_to_test, out_dims_to_test))


@pytest.mark.parametrize(("embedding_name", "num_images", "out_dim"), cases_to_test)
def test_embedding(embedding_name, num_images, out_dim):

    if "FFT_FILTER_" in embedding_name:
        size = embedding_name.split("FFT_FILTER_")[1]
        test_images = torch.randn(num_images, int(size), int(size))
    else:
        test_images = torch.randn(num_images, 128, 128)

    embedding = EMBEDDING_NETS[embedding_name](out_dim)
    out = embedding(test_images).shape
    assert out == torch.Size([num_images, out_dim]), embedding_name
