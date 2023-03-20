import pytest
import torch
import numpy as np
import json

from cryo_sbi.inference.models.embedding_nets import EMBEDDING_NETS
from cryo_sbi.inference.validate_train_config import check_train_params


def test_embedding():
    for num_images in [1, 10, 100]:
        for out_dim in [10, 100]:
            test_images = torch.randn(num_images, 64, 64)
            for name, embedding in EMBEDDING_NETS.items():
                if name == "CNN" or name == "DEEPCNN":
                    continue
                out = embedding(out_dim)(test_images).shape
                assert out == torch.Size([num_images, out_dim]), name
