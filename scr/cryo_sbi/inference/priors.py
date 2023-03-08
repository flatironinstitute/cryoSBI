import torch
import zuko


def get_unirom_prior_1d(max_index):
    """Return uniform prior in 1d from 0 to 19"""

    assert isinstance(max_index, int), 'max_index is no INT'

    LOWER = torch.zeros(1)
    UPPER = max_index * torch.ones(1)
    prior = zuko.distributions.BoxUniform(LOWER, UPPER)

    return prior
