import torch
from torch.distributions.distribution import Distribution


class QuatDistribution(Distribution):

    arg_constraints = {}
    has_rsample = True

    def __init__(self):
        
        batch_shape = (1, )
        super().__init__(batch_shape,)

    def rsample(self, sample_shape=torch.Size()):

        shape = self._extended_shape(sample_shape)
        n_quats = shape[0]

        quats = torch.zeros((n_quats, 4))
        count = 0
        while count < n_quats:
            quat = torch.rand(4) * 2.0 - 1.0
            norm = torch.linalg.vector_norm(quat, ord=2)

            if 0.2 <= norm <= 1.0:
                quat /= norm
                quats[count] = quat
                count += 1

        return quats
