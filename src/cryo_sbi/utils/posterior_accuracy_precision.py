from estimator_utils import sample_posterior
from wpa_simulator.cryo_em_simulator import CryoEmSimulator


def posterior_accuracy_precision(
    estimator, image_config, num_samples, num_posterior_samples, alpha
):
    # !!! needs image config which also generates different kinds of noise
    # draw num_samples prior samples
    # run simulations with prior samples
    # draw num_posterior_samples from posterior
    # compute mean posterior - true index
    # compute confidence intervall
    # retrurn mean posterior - true index, confidence intervall
    pass
