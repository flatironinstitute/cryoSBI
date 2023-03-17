from estimator_utils import sample_posterior
from cryo_sbi import CryoEmSimulator


def posterior_accuracy_precision(estimator, train_config, image_config, num_samples, num_posterior_samples):
    # !!! needs image config which also generates different kinds of noise
    # draw num_samples prior samples
    # run simulations with prior samples
    # draw num_posterior_samples from posterior
    # compute mean posterior - true index
    # compute confidence intervall
    # retrurn mean posterior - true index, confidence intervall
    pass