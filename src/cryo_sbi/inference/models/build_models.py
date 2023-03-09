import torch.nn as nn
from functools import partial
import zuko
import lampe
import cryo_sbi.inference.models.estimator_models as estimator_models
from cryo_sbi.inference.models.embedding_nets import EMBEDDING_NETS


def build_npe_flow_model(config):
    """
    Function to build NPE estimator with embedding net
    from config_file
    """

    if config["MODEL"] == "MAF":
        model = zuko.flows.MAF
    elif config["MODEL"] == "NSF":
        model = zuko.flows.NSF
    elif config["MODEL"] == "SOSPF":
        model = zuko.flows.SOSPF
    else:
        raise NotImplementedError(
            f"Model : {config['MODEL']} has not been implemented yet!"
        )

    try:
        embedding = partial(EMBEDDING_NETS[config["EMBEDDING"]], config["OUT_DIM"])
    except:
        raise NotImplementedError(
            f"Model : {config['EMBEDDING']} has not been implemented yet! \
The following embeddings are implemented : {[key for key in EMBEDDING_NETS.keys()]}"
        )

    estimator = estimator_models.NPEWithEmbedding(
        embedding_net=embedding,
        output_embedding_dim=config["OUT_DIM"],
        num_transforms=config["NUM_TRANSFORM"],
        num_hidden_flow=config["NUM_HIDDEN_FLOW"],
        hidden_flow_dim=config["HIDDEN_DIM_FLOW"],
        flow=model,
        theta_shift=config["THETA_SHIFT"],
        theta_scale=config["THETA_SCALE"],
        **{"activation": partial(nn.LeakyReLU, 0.1)},
    )

    return estimator


def build_nre_classifier_model(config):
    """
    Function to build NRE estimator with embedding net
    from config_file
    """

    if config["MODEL"] == "RESMLP":
        model = lampe.nn.ResMLP
    elif config["MODEL"] == "MLP":
        model = zuko.nn.MLP
    else:
        raise NotImplementedError(
            f"Model : {config['MODEL']} has not been implemented yet!"
        )

    try:
        embedding = partial(EMBEDDING_NETS[config["EMBEDDING"]], config["OUT_DIM"])
    except:
        raise NotImplementedError(
            f"Model : {config['EMBEDDING']} has not been implemented yet! \
The following embeddings are implemented : {[key for key in EMBEDDING_NETS.keys()]}"
        )

    estimator = estimator_models.NREWithEmbedding(
        embedding_net=embedding,
        output_embedding_dim=config["OUT_DIM"],
        hidden_features=config["HIDDEN_FEATURES"],
        activation=partial(nn.LeakyReLU, 0.1),
        network=model,
        theta_scale=config["THETA_SCALE"],
        theta_shift=config["THETA_SHIFT"],
    )

    return estimator
