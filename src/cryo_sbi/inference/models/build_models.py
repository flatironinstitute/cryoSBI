import torch.nn as nn
from functools import partial
import zuko
import lampe
import cryo_sbi.inference.models.estimator_models as estimator_models
from cryo_sbi.inference.models.embedding_nets import EMBEDDING_NETS


def build_npe_flow_model(config: dict, **embedding_kwargs) -> nn.Module:
    """
    Function to build NPE estimator with embedding net
    from config_file

    Args:
        config (dict): config file
        embedding_kwargs (dict): kwargs for embedding net

    Returns:
        estimator (nn.Module): NPE estimator
    """

    if config["MODEL"] == "MAF":
        model = zuko.flows.MAF
    elif config["MODEL"] == "GF":
        model = zuko.flows.GF
    elif config["MODEL"] == "CNF":
        model = zuko.flows.CNF
    elif config["MODEL"] == "UMNN":
        model = zuko.flows. UMNN
    elif config["MODEL"] == "NSF":
        model = zuko.flows.NSF
    elif config["MODEL"] == "SOSPF":
        model = partial(zuko.flows.SOSPF, polynomials=8, degree=5)
    else:
        raise NotImplementedError(
            f"Model : {config['MODEL']} has not been implemented yet!"
        )

    try:
        embedding = partial(
            EMBEDDING_NETS[config["EMBEDDING"]], config["OUT_DIM"], **embedding_kwargs
        )
    except KeyError:
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
        **{"activation": nn.GELU},
    )

    return estimator


def build_fmpe_flow_model(config: dict, **embedding_kwargs) -> nn.Module:
    
    try:
        embedding = partial(
            EMBEDDING_NETS[config["EMBEDDING"]], config["OUT_DIM"], **embedding_kwargs
        )
    except KeyError:
        raise NotImplementedError(
            f"Model : {config['EMBEDDING']} has not been implemented yet! \
The following embeddings are implemented : {[key for key in EMBEDDING_NETS.keys()]}"
        )
    
    estimator = estimator_models.FMPEWithEmbedding(
        embedding_net=embedding,
        output_embedding_dim=config["OUT_DIM"],
        num_hidden_flow=config["NUM_HIDDEN_FLOW"],
        hidden_flow_dim=config["HIDDEN_DIM_FLOW"],
        theta_shift=config["THETA_SHIFT"],
        theta_scale=config["THETA_SCALE"],
    )

    return estimator


def build_nre_classifier_model(config: dict, **embedding_kwargs) -> nn.Module:
    raise NotImplementedError("NRE classifier model has not been implemented yet!")
