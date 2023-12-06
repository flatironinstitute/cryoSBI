from typing import Union
import json
import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import TensorDataset
from torchvision import transforms
from tqdm import tqdm
from lampe.data import JointLoader, H5Dataset
from lampe.inference import NPELoss
from lampe.utils import GDStep
from itertools import islice

from cryo_sbi.inference.priors import get_image_priors, PriorLoader
from cryo_sbi.inference.models.build_models import build_npe_flow_model
from cryo_sbi.inference.validate_train_config import check_train_params
from cryo_sbi.wpa_simulator.cryo_em_simulator import cryo_em_simulator
from cryo_sbi.wpa_simulator.validate_image_config import check_image_params
from cryo_sbi.inference.validate_train_config import check_train_params
import cryo_sbi.utils.image_utils as img_utils


def load_model(
    train_config: str, model_state_dict: str, device: str, train_from_checkpoint: bool
) -> torch.nn.Module:
    """
    Load model from checkpoint or from scratch.

    Args:
        train_config (str): path to train config file
        model_state_dict (str): path to model state dict
        device (str): device to load model to
        train_from_checkpoint (bool): whether to load model from checkpoint or from scratch
    """

    check_train_params(train_config)
    estimator = build_npe_flow_model(train_config)
    if train_from_checkpoint:
        if not isinstance(model_state_dict, str):
            raise Warning("No model state dict specified! --model_state_dict is empty")
        print(f"Loading model parameters from {model_state_dict}")
        estimator.load_state_dict(torch.load(model_state_dict))
    estimator.to(device=device)
    return estimator


def npe_train_no_saving(
    image_config: str,
    train_config: str,
    epochs: int,
    estimator_file: str,
    loss_file: str,
    train_from_checkpoint: bool = False,
    model_state_dict: Union[str, None] = None,
    n_workers: int = 1,
    device: str = "cpu",
    saving_frequency: int = 20,
    simulation_batch_size: int = 1024,
) -> None:
    """
    Train NPE model by simulating training data on the fly.
    Saves model and loss to disk.

    Args:
        image_config (str): path to image config file
        train_config (str): path to train config file
        epochs (int): number of epochs
        estimator_file (str): path to estimator file
        loss_file (str): path to loss file
        train_from_checkpoint (bool, optional): train from checkpoint. Defaults to False.
        model_state_dict (str, optional): path to pretrained model state dict. Defaults to None.
        n_workers (int, optional): number of workers. Defaults to 1.
        device (str, optional): training device. Defaults to "cpu".
        saving_frequency (int, optional): frequency of saving model. Defaults to 20.
        whiten_filter (Union[None, str], optional): path to whiten filter. Defaults to None.

    Raises:
        Warning: No model state dict specified! --model_state_dict is empty

    Returns:
        None
    """

    train_config = json.load(open(train_config))
    check_train_params(train_config)
    image_config = json.load(open(image_config))

    assert simulation_batch_size >= train_config["BATCH_SIZE"]
    assert simulation_batch_size % train_config["BATCH_SIZE"] == 0

    if image_config["MODEL_FILE"].endswith("npy"):
        models = (
            torch.from_numpy(
                np.load(image_config["MODEL_FILE"]),
            )
            .to(device)
            .to(torch.float32)
        )
    else:
        models = torch.load(
            image_config["MODEL_FILE"]).to(device).to(torch.float32)

    image_prior = get_image_priors(len(models) - 1, image_config, device="cpu")
    prior_loader = PriorLoader(
        image_prior, batch_size=simulation_batch_size, num_workers=n_workers
    )

    num_pixels = torch.tensor(
        image_config["N_PIXELS"], dtype=torch.float32, device=device
    )
    pixel_size = torch.tensor(
        image_config["PIXEL_SIZE"], dtype=torch.float32, device=device
    )

    estimator = load_model(
        train_config, model_state_dict, device, train_from_checkpoint
    )

    loss = NPELoss(estimator)
    optimizer = optim.AdamW(
        estimator.parameters(), lr=train_config["LEARNING_RATE"], weight_decay=0.001
    )
    step = GDStep(optimizer, clip=train_config["CLIP_GRADIENT"])
    mean_loss = []
    
    print("Training neural netowrk:")
    estimator.train()
    with tqdm(range(epochs), unit="epoch") as tq:
        for epoch in tq:
            losses = []
            for parameters in islice(prior_loader, 100):
                (
                    indices,
                    quaternions,
                    res,
                    shift,
                    defocus,
                    b_factor,
                    amp,
                    snr,
                ) = parameters
                images = cryo_em_simulator(
                    models,
                    indices.to(device, non_blocking=True),
                    quaternions.to(device, non_blocking=True),
                    res.to(device, non_blocking=True),
                    shift.to(device, non_blocking=True),
                    defocus.to(device, non_blocking=True),
                    b_factor.to(device, non_blocking=True),
                    amp.to(device, non_blocking=True),
                    snr.to(device, non_blocking=True),
                    num_pixels,
                    pixel_size,
                )
                for _indices, _images in zip(
                    indices.split(train_config["BATCH_SIZE"]),
                    images.split(train_config["BATCH_SIZE"]),
                ):
                    losses.append(
                        step(
                            loss(
                                _indices.to(device, non_blocking=True),
                                _images.to(device, non_blocking=True),
                            )
                        )
                    )
            losses = torch.stack(losses)

            tq.set_postfix(loss=losses.mean().item())
            mean_loss.append(losses.mean().item())
            if epoch % saving_frequency == 0:
                torch.save(estimator.state_dict(), estimator_file + f"_epoch={epoch}")

    torch.save(estimator.state_dict(), estimator_file)
    torch.save(torch.tensor(mean_loss), loss_file)
