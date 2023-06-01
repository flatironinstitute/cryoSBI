from typing import Union
import json
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset
from torchvision import transforms
from tqdm import tqdm
from lampe.data import JointLoader, H5Dataset
from lampe.inference import NPELoss
from lampe.utils import GDStep
from itertools import islice

from cryo_sbi.inference.priors import get_uniform_prior_1d
from cryo_sbi.inference.models.build_models import build_npe_flow_model
from cryo_sbi.inference.validate_train_config import check_train_params
from cryo_sbi import CryoEmSimulator
from cryo_sbi.utils.image_utils import WhitenImage, NormalizeIndividual


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
    train_config = json.load(open(train_config))
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
    whiten_filter: Union[None, str] = None,
    **simulator_kwargs,
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

    cryo_simulator = CryoEmSimulator(image_config, **simulator_kwargs)

    estimator = load_model(
        train_config, model_state_dict, device, train_from_checkpoint
    )

    train_config = json.load(open(train_config))

    loader = JointLoader(
        get_uniform_prior_1d(cryo_simulator.max_index),
        cryo_simulator.simulator,
        vectorized=False,
        batch_size=train_config["BATCH_SIZE"],
        num_workers=n_workers,
        pin_memory=True,
        prefetch_factor=1,
    )

    if isinstance(whiten_filter, str):
        whiten_filter = torch.load(whiten_filter).to(device=device)
        whitening_transform = transforms.Compose(
            [
                WhitenImage(whiten_filter),
                NormalizeIndividual(),
            ]
        )
    else:  # Refactor this so weird lambda is not needed
        whitening_transform = lambda x: x

    loss = NPELoss(estimator)
    optimizer = optim.AdamW(estimator.parameters(), lr=train_config["LEARNING_RATE"])
    step = GDStep(optimizer, clip=train_config["CLIP_GRADIENT"])
    mean_loss = []

    print("Training neural netowrk:")
    estimator.train()
    with tqdm(range(epochs), unit="epoch") as tq:
        for epoch in tq:
            losses = torch.stack(
                [
                    step(
                        loss(
                            theta.to(device=device, non_blocking=True),
                            whitening_transform(x.to(device=device, non_blocking=True)),
                        )
                    )
                    for theta, x in islice(loader, 1000)
                ]
            )

            tq.set_postfix(loss=losses.mean().item())
            mean_loss.append(losses.mean().item())
            if epoch % saving_frequency == 0:
                torch.save(estimator.state_dict(), estimator_file + f"_epoch={epoch}")

    torch.save(estimator.state_dict(), estimator_file)
    torch.save(torch.tensor(mean_loss), loss_file)


def npe_train_from_vram(
    train_config: str,
    epochs: int,
    train_data_dir: str,
    val_data_dir: str,
    estimator_file: str,
    loss_file: str,
    train_from_checkpoint: bool = False,
    model_state_dict: Union[str, None] = None,
    device: str = "cpu",
    saving_frequency: int = 20,
    num_workers: int = 0,
) -> None:
    """
    Train NPE model with data loaded from VRAM.
    Saves model and loss to disk.

    Args:
        train_config (str): path to train config file
        epochs (int): number of epochs
        train_data_dir (str): path to train data directory
        val_data_dir (str): path to validation data directory
        estimator_file (str): path to estimator file
        loss_file (str): path to loss file
        train_from_checkpoint (bool, optional): Whether to start training from checkpoint. Defaults to False.
        model_state_dict (Union[str, None], optional): Path to pretrained model state dict. Defaults to None.
        device (str, optional): training device. Defaults to "cpu".
        saving_frequency (int, optional): frequency of saving model. Defaults to 20.
        num_workers (int, optional): number of workers for train and val loaders. Defaults to 0.

    Raises:
        Warning: No model state dict specified! --model_state_dict is empty

    Returns:
        None
    """

    estimator = load_model(
        train_config, model_state_dict, device, train_from_checkpoint
    )

    train_config = json.load(open(train_config))

    trainset = TensorDataset(
        torch.load(
            f"{train_data_dir}_theta_train.pt", map_location=torch.device("cuda")
        ),
        torch.load(f"{train_data_dir}_x_train.pt", map_location=torch.device("cuda")),
    )

    validset = TensorDataset(
        torch.load(f"{val_data_dir}_theta_val.pt", map_location=torch.device("cuda")),
        torch.load(f"{val_data_dir}_x_val.pt", map_location=torch.device("cuda")),
    )

    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=train_config["BATCH_SIZE"],
        num_workers=num_workers,
        pin_memory=False,
        prefetch_factor=2,
        shuffle=True,
    )

    val_loader = torch.utils.data.DataLoader(
        validset,
        batch_size=train_config["BATCH_SIZE"],
        num_workers=num_workers,
        pin_memory=False,
        prefetch_factor=2,
        shuffle=True,
    )

    loss = NPELoss(estimator)
    optimizer = optim.AdamW(estimator.parameters(), lr=train_config["LEARNING_RATE"])
    step = GDStep(optimizer, clip=train_config["CLIP_GRADIENT"])
    mean_val_loss = []
    mean_train_loss = []

    print("Training neural netowrk:")
    with tqdm(range(epochs), unit="epoch") as tq:
        for epoch in tq:
            estimator.train()
            train_losses = torch.stack(
                [step(loss(theta, x)) for theta, x in train_loader]
            )
            estimator.eval()
            with torch.no_grad():
                val_losses = torch.stack([loss(theta, x) for theta, x in val_loader])
            tq.set_postfix(
                train_loss=train_losses.mean().item(), val_loss=val_losses.mean().item()
            )
            mean_train_loss.append(train_losses.mean().item())
            mean_val_loss.append(val_losses.mean().item())
            if epoch % saving_frequency == 0:
                torch.save(estimator.state_dict(), estimator_file + f"_epoch={epoch}")

    torch.save(estimator.state_dict(), estimator_file)
    torch.save(torch.tensor((mean_train_loss, mean_val_loss)), loss_file)


def npe_train_from_disk(
    train_config: str,
    epochs: int,
    train_data_dir: str,
    val_data_dir: str,
    estimator_file: str,
    loss_file: str,
    train_from_checkpoint: bool = False,
    model_state_dict: Union[str, None] = None,
    n_workers: int = 1,
    device: str = "cpu",
    saving_frequency: int = 20,
) -> None:
    """
    Train NPE model with data loaded from disk.
    Saves model and loss to disk.

    Args:
        train_config (str): path to train config file
        epochs (int): number of epochs
        train_data_dir (str): path to train data directory
        val_data_dir (str): path to validation data directory
        estimator_file (str): path to estimator file
        loss_file (str): path to loss file
        train_from_checkpoint (bool, optional): Whether to start training from checkpoint. Defaults to False.
        model_state_dict (Union[str, None], optional): path to model state dict. Defaults to None.
        n_workers (int, optional): number of workers. Defaults to 1.
        device (str, optional): training device. Defaults to "cpu".
        saving_frequency (int, optional): frequency of saving model. Defaults to 20.

    Raises:
        Warning: No model state dict specified! --model_state_dict is empty

    Returns:
        None
    """
    estimator = load_model(
        train_config, model_state_dict, device, train_from_checkpoint
    )

    train_config = json.load(open(train_config))

    trainset = H5Dataset(
        train_data_dir,
        shuffle=True,
        batch_size=train_config["BATCH_SIZE"],
        chunk_size=train_config["BATCH_SIZE"],
        chunk_step=2**2,
    )

    validset = H5Dataset(
        val_data_dir,
        shuffle=True,
        batch_size=train_config["BATCH_SIZE"],
        chunk_size=train_config["BATCH_SIZE"],
        chunk_step=2**2,
    )

    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=None,
        num_workers=n_workers,
        pin_memory=True,
        prefetch_factor=100,
    )

    val_loader = torch.utils.data.DataLoader(
        validset,
        batch_size=None,
        num_workers=n_workers,
        pin_memory=True,
        prefetch_factor=100,
    )

    loss = NPELoss(estimator)
    optimizer = optim.AdamW(estimator.parameters(), lr=train_config["LEARNING_RATE"])
    step = GDStep(optimizer, clip=train_config["CLIP_GRADIENT"])
    mean_loss = []

    print("Training neural netowrk:")
    with tqdm(range(epochs), unit="epoch") as tq:
        for epoch in tq:
            estimator.train()
            train_losses = torch.stack(
                [
                    step(loss(theta.cuda(non_blocking=True), x.cuda(non_blocking=True)))
                    for theta, x in train_loader
                ]
            )

            estimator.eval()
            with torch.no_grad():
                val_losses = torch.stack(
                    [loss(theta.cuda(), x.cuda()) for theta, x in val_loader]
                )

            if epoch % saving_frequency == 0:
                torch.save(estimator.state_dict(), estimator_file + f"_epoch={epoch}")

            tq.set_postfix(
                train_loss=train_losses.mean().item(), val_loss=val_losses.mean().item()
            )

            mean_loss.append(val_losses.mean().item())

    torch.save(estimator, estimator_file)
    torch.save(torch.tensor(mean_loss), loss_file)
