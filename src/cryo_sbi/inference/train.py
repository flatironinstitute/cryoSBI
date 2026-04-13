import os
from typing import Union
import json
import torch
import time
import torch.nn as nn
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from itertools import islice
import logging

from cryo_sbi.inference.priors import get_image_priors, PriorLoader
from cryo_sbi.inference.models.build_models import build_classifier
from cryo_sbi.wpa_simulator.cryo_em_simulator import cryo_em_simulator
from cryo_sbi.utils.check_config import check_train_params, check_image_params

torch.backends.cudnn.benchmark = True

def setup_logging(debug: bool = False):
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


class ClassifierLoss(nn.Module):
    def __init__(
        self, estimator: torch.nn.Module, label_smoothing: float = 0.0
    ) -> None:
        super().__init__()
        self.estimator = estimator
        self.label_smoothing = label_smoothing

    def forward(self, indices: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        logits = self.estimator(images)
        return torch.nn.functional.cross_entropy(
            logits, indices, reduction="mean", label_smoothing=self.label_smoothing
        )


class GDStep:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        clip: float = None,
        lr_scheduler: torch.optim.lr_scheduler = None,
    ) -> None:
        self.optimizer = optimizer
        self.parameters = [
            p for group in optimizer.param_groups for p in group["params"]
        ]
        self.clip = clip
        self.lr_scheduler = lr_scheduler

    def __call__(self, loss: torch.tensor) -> torch.tensor:
        if loss.isfinite().all():
            self.optimizer.zero_grad()
            loss.backward()

            if self.clip is None:
                self.optimizer.step()
            else:
                norm = nn.utils.clip_grad_norm_(self.parameters, self.clip)
                if norm.isfinite():
                    self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        return loss.detach()


def load_model(
    train_config: str, model_state_dict: str, device: str, train_from_checkpoint: bool
) -> torch.nn.Module:
    train_config = check_train_params(train_config)
    estimator = build_classifier(train_config)
    if train_from_checkpoint:
        if not os.path.isfile(model_state_dict):
            raise ValueError(
                "Model state dict file does not exist. Please provide a valid path."
            )
        print(f"Loading model parameters from {model_state_dict}")
        estimator.load_state_dict(torch.load(model_state_dict, weights_only=True))
    estimator.to(device=device)
    return estimator


def train_classifier(
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
    setup_logging()

    train_config = json.load(open(train_config))
    train_config = check_train_params(train_config)
    image_config = json.load(open(image_config))
    image_config = check_image_params(image_config)

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
        models = torch.load(image_config["MODEL_FILE"]).to(device).to(torch.float32)

    if models.ndim == 3:
        num_models = models.shape[0]
        num_representatives = None
    if models.ndim == 4:
        num_models = models.shape[0]
        num_representatives = models.shape[1]

    logging.info(
        f"Training on {num_models} models with {num_representatives if num_representatives is not None else 1} representatives"
    )

    image_prior = get_image_priors(
        num_models, num_representatives, image_config, device="cpu"
    )
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

    loss = ClassifierLoss(estimator)

    optimizer = optim.AdamW(
        estimator.parameters(),
        lr=train_config["LEARNING_RATE"],
        weight_decay=train_config["WEIGHT_DECAY"],
    )

    if train_config["ONE_CYCLE_SCHEDULER"]:
        logging.info("Using One Cycle LR Scheduler")
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=train_config["LEARNING_RATE"],
            total_steps=epochs
            * 100
            * (simulation_batch_size // train_config["BATCH_SIZE"]),
        )
    else:
        lr_scheduler = None

    step = GDStep(
        optimizer, clip=train_config["CLIP_GRADIENT"], lr_scheduler=lr_scheduler
    )
    mean_loss = []

    logging.info("Starting training loop")
    start_time = time.time()
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
                indices = indices[:, 0] if indices.ndim == 2 else indices
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

            tq.set_postfix(
                loss=losses.mean().item(), lr=optimizer.param_groups[0]["lr"]
            )
            mean_loss.append(losses.mean().item())
            if epoch % saving_frequency == 0:
                torch.save(estimator.state_dict(), estimator_file + f"_epoch={epoch}")

    end_time = time.time()
    logging.info(f"Training completed in {end_time - start_time:.2f} seconds")
    torch.save(estimator.state_dict(), estimator_file)
    torch.save(torch.tensor(mean_loss), loss_file)
