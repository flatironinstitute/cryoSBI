import os
import time
import logging
from typing import Union

import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from itertools import islice
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from cryo_sbi.simulator.priors import PriorLoader
from cryo_sbi.simulator.cryo_em_simulator import CryoEmSimulator
from cryo_sbi.simulator.multi_particle_simulator import MultiParticleCryoEmSimulator
from cryo_sbi.models.build_models import build_classifier

torch.backends.cudnn.benchmark = True


def setup_logging(debug: bool = False):
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


class ClassifierLoss(nn.Module):
    def __init__(self, estimator: nn.Module, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.estimator = estimator
        self.label_smoothing = label_smoothing

    def forward(self, indices: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        logits = self.estimator(images)
        return nn.functional.cross_entropy(
            logits, indices, reduction="mean", label_smoothing=self.label_smoothing
        )


class GDStep:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        clip: float = None,
        lr_scheduler=None,
    ) -> None:
        self.optimizer = optimizer
        self.parameters = [
            p for group in optimizer.param_groups for p in group["params"]
        ]
        self.clip = clip
        self.lr_scheduler = lr_scheduler

    def __call__(self, loss: torch.Tensor) -> torch.Tensor:
        if loss.isfinite().all():
            self.optimizer.zero_grad()
            loss.backward()

            if self.clip is None:
                self.optimizer.step()
                grad_norm = None
            else:
                grad_norm = nn.utils.clip_grad_norm_(self.parameters, self.clip)
                if grad_norm.isfinite():
                    self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            return loss.detach(), grad_norm

        return loss.detach(), None


def load_model(
    train_cfg: DictConfig,
    model_state_dict: Union[str, None],
    device: str,
    train_from_checkpoint: bool,
) -> nn.Module:
    estimator = build_classifier(train_cfg)
    if train_from_checkpoint:
        if not os.path.isfile(model_state_dict):
            raise ValueError(f"Checkpoint not found: {model_state_dict}")
        logging.info(f"Loading model parameters from {model_state_dict}")
        estimator.load_state_dict(
            torch.load(model_state_dict, weights_only=True)
        )
    estimator.to(device=device)
    return estimator


def train_classifier(cfg: DictConfig) -> None:
    """
    Main training function.

    Args:
        cfg: Hydra DictConfig with the full config tree (image, training, train, output, mode).
    """
    setup_logging()

    device = cfg.train.device
    epochs = cfg.train.epochs
    n_workers = cfg.train.n_workers
    saving_frequency = cfg.train.saving_frequency
    simulation_batch_size = cfg.train.simulation_batch_size
    batches_per_epoch = cfg.train.batches_per_epoch
    prefetch_factor = cfg.train.prefetch_factor
    train_from_checkpoint = cfg.train.train_from_checkpoint
    checkpoint_file = cfg.train.get("checkpoint_file", None)

    train_cfg = cfg.training   # embedding + classifier + lr + ...
    image_cfg = cfg.image      # n_pixels, pixel_size, sigma, ...

    batch_size = train_cfg.batch_size
    assert simulation_batch_size >= batch_size
    assert simulation_batch_size % batch_size == 0

    # Build simulator
    if cfg.mode.multi_particle:
        logging.info("Using MultiParticleCryoEmSimulator")
        simulator = MultiParticleCryoEmSimulator(image_cfg, device=device)
    else:
        simulator = CryoEmSimulator(image_cfg, device=device)

    logging.info(
        f"Training on {simulator.num_models} models with "
        f"{simulator.num_representatives if simulator.num_representatives is not None else 1} representatives"
    )

    prior_loader = PriorLoader(
        simulator._priors,
        batch_size=simulation_batch_size,
        num_workers=n_workers,
        prefetch_factor=prefetch_factor,
    )

    estimator = load_model(train_cfg, checkpoint_file, device, train_from_checkpoint)
    loss_fn = ClassifierLoss(estimator)

    optimizer = optim.AdamW(
        estimator.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )

    lr_scheduler = None
    if train_cfg.one_cycle_scheduler:
        logging.info("Using OneCycleLR scheduler")
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=train_cfg.learning_rate,
            total_steps=epochs * batches_per_epoch * (simulation_batch_size // batch_size),
        )

    step = GDStep(optimizer, clip=train_cfg.clip_gradient, lr_scheduler=lr_scheduler)

    # TensorBoard
    tb_dir = cfg.output.tensorboard_dir
    writer = SummaryWriter(log_dir=tb_dir)
    hparams = OmegaConf.to_container(train_cfg, resolve=True)
    hparams.update(OmegaConf.to_container(cfg.train, resolve=True))
    writer.add_hparams(
        {k: str(v) if isinstance(v, (dict, list)) else v for k, v in hparams.items()},
        metric_dict={},
    )

    os.makedirs(cfg.output.checkpoint_dir, exist_ok=True)
    estimator_file = cfg.output.estimator_file
    os.makedirs(os.path.dirname(estimator_file) or ".", exist_ok=True)

    logging.info("Starting training loop")
    global_step = 0
    start_time = time.time()
    estimator.train()

    with tqdm(range(epochs), unit="epoch") as tq:
        for epoch in tq:
            epoch_losses = []

            for parameters in islice(prior_loader, batches_per_epoch):
                images = simulator.simulate(*parameters)

                # First element is always fg model indices
                fg_indices = parameters[0]
                batch_indices = fg_indices[:, 0] if fg_indices.ndim == 2 else fg_indices

                for _idx, _img in zip(
                    batch_indices.split(batch_size),
                    images.split(batch_size),
                ):
                    batch_loss, grad_norm = step(
                        loss_fn(
                            _idx.to(device, non_blocking=True),
                            _img.to(device, non_blocking=True),
                        )
                    )
                    epoch_losses.append(batch_loss)
                    writer.add_scalar("Loss/batch", batch_loss.item(), global_step)
                    writer.add_scalar(
                        "LR/step", optimizer.param_groups[0]["lr"], global_step
                    )
                    if grad_norm is not None:
                        writer.add_scalar("Gradients/norm", grad_norm.item(), global_step)
                    global_step += 1

            mean_loss = torch.stack(epoch_losses).mean().item()
            writer.add_scalar("Loss/epoch_mean", mean_loss, epoch)
            tq.set_postfix(loss=mean_loss, lr=optimizer.param_groups[0]["lr"])

            if epoch % saving_frequency == 0:
                ckpt_path = os.path.join(
                    cfg.output.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt"
                )
                torch.save(estimator.state_dict(), ckpt_path)

    end_time = time.time()
    logging.info(f"Training completed in {end_time - start_time:.2f} seconds")
    torch.save(estimator.state_dict(), estimator_file)
    writer.close()
