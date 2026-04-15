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

    def forward(
        self, indices: torch.Tensor, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.estimator(images)
        loss = nn.functional.cross_entropy(
            logits, indices, reduction="mean", label_smoothing=self.label_smoothing
        )
        return loss, logits


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
        cfg: Hydra DictConfig with keys cfg.simulation and cfg.train.
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

    train_cfg = cfg.train
    image_cfg = cfg.simulation

    batch_size = train_cfg.batch_size
    assert simulation_batch_size >= batch_size
    assert simulation_batch_size % batch_size == 0

    simulator = CryoEmSimulator(image_cfg, device=device)

    if simulator.garbage_class:
        train_cfg.classifier.num_classes = simulator.num_models + 1
        logging.info(
            f"Garbage class enabled, num_classes set to {simulator.num_models + 1}"
        )

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
    writer = SummaryWriter(log_dir=cfg.train.output.tensorboard_dir)
    hparams = OmegaConf.to_container(cfg.train, resolve=True)
    writer.add_hparams(
        {k: str(v) if isinstance(v, (dict, list)) else v for k, v in hparams.items()},
        metric_dict={},
    )

    os.makedirs(cfg.train.output.checkpoint_dir, exist_ok=True)
    estimator_file = cfg.train.output.estimator_file
    os.makedirs(os.path.dirname(estimator_file) or ".", exist_ok=True)

    logging.info("Starting training loop")
    start_time = time.time()
    estimator.train()

    with tqdm(range(epochs), unit="epoch") as tq:
        for epoch in tq:
            epoch_losses = []
            epoch_accs = []
            epoch_samples = 0
            epoch_start = time.time()

            for parameters in islice(prior_loader, batches_per_epoch):
                images = simulator.simulate(*parameters)

                # First element is always fg model indices
                fg_indices = parameters[0]
                batch_indices = fg_indices[:, 0] if fg_indices.ndim == 2 else fg_indices
                batch_indices = batch_indices.clone()

                if simulator.garbage_class:
                    garbage_mask = parameters[13]
                    batch_indices[garbage_mask] = simulator.num_models

                for _idx, _img in zip(
                    batch_indices.split(batch_size),
                    images.split(batch_size),
                ):
                    _idx_dev = _idx.to(device, non_blocking=True)
                    _img_dev = _img.to(device, non_blocking=True)

                    loss, logits = loss_fn(_idx_dev, _img_dev)
                    batch_loss, _ = step(loss)

                    with torch.no_grad():
                        acc = (_idx_dev == logits.argmax(dim=1)).float().mean()

                    epoch_losses.append(batch_loss)
                    epoch_accs.append(acc)
                    epoch_samples += _idx.shape[0]

            mean_loss = torch.stack(epoch_losses).mean().item()
            mean_acc = torch.stack(epoch_accs).mean().item()
            current_lr = optimizer.param_groups[0]["lr"]
            throughput = epoch_samples / (time.time() - epoch_start)
            writer.add_scalar("Loss/epoch", mean_loss, epoch)
            writer.add_scalar("Accuracy/epoch", mean_acc, epoch)
            writer.add_scalar("LR/epoch", current_lr, epoch)
            writer.add_scalar("Throughput/epoch", throughput, epoch)
            tq.set_postfix(loss=mean_loss, acc=f"{mean_acc:.3f}", lr=current_lr)

            if epoch % saving_frequency == 0:
                ckpt_path = os.path.join(
                    cfg.train.output.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt"
                )
                torch.save(estimator.state_dict(), ckpt_path)

    end_time = time.time()
    logging.info(f"Training completed in {end_time - start_time:.2f} seconds")
    torch.save(estimator.state_dict(), estimator_file)
    writer.close()
