import os
import time
import logging

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
    """
    One optimizer step, optionally with gradient clipping, LR scheduling, and
    AMP gradient scaling.

    The scaler is always used — when disabled (e.g. fp32 training or CPU), every
    GradScaler call is a no-op, so the same code path handles both modes.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        clip: float = None,
        lr_scheduler=None,
        scaler=None,
    ) -> None:
        self.optimizer = optimizer
        self.parameters = [
            p for group in optimizer.param_groups for p in group["params"]
        ]
        self.clip = clip
        self.lr_scheduler = lr_scheduler
        self.scaler = scaler if scaler is not None else torch.amp.GradScaler("cuda", enabled=False)

    def __call__(self, loss: torch.Tensor) -> torch.Tensor:
        if not loss.isfinite().all():
            return loss.detach(), None

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()

        if self.clip is not None:
            # unscale before clipping so the threshold has its real meaning.
            self.scaler.unscale_(self.optimizer)
            grad_norm = nn.utils.clip_grad_norm_(self.parameters, self.clip)
        else:
            grad_norm = None

        # scaler.step internally skips optimizer.step() if grads are non-finite.
        # Compare the loss scale before vs. after update() to detect a skip:
        # scale shrinks on a skipped step, stays same or grows on a successful one.
        scale_before = self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        stepped = self.scaler.get_scale() >= scale_before

        # Only step the scheduler when the optimizer actually stepped — otherwise
        # OneCycleLR drifts off its schedule after a single non-finite grad batch.
        if stepped and self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return loss.detach(), grad_norm


def _underlying_module(model: nn.Module) -> nn.Module:
    """Return the underlying nn.Module, unwrapping torch.compile if present."""
    return getattr(model, "_orig_mod", model)


def _save_checkpoint(
    path: str,
    estimator: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    epoch: int,
    scaler=None,
) -> None:
    """Save a full training checkpoint (model + optimizer + scheduler + epoch + RNG)."""
    state = {
        "model": _underlying_module(estimator).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
        "epoch": int(epoch),
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }
    torch.save(state, path)


def _load_checkpoint(
    path: str,
    estimator: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    scaler=None,
) -> int:
    """
    Load a training checkpoint into the given model/optimizer/scheduler.

    Accepts both the new dict format (model + optimizer + scheduler + epoch + RNG)
    and the legacy state-dict-only format. For legacy checkpoints, only model
    weights are restored and the resumed epoch is 0.

    Returns:
        int: epoch index to resume from (0 for legacy checkpoints).
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    state = torch.load(path, weights_only=False, map_location="cpu")

    # Legacy: a raw state_dict — keys are weight tensor names.
    if not isinstance(state, dict) or "model" not in state:
        logging.warning(
            f"Loading legacy weights-only checkpoint from {path}; "
            "optimizer / scheduler / epoch / RNG state will not be restored."
        )
        _underlying_module(estimator).load_state_dict(state)
        return 0

    _underlying_module(estimator).load_state_dict(state["model"])
    if "optimizer" in state and state["optimizer"] is not None:
        optimizer.load_state_dict(state["optimizer"])
    if lr_scheduler is not None and state.get("scheduler") is not None:
        lr_scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])
    if state.get("rng_state") is not None:
        torch.set_rng_state(state["rng_state"].cpu().to(torch.uint8))
    if (
        state.get("cuda_rng_state") is not None
        and torch.cuda.is_available()
    ):
        torch.cuda.set_rng_state_all(state["cuda_rng_state"])

    start_epoch = int(state.get("epoch", 0))
    logging.info(f"Resuming from {path} at epoch {start_epoch}")
    return start_epoch


def train_classifier(cfg: DictConfig) -> None:
    """
    Main training function.

    Args:
        cfg: Hydra DictConfig with keys cfg.simulation and cfg.train.
    """
    setup_logging()
    torch.backends.cudnn.benchmark = True

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
        # Build a local copy with the derived num_classes — mutating the Hydra
        # config in place can raise under struct mode and hides the original
        # value in saved hparams.
        train_cfg = OmegaConf.create(OmegaConf.to_container(train_cfg, resolve=True))
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

    if train_from_checkpoint and not checkpoint_file:
        raise ValueError(
            "train.train_from_checkpoint=true but train.checkpoint_file is unset. "
            "Provide a path with train.checkpoint_file=<path>."
        )

    use_amp = bool(getattr(train_cfg, "use_amp", False))
    compile_model = bool(getattr(train_cfg, "compile_model", False))
    is_cuda = str(device).startswith("cuda")
    if use_amp and not is_cuda:
        logging.warning(
            "train.use_amp=true ignored: AMP requires a CUDA device "
            f"(train.device={device})."
        )
        use_amp = False

    estimator = build_classifier(train_cfg).to(device=device)
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

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 0
    if train_from_checkpoint:
        start_epoch = _load_checkpoint(
            checkpoint_file, estimator, optimizer, lr_scheduler, scaler=scaler
        )
        if start_epoch >= epochs:
            raise ValueError(
                f"checkpoint_file resumes at epoch {start_epoch} but train.epochs={epochs}; "
                "increase train.epochs to continue training."
            )

    if compile_model:
        logging.info("Compiling model with torch.compile")
        estimator = torch.compile(estimator)
        loss_fn = ClassifierLoss(estimator)

    step = GDStep(
        optimizer,
        clip=train_cfg.clip_gradient,
        lr_scheduler=lr_scheduler,
        scaler=scaler,
    )

    # TensorBoard
    writer = SummaryWriter(log_dir=cfg.train.output.tensorboard_dir)

    os.makedirs(cfg.train.output.checkpoint_dir, exist_ok=True)
    estimator_file = cfg.train.output.estimator_file
    os.makedirs(os.path.dirname(estimator_file) or ".", exist_ok=True)

    logging.info("Starting training loop")
    start_time = time.time()
    estimator.train()
    final_loss = float("nan")
    final_acc = float("nan")
    global_step = start_epoch * batches_per_epoch * (simulation_batch_size // batch_size)

    with tqdm(range(start_epoch, epochs), unit="epoch", initial=start_epoch, total=epochs) as tq:
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

                    with torch.autocast(
                        device_type="cuda" if is_cuda else "cpu",
                        dtype=torch.float16,
                        enabled=use_amp,
                    ):
                        loss, logits = loss_fn(_idx_dev, _img_dev)
                    batch_loss, grad_norm = step(loss)

                    with torch.no_grad():
                        acc = (_idx_dev == logits.argmax(dim=1)).float().mean()

                    writer.add_scalar("Loss/batch", batch_loss.item(), global_step)
                    writer.add_scalar(
                        "LR/step", optimizer.param_groups[0]["lr"], global_step
                    )
                    if grad_norm is not None and torch.isfinite(grad_norm):
                        writer.add_scalar("Gradients/norm", grad_norm.item(), global_step)
                    global_step += 1

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
            final_loss, final_acc = mean_loss, mean_acc

            # Save after the epoch completes; (epoch+1) so we never save an
            # untrained model at epoch 0 and we always save the final epoch.
            if (epoch + 1) % saving_frequency == 0:
                ckpt_path = os.path.join(
                    cfg.train.output.checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt"
                )
                _save_checkpoint(
                    ckpt_path, estimator, optimizer, lr_scheduler, epoch + 1, scaler=scaler
                )

    end_time = time.time()
    logging.info(f"Training completed in {end_time - start_time:.2f} seconds")
    # Final estimator: weights only — used by classifier_utils.load_classifier
    # at inference time. Periodic checkpoints (above) carry the full state.
    torch.save(_underlying_module(estimator).state_dict(), estimator_file)

    # Bind hparams to this run with the final metrics so they share a TB run dir.
    hparams = OmegaConf.to_container(cfg.train, resolve=True)
    flat_hparams = {
        k: str(v) if isinstance(v, (dict, list)) else v for k, v in hparams.items()
    }
    writer.add_hparams(
        flat_hparams,
        metric_dict={"final/loss": final_loss, "final/acc": final_acc},
    )
    writer.close()
