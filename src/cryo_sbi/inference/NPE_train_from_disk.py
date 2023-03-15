import argparse
import json
import torch
import torch.optim as optim
from tqdm import tqdm
from lampe.utils import GDStep
from lampe.data import H5Dataset
from lampe.inference import NPELoss

from cryo_sbi.inference.models.build_models import build_npe_flow_model
from cryo_sbi.inference.validate_train_config import check_train_params


def npe_train_from_disk(
    train_config,
    epochs,
    train_data_dir,
    val_data_dir,
    estimator_file,
    loss_file,
    train_from_checkpoint=False,
    model_state_dict=None,
    n_workers=1,
):
    train_config = json.load(open(train_config))
    check_train_params(train_config)

    estimator = build_npe_flow_model(train_config)

    if train_from_checkpoint:
        if not isinstance(model_state_dict, str):
            raise Warning("No model state dict specified! --model_state_dict is empty")

        print(f"Loading model parameters from {model_state_dict}")
        estimator.load_state_dict(torch.load(model_state_dict))

    estimator.cuda()

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

            tq.set_postfix(
                train_loss=train_losses.mean().item(), val_loss=val_losses.mean().item()
            )

            mean_loss.append(val_losses.mean().item())

    torch.save(estimator, estimator_file)
    torch.save(torch.tensor(mean_loss), loss_file)


if __name__ == "__main__":
    cl_parser = argparse.ArgumentParser()
    cl_parser.add_argument(
        "--train_config_file", action="store", type=str, required=True
    )
    cl_parser.add_argument("--epochs", action="store", type=int, required=True)
    cl_parser.add_argument(
        "--training_data_file", action="store", type=str, required=True
    )
    cl_parser.add_argument(
        "--validation_data_file", action="store", type=str, required=True
    )
    cl_parser.add_argument("--estimator_file", action="store", type=str, required=True)
    cl_parser.add_argument("--loss_file", action="store", type=str, required=True)
    cl_parser.add_argument(
        "--train_from_checkpoint",
        action="store",
        type=bool,
        nargs="?",
        required=False,
        const=True,
        default=False,
    )
    cl_parser.add_argument(
        "--state_dict_file", action="store", type=str, required=False, default=False
    )

    cl_parser.add_argument(
        "--n_workers", action="store", type=int, required=False, default=1
    )
    args = cl_parser.parse_args()

    npe_train_from_disk(
        train_config=args.train_config_file,
        epochs=args.epochs,
        train_data_dir=args.training_data_file,
        val_data_Dir=args.validation_data_file,
        estimator_file=args.estimator_file,
        loss_file=args.loss_file,
        train_from_checkpoint=args.train_from_checkpoint,
        state_dict_file=args.state_dict_file,
        n_workers=args.n_workers,
    )
