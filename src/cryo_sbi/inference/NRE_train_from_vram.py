import argparse
import json
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset
from tqdm import tqdm
from lampe.inference import NRELoss
from lampe.utils import GDStep

from cryo_sbi.inference.models.build_models import build_nre_classifier_model
from cryo_sbi.inference.validate_train_config import check_train_params


def nre_train_from_vram(
    train_config,
    train_data_dir,
    val_data_dir,
    epochs,
    estimator_file,
    loss_file,
    train_from_checkpoint=False,
    model_state_dict=None,
):
    train_config = json.load(open(train_config))
    check_train_params(train_config)
    estimator = build_nre_classifier_model(train_config)
    if train_from_checkpoint:
        if not isinstance(model_state_dict, str):
            raise Warning("No model state dict specified! --model_state_dict is empty")
        print(f"Loading model parameters from {model_state_dict}")
        estimator.load_state_dict(torch.load(model_state_dict))
    estimator.cuda()

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
        num_workers=0,
        pin_memory=False,
        prefetch_factor=2,
        shuffle=True,
    )

    val_loader = torch.utils.data.DataLoader(
        validset,
        batch_size=train_config["BATCH_SIZE"],
        num_workers=0,
        pin_memory=False,
        prefetch_factor=2,
        shuffle=True,
    )

    loss = NRELoss(estimator)
    optimizer = optim.AdamW(estimator.parameters(), lr=train_config["LEARNING_RATE"])
    step = GDStep(optimizer, clip=train_config["CLIP_GRADIENT"])
    mean_val_loss = []
    mean_train_loss = []

    print("Training neural netowrk:")
    estimator.train()
    with tqdm(range(epochs), unit="epoch") as tq:
        for epoch in tq:
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
            if epoch % 100 == 0:
                torch.save(estimator.state_dict(), estimator_file + f"_epoch={epoch}")

    torch.save(estimator.state_dict(), estimator_file)
    torch.save(torch.tensor((mean_train_loss, mean_val_loss)), loss_file)


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
        "--n_workers", action="store", type=int, required=False, default=0
    )
    args = cl_parser.parse_args()

    nre_train_from_vram(
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
