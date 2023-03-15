import argparse
import json
import torch
import torch.optim as optim
from tqdm import tqdm
from lampe.data import JointLoader
from lampe.inference import NPELoss
from lampe.utils import GDStep
from itertools import islice

from cryo_sbi.inference.priors import get_uniform_prior_1d
from cryo_sbi.inference.models.build_models import build_npe_flow_model
from cryo_sbi.inference.validate_train_config import check_train_params
from cryo_sbi import CryoEmSimulator


def npe_train_no_saving(
    image_config,
    train_config,
    epochs,
    estimator_file,
    loss_file,
    train_from_checkpoint=False,
    model_state_dict=None,
    n_workers=1
):
    cryo_simulator = CryoEmSimulator(image_config)

    train_config = json.load(open(train_config))
    check_train_params(train_config)
    estimator = build_npe_flow_model(train_config)
    if train_from_checkpoint:
        if not isinstance(model_state_dict, str):
            raise Warning("No model state dict specified! --model_state_dict is empty")
        print(f"Loading model parameters from {model_state_dict}")
        estimator.load_state_dict(torch.load(model_state_dict))
    estimator.cuda()

    loader = JointLoader(
        get_uniform_prior_1d(cryo_simulator.max_index),
        cryo_simulator.simulator,
        vectorized=False,
        batch_size=train_config["BATCH_SIZE"],
        num_workers=n_workers,
        pin_memory=True,
        prefetch_factor=1,
    )

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
                            theta.to(device="cuda", non_blocking=True),
                            x.to(device="cuda", non_blocking=True),
                        )
                    )
                    for theta, x in islice(loader, 1000)
                ]
            )

            tq.set_postfix(loss=losses.mean().item())
            mean_loss.append(losses.mean().item())
            if epoch % 20 == 0:
                torch.save(estimator.state_dict(), estimator_file + f"_epoch={epoch}")

    torch.save(estimator.state_dict(), estimator_file)
    torch.save(torch.tensor(mean_loss), loss_file)


if __name__ == "__main__":
    cl_parser = argparse.ArgumentParser()

    cl_parser.add_argument(
        "--image_config_file", action="store", type=str, required=True
    )
    cl_parser.add_argument(
        "--train_config_file", action="store", type=str, required=True
    )
    cl_parser.add_argument("--epochs", action="store", type=int, required=True)
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
    args = cl_parser.parse_args()

    npe_train_no_saving(
        args.image_config_file,
        args.train_config_file,
        args.epochs,
        args.estimator_file,
        args.loss_file,
        args.train_from_checkpoint,
        args.state_dict_file,
    )
