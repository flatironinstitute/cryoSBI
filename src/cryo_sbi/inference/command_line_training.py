import argparse
from cryo_sbi.inference.train_npe_model import (
    npe_train_no_saving,
    npe_train_from_vram,
    npe_train_from_disk,
)


def cl_npe_train_no_saving():
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
    cl_parser.add_argument(
        "--n_workers", action="store", type=int, required=False, default=1
    )
    cl_parser.add_argument(
        "--train_device", action="store", type=str, required=False, default="cpu"
    )
    cl_parser.add_argument(
        "--saving_freq", action="store", type=int, required=False, default=20
    )

    args = cl_parser.parse_args()

    npe_train_no_saving(
        image_config=args.image_config_file,
        train_config=args.train_config_file,
        epochs=args.epochs,
        estimator_file=args.estimator_file,
        loss_file=args.loss_file,
        train_from_checkpoint=args.train_from_checkpoint,
        model_state_dict=args.state_dict_file,
        n_workers=args.n_workers,
        device=args.train_device,
        saving_freq=args.saving_freq,
    )


def cl_npe_train_from_vram():
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
        "--train_device", action="store", type=str, required=False, default="cpu"
    )
    cl_parser.add_argument(
        "--saving_freq", action="store", type=int, required=False, default=20
    )
    args = cl_parser.parse_args()

    npe_train_from_vram(
        train_config=args.train_config_file,
        epochs=args.epochs,
        train_data_dir=args.training_data_file,
        val_data_Dir=args.validation_data_file,
        estimator_file=args.estimator_file,
        loss_file=args.loss_file,
        train_from_checkpoint=args.train_from_checkpoint,
        state_dict_file=args.state_dict_file,
        device=args.train_device,
        saving_freq=args.saving_freq,
    )


def cl_npe_train_from_disk():
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
    cl_parser.add_argument(
        "--train_device", action="store", type=str, required=False, default="cpu"
    )
    cl_parser.add_argument(
        "--saving_freq", action="store", type=int, required=False, default=20
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
        device=args.train_device,
        saving_freq=args.saving_freq,
    )