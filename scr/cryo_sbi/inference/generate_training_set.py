import argparse
import torch
from itertools import islice
from lampe.data import JointLoader, H5Dataset
from priors import get_unirom_prior_1d
from tqdm import tqdm

import sys
sys.path.insert(0, "../wpa_simulator")
from cryo_em_simulator import CryoEmSimulator


def main(config_file, num_train_samples, num_val_samples, file_name, save_as_tensor):
    
    cryo_simulator = CryoEmSimulator(config_file)
    loader = JointLoader(
        get_unirom_prior_1d(cryo_simulator.get_max_index()),
        cryo_simulator.simulator,
        vectorized=False,
        batch_size=1000,
        num_workers=24,
        prefetch_factor=1
    )

    if save_as_tensor:
        torch.multiprocessing.set_sharing_strategy('file_system')
        train_data = islice(loader, num_train_samples)
        thetas = []
        xs = []
        with tqdm(range(num_train_samples), unit='batch') as tq:
            for theta, x in train_data:
                thetas.append(theta.to(dtype=torch.float))
                xs.append(x.to(dtype=torch.float))
                del theta, x
                tq.update(1)
        train_theta = torch.cat(thetas, dim=0)
        torch.save(train_theta, f"{file_name}_theta_train.pt")
        train_x = torch.cat(xs, dim=0)
        torch.save(train_x, f"{file_name}_x_train.pt")
        del train_data, train_x, train_theta, thetas, xs

        val_data = islice(loader, num_val_samples)
        thetas = []
        xs = []
        with tqdm(range(num_val_samples), unit='batch') as tq:
            for theta, x in val_data:
                thetas.append(theta.to(dtype=torch.float))
                xs.append(x.to(dtype=torch.float))
                del theta, x
                tq.update(1)
        val_theta = torch.cat(thetas, dim=0)
        torch.save(val_theta, f"{file_name}_theta_val.pt")
        val_x = torch.cat(xs, dim=0)
        torch.save(val_x, f"{file_name}_x_val.pt")
        del val_data, val_x, val_theta, thetas, xs

    else:
        H5Dataset.store(
            loader, 
            f'{file_name}_train.h5',
            size=num_train_samples,
            overwrite=True
        )
        H5Dataset.store(
            loader,
            f'{file_name}_valid.h5',
            size=num_val_samples,
            overwrite=True
        )


if __name__ == '__main__':
    cl_parser = argparse.ArgumentParser()
    
    cl_parser.add_argument(
        '--config_file',
        action='store',
        type=str,
        required=True
    )

    cl_parser.add_argument(
        '--num_train_samples',
        action='store',
        type=int,
        required=True
    )

    cl_parser.add_argument(
        '--num_val_samples',
        action='store',
        type=int,
        required=True
    )

    cl_parser.add_argument(
        '--file_name',
        action='store',
        type=str,
        required=True
    )

    cl_parser.add_argument(
        '--save_as_tensor',
        action='store',
        type=bool,
        nargs='?',
        required=True,
        const=True,
        default=False
        )
    args = cl_parser.parse_args()
    main(args.config_file, args.num_train_samples, args.num_val_samples, args.file_name, args.save_as_tensor)