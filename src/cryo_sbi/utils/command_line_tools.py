import argparse
from cryo_sbi.utils.generate_models import models_to_tensor


def cl_models_to_tensor():
    cl_parser = argparse.ArgumentParser(
        description="Convert models to tensor for cryoSBI"
        epilog="pdb-files: The name for the pdbs must contain a {} to be replaced by the index of the pdb file. The index starts at 0. \
        For example protein_{}.pdb. trr-files: For .trr files you must provide a topology file.")
    cl_parser.add_argument(
        "--model_files", action="store", type=str, required=True
    )
    cl_parser.add_argument(
        "--output_file", action="store", type=str, required=True
    )
    cl_parser.add_argument(
        "--n_pdbs", action="store", type=int, required=False, default=None
    )
    cl_parser.add_argument(
        "--top_file", action="store", type=str, required=False, default=None
    )
    args = cl_parser.parse_args()
    models_to_tensor(
        model_files=args.model_files,
        output_file=args.output_file,
        n_pdbs=args.n_pdbs,
        top_file=args.top_file
    )