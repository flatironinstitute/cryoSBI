import argparse
from cryo_sbi.utils.generate_models import make_torch_models
from cryo_sbi.inference.inference import classifier_inference


def cl_make_torch_models():
    cl_parser = argparse.ArgumentParser()
    cl_parser.add_argument("--pdb_files", type=str, required=True)
    cl_parser.add_argument("--output_file", type=str, required=True)
    cl_parser.add_argument(
        "--atom_selection", type=str, required=False, default="all"
    )
    args = cl_parser.parse_args()
    pdb_files = args.pdb_files.split(",")
    make_torch_models(
        pdb_files=pdb_files,
        save_path=args.output_file,
        atom_selection=args.atom_selection,
    )


def cl_inference():
    cl_parser = argparse.ArgumentParser(
        description="Run classifier inference on cryo-EM MRC files."
    )
    cl_parser.add_argument(
        "--folder_with_mrcs", type=str, required=True,
        help="Path to folder containing .mrc files."
    )
    cl_parser.add_argument(
        "--estimator_weights", type=str, required=True,
        help="Path to trained model weights (.pt)."
    )
    cl_parser.add_argument(
        "--config", type=str, required=True,
        help="Path to training config file (YAML or JSON). Model architecture and all"
             " hyperparameters are read from here."
    )
    cl_parser.add_argument(
        "--file_name", type=str, required=True,
        help="Base filename for output tensors (likelihoods_<file_name>.pt, embeddings_<file_name>.pt)."
    )
    cl_parser.add_argument(
        "--num_workers", type=int, default=2,
        help="Number of data-loading workers (default: 2)."
    )
    cl_parser.add_argument(
        "--output_dir", type=str, default=".",
        help="Directory to write output files (default: current directory)."
    )
    cl_parser.add_argument(
        "--image_size", type=int, default=128,
        help="Image size in pixels (default: 128)."
    )
    cl_parser.add_argument(
        "--prefetch_factor", type=int, default=2,
        help="Prefetch factor for data loading (default: 2)."
    )
    cl_parser.add_argument(
        "--max_batch_size", type=int, default=32,
        help="Max batch size during inference (default: 32)."
    )
    cl_parser.add_argument(
        "--no_whitening", action="store_true", default=False,
        help="Disable image whitening (whitening is on by default)."
    )
    args = cl_parser.parse_args()

    classifier_inference(
        folder_with_mrcs=args.folder_with_mrcs,
        estimator_weights=args.estimator_weights,
        estimator_config=args.config,
        file_name=args.file_name,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
        image_size=args.image_size,
        prefetch_factor=args.prefetch_factor,
        max_batch_size=args.max_batch_size,
        whitening=not args.no_whitening,
    )
