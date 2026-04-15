import argparse
from cryo_sbi.utils.generate_models import make_torch_models


def main():
    parser = argparse.ArgumentParser(description="Convert PDB files to a torch model tensor.")
    parser.add_argument("--pdb_files", required=True, help="Comma-separated PDB file paths.")
    parser.add_argument("--output_file", required=True, help="Output .pt file path.")
    parser.add_argument("--atom_selection", default="all", help="MDAnalysis atom selection string.")
    args = parser.parse_args()
    make_torch_models(
        pdb_files=args.pdb_files.split(","),
        output_file=args.output_file,
        atom_selection=args.atom_selection,
    )


if __name__ == "__main__":
    main()
