import MDAnalysis as mda
import torch


def pdb_parser_(fname: str, atom_selection: str = "all") -> torch.tensor:
    """
    Parses a pdb file and returns a coarsed grained atomic model of the protein.
    The atomic model is a 5xN array, where N is the number of residues in the protein.
    The first three rows are the x, y, z coordinates of the alpha carbons.

    Args:
        fname (str): The path to the pdb file.
        atom_selection (str, optional): MDAnalysis selection string for the atoms to keep. Defaults to "all".

    Returns:
        atomic_model (torch.tensor): The coarse grained atomic model of the protein.
    """

    univ = mda.Universe(fname)
    selected = univ.select_atoms(atom_selection)
    selected.translate(-selected.center_of_mass())

    model = torch.from_numpy(selected.positions.T)

    return model


def make_torch_models(pdb_files: list[str], output_file: str, atom_selection: str = "all") -> None:
    """
    Converts a list of pdb files to a single torch tensor and saves it to disk.

    Args:
        pdb_files (list[str]): List of paths to pdb files.
        output_file (str): Path to save the torch tensor.
        atom_selection (str, optional): MDAnalysis selection string passed to ``pdb_parser_``. Defaults to "all".
    """

    models = []
    max_num_atoms = 0
    for pdb_file in pdb_files:
        model = pdb_parser_(pdb_file, atom_selection=atom_selection)
        models.append(model)
        if model.shape[1] > max_num_atoms:
            max_num_atoms = model.shape[1]

    models_torch = torch.full((len(models), 3, max_num_atoms), torch.inf)
    for i, model in enumerate(models):
        models_torch[i, :, : model.shape[1]] = model

    torch.save(models_torch, output_file)
