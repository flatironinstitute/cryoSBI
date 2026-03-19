from typing import Union
import MDAnalysis as mda
from MDAnalysis.analysis import align
import torch


def pdb_parser_(fname: str, atom_selection: str = "name CA") -> torch.tensor:
    """
    Parse a pdb file and return a coarse grained atomic model of the protein.

    The atomic model is a 3xN array, where N is the number of selected atoms.
    The three rows are the x, y, z coordinates.

    Args:
        fname (str): Path to the pdb file.
        atom_selection (str): MDAnalysis atom selection string.

    Returns:
        torch.tensor: Coarse grained atomic model of the protein.
    """

    univ = mda.Universe(fname)
    univ.atoms.translate(-univ.atoms.center_of_mass())

    model = torch.from_numpy(univ.select_atoms(atom_selection).positions.T)

    return model


def pdb_parser(file_formatter, n_pdbs, output_file, start_index=1, **kwargs):
    """
    Parse multiple pdb files and save them as a tensor.

    Args:
        file_formatter (str): Path format for pdb files containing "{}" as index
            placeholder (for example, ``"data/pdb/{}.pdb"``).
        n_pdbs (int): Number of pdb files to parse.
        output_file (str): Path to the output ``.pt`` file.
        start_index (int): Starting index used to format the file names.
        **kwargs: Additional arguments passed to :func:`pdb_parser_`.
    """

    models = pdb_parser_(file_formatter.format(start_index), **kwargs)
    models = torch.zeros((n_pdbs, *models.shape))

    for i in range(0, n_pdbs):
        models[i] = pdb_parser_(file_formatter.format(start_index + i), **kwargs)

    if output_file.endswith("pt"):
        torch.save(models, output_file)

    else:
        raise ValueError("Model file format not supported. Please use .pt.")

    return


def traj_parser_(top_file: str, traj_file: str) -> torch.tensor:
    """
    Parse a trajectory and return coarse grained atomic models.

    The atomic model is an Mx3xN tensor, where M is the number of frames in the
    trajectory and N is the number of residues in the protein.

    Args:
        top_file (str): Path to the topology file.
        traj_file (str): Path to the trajectory file.

    Returns:
        torch.tensor: Coarse grained atomic model of the protein for all frames.
    """

    ref = mda.Universe(top_file)
    ref.atoms.translate(-ref.atoms.center_of_mass())

    mobile = mda.Universe(top_file, traj_file)
    align.AlignTraj(mobile, ref, select="name CA", in_memory=True).run()

    atomic_models = torch.zeros(
        (mobile.trajectory.n_frames, 3, mobile.select_atoms("name CA").n_atoms)
    )

    for i in range(mobile.trajectory.n_frames):
        mobile.trajectory[i]

        atomic_models[i, 0:3, :] = torch.from_numpy(
            mobile.select_atoms("name CA").positions.T
        )

    return atomic_models


def traj_parser(top_file: str, traj_file: str, output_file: str) -> None:
    """
    Parse a trajectory and save atomic models as a tensor.

    Args:
        top_file (str): Path to the topology file.
        traj_file (str): Path to the trajectory file.
        output_file (str): Path to the output ``.pt`` file.
    """

    atomic_models = traj_parser_(top_file, traj_file)

    if output_file.endswith("pt"):
        torch.save(atomic_models, output_file)

    else:
        raise ValueError("Model file format not supported. Please use .pt.")

    return


def models_to_tensor(
    model_files,
    output_file,
    n_pdbs: Union[int, None] = None,
    top_file: Union[str, None] = None,
):
    """
    Converts different model files to a torch tensor.

    Args:
        model_files (str): Model file path or file pattern.
        output_file (str): Path to the output ``.pt`` file.
        n_pdbs (Union[int, None]): Number of pdb files. Required for pdb input.
        top_file (Union[str, None]): Topology file path. Required for trr input.
    """
    assert output_file.endswith("pt"), "The output file must be a .pt file."
    if model_files.endswith("trr"):
        assert top_file is not None, "Please provide a topology file."
        assert n_pdbs is None, "The number of pdb files is not needed for trr files."
        traj_parser(top_file, model_files, output_file)
    elif model_files.endswith("pdb"):
        assert n_pdbs is not None, "Please provide the number of pdb files."
        assert top_file is None, "The topology file is not needed for pdb files."
        pdb_parser(model_files, n_pdbs, output_file)
