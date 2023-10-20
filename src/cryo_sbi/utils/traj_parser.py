import MDAnalysis as mda
from MDAnalysis.analysis import align
import torch


def traj_parser_all_atom_(top_file: str, traj_file: str) -> torch.tensor:
    """
    Parses a traj file and returns an atomic model of the protein. The atomic model is a Mx5xN array, where M is the number of frames in the trajectory, and N is the number of atoms in the protein. The first three rows in axis 1 are the x, y, z coordinates of the atoms. The fourth row is the atomic number of the atoms. The fifth row is the variance of the atoms before the resolution is applied.
    Parameters
    ----------
    top_file : str
        The path to the top file.
    traj_file : str
        The path to the traj file.

    Returns
    -------
    atomic_model : torch.tensor
        The atomic model of the protein.

    """

    atomic_numbers = {
        "C": 6.0,
        "A": 7.0,
        "N": 7.0,
        "O": 8.0,
        "P": 15.0,
        "K": 19.0,
        "S": 16.0,
        "AU": 79.0,
    }

    ref = mda.Universe(top_file)
    ref.atoms.translate(-ref.atoms.center_of_mass())
    mobile = mda.Universe(top_file, traj_file)
    align.AlignTraj(mobile, ref, select="name CA", in_memory=True).run()

    atomic_models = torch.zeros(
        (mobile.trajectory.n_frames, 5, mobile.select_atoms("not name H*").n_atoms)
    )

    for i in range(mobile.trajectory.n_frames):
        mobile.trajectory[i]

        atomic_models[i, 0:3, :] = torch.from_numpy(
            mobile.select_atoms("not name H*").positions.T
        )
        atomic_models[i, 3, :] = torch.tensor(
            [atomic_numbers[x] for x in mobile.select_atoms("not name H*").elements]
        )

        atomic_models[i, 4, :] = (1 / torch.pi) ** 2

    return atomic_models


def traj_parser_resid_(top_file: str, traj_file: str) -> torch.tensor:
    """
    Parses a traj file and returns a coarsed grained atomic model of the protein. The atomic model is a Mx5xN array, where M is the number of frames in the trajectory, and N is the number of residues in the protein. The first three rows in axis 1 are the x, y, z coordinates of the alpha carbons. The fourth row is the density of the residues, i.e., the total number of electrons. The fifth row is the radius of the residues squared, which we use as the variance of the residues for the forward model.

    Parameters
    ----------
    top_file : str
        The path to the traj file.

    Returns
    -------
    atomic_model : torch.tensor
        The coarse grained atomic model of the protein.
    """

    resid_radius = {
        "CYS": 2.75,
        "PHE": 3.2,
        "LEU": 3.1,
        "TRP": 3.4,
        "VAL": 2.95,
        "ILE": 3.1,
        "MET": 3.1,
        "HIS": 3.05,
        "TYR": 3.25,
        "ALA": 2.5,
        "GLY": 2.25,
        "PRO": 2.8,
        "ASN": 2.85,
        "THR": 2.8,
        "SER": 2.6,
        "ARG": 3.3,
        "GLN": 3.0,
        "ASP": 2.8,
        "LYS": 3.2,
        "GLU": 2.95,
    }

    resid_density = {
        "CYS": 64.0,
        "PHE": 88.0,
        "LEU": 72.0,
        "TRP": 108.0,
        "VAL": 64.0,
        "ILE": 72.0,
        "MET": 80.0,
        "HIS": 82.0,
        "TYR": 96.0,
        "ALA": 48.0,
        "GLY": 40.0,
        "PRO": 62.0,
        "ASN": 66.0,
        "THR": 64.0,
        "SER": 56.0,
        "ARG": 93.0,
        "GLN": 78.0,
        "ASP": 59.0,
        "LYS": 79.0,
        "GLU": 53.0,
    }

    ref = mda.Universe(top_file)
    residues = ref.residues
    ref.atoms.translate(-ref.atoms.center_of_mass())

    mobile = mda.Universe(top_file, traj_file)
    align.AlignTraj(mobile, ref, select="name CA", in_memory=True).run()

    atomic_models = torch.zeros(
        (mobile.trajectory.n_frames, 5, mobile.select_atoms("name CA").n_atoms)
    )

    for i in range(mobile.trajectory.n_frames):
        mobile.trajectory[i]

        atomic_models[i, 0:3, :] = torch.from_numpy(
            mobile.select_atoms("name CA").positions.T
        )

        atomic_models[i, 3, :] = torch.tensor(
            [resid_density[x] for x in residues.resnames]
        )
        atomic_models[i, 4, :] = 2 * (torch.tensor([resid_radius[x] for x in residues.resnames]) ** 2) # Residue radius is will be the 2 sigma interval of the gaussian

    return atomic_models


def traj_parser(top_file: str, traj_file: str, output_file: str, mode: str) -> None:
    """
    Parses a traj file and returns an atomic model of the protein. The atomic model is a Mx5xN array, where M is the number of frames in the trajectory, and N is the number of atoms in the protein. The first three rows in axis 1 are the x, y, z coordinates of the atoms. The fourth row is the atomic number of the atoms. The fifth row is the variance of the atoms before the resolution is applied.

    Parameters
    ----------
    top_file : str
        The path to the topology file.
    traj_file : str
        The path to the trajectory file.
    output_file : str
        The path to the output file. Must be a .pt file.
    mode : str
        The mode of the atomic model. Either "resid" or "all-atom". Resid mode returns a coarse grained atomic model of the protein. All atom mode returns an all atom atomic model of the protein.

    Returns
    -------
    None
    """

    if mode == "resid":
        atomic_models = traj_parser_resid_(top_file, traj_file)

    elif mode == "all-atom":
        atomic_models = traj_parser_all_atom_(top_file, traj_file)

    else:
        raise ValueError("Mode must be either 'resid' or 'all-atom'.")

    if output_file.endswith("pt"):
        torch.save(atomic_models, output_file)

    else:
        raise ValueError("Model file format not supported. Please use .pt.")

    return
