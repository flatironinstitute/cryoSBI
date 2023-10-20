import MDAnalysis as mda
from MDAnalysis.analysis import align
import torch


def pdb_parser_(fname: str) -> torch.tensor:
    """
    Parses a pdb file and returns a coarsed grained atomic model of the protein.
    The atomic model is a 5xN array, where N is the number of residues in the protein.
    The first three rows are the x, y, z coordinates of the alpha carbons.
    The fourth row is the density of the residues, i.e., the total number of electrons.
    The fifth row is the radius of the residues squared, which we use as the variance of the residues for the forward model.

    Parameters
    ----------
    fname : str
        The path to the pdb file.

    Returns
    -------
    atomic_model : torch.tensor
        The coarse grained atomic model of the protein.
    """


    univ = mda.Universe(fname)
    univ.atoms.translate(-univ.atoms.center_of_mass())

    atomic_model = torch.from_numpy(univ.select_atoms("name CA").positions.T)
   
    return atomic_model


def pdb_parser(file_formatter, n_pdbs, output_file, start_index=1):
    """
    Parses a pdb file and returns an atomic model of the protein. The atomic model is a 5xN array, where N is the number of atoms or residues in the protein. The first three rows are the x, y, z coordinates of the atoms or residues. The fourth row is the atomic number of the atoms or the density of the residues. The fifth row is the variance of the atoms or residues, which is the resolution of the cryo-EM map divided by pi squared.

    Parameters
    ----------
    file_formatter : str
        The path to the pdb file. The path must contain the placeholder {} for the pdb index. For example, if the path is "data/pdb/{}.pdb", then the placeholder is {}.
    n_pdbs : int
        The number of pdb files to parse.
    output_file : str
        The path to the output file. The output file must be a .pt file.
    mode : str
        The mode of the atomic model. Either "resid" or "all atom". Resid mode returns a coarse grained atomic model of the protein. All atom mode returns an all atom atomic model of the protein.
    """

    atomic_model = pdb_parser_(file_formatter.format(start_index))
    atomic_models = torch.zeros((n_pdbs, *atomic_model.shape))

    for i in range(0, n_pdbs):
        atomic_models[i] = pdb_parser_(file_formatter.format(start_index+i))


    if output_file.endswith("pt"):
        torch.save(atomic_models, output_file)

    else:
        raise ValueError("Model file format not supported. Please use .pt.")

    return


def traj_parser_(top_file: str, traj_file: str) -> torch.tensor:
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

    atomic_models = traj_parser_(top_file, traj_file)

    if output_file.endswith("pt"):
        torch.save(atomic_models, output_file)

    else:
        raise ValueError("Model file format not supported. Please use .pt.")

    return