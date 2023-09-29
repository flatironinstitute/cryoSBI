import MDAnalysis as mda
import torch


def pdb_parser_all_atom_(fname: str) -> torch.tensor:
    """
    Parses a pdb file and returns an atomic model of the protein. The atomic model is a 5xN array, where N is the number of atoms in the protein. The first three rows are the x, y, z coordinates of the atoms. The fourth row is the atomic number of the atoms. The fifth row is the variance of the atoms before the resolution is applied.
    Parameters
    ----------
    fname : str
        The path to the pdb file.

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

    univ = mda.Universe(fname)
    univ.atoms.translate(-univ.atoms.center_of_mass())

    atomic_model = torch.zeros((5, univ.select_atoms("not name H*").n_atoms))

    atomic_model[0:3, :] = torch.from_numpy(
        univ.select_atoms("not name H*").positions.T
    )
    atomic_model[3, :] = torch.tensor(
        [atomic_numbers[x] for x in univ.select_atoms("not name H*").elements]
    )

    atomic_model[4, :] = (1 / torch.pi) ** 2

    return atomic_model


def pdb_parser_resid_(fname: str) -> torch.tensor:
    """
    Parses a pdb file and returns a coarsed grained atomic model of the protein. The atomic model is a 5xN array, where N is the number of residues in the protein. The first three rows are the x, y, z coordinates of the alpha carbons. The fourth row is the density of the residues, i.e., the total number of electrons. The fifth row is the radius of the residues squared, which we use as the variance of the residues for the forward model.

    Parameters
    ----------
    fname : str
        The path to the pdb file.

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
        "CGB": 4.5,
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
        "CGB": 36.0,
    }

    univ = mda.Universe(fname)
    residues = univ.residues
    univ.atoms.translate(-univ.atoms.center_of_mass())

    atomic_model = torch.zeros((5, residues.n_residues))
    atomic_model[0:3, :] = torch.from_numpy(univ.select_atoms("name CA").positions.T)
    atomic_model[3, :] = torch.tensor([resid_density[x] for x in residues.resnames])
    atomic_model[4, :] = (
        torch.tensor([resid_radius[x] for x in residues.resnames]) / torch.pi
    ) ** 2

    return atomic_model


def pdb_parser_(input_file: str, mode: str) -> torch.tensor:
    """
    Parses a pdb file and returns an atomic model of the protein. The atomic model is a 5xN array, where N is the number of atoms or residues in the protein. The first three rows are the x, y, z coordinates of the atoms or residues. The fourth row is the atomic number of the atoms or the density of the residues. The fifth row is the variance of the atoms or residues, which is the resolution of the cryo-EM map divided by pi squared.

    Parameters
    ----------
    input_file : str
        The path to the pdb file.
    mode : str
        The mode of the atomic model. Either "resid" or "all-atom". Resid mode returns a coarse grained atomic model of the protein. All atom mode returns an all atom atomic model of the protein.

    Returns
    -------
    atomic_model : torch.tensor
        The atomic model of the protein.
    """

    if mode == "resid":
        atomic_model = pdb_parser_resid_(input_file)

    elif mode == "all-atom":
        atomic_model = pdb_parser_all_atom_(input_file)

    else:
        raise ValueError("Mode must be either 'resid' or 'all-atom'.")

    return atomic_model


def pdb_parser(input_file_prefix, n_pdbs, output_file, mode):
    """
    Parses a pdb file and returns an atomic model of the protein. The atomic model is a 5xN array, where N is the number of atoms or residues in the protein. The first three rows are the x, y, z coordinates of the atoms or residues. The fourth row is the atomic number of the atoms or the density of the residues. The fifth row is the variance of the atoms or residues, which is the resolution of the cryo-EM map divided by pi squared.

    Parameters
    ----------
    input_file_prefix : str
        The path to the pdb file. The pdb files should be named as input_file_prefix0.pdb, input_file_prefix1.pdb, etc.
    n_pdbs : int
        The number of pdb files to parse.
    output_file : str
        The path to the output file. The output file must be a .pt file.
    mode : str
        The mode of the atomic model. Either "resid" or "all atom". Resid mode returns a coarse grained atomic model of the protein. All atom mode returns an all atom atomic model of the protein.
    """

    atomic_model = pdb_parser_(f"{input_file_prefix}{0}.pdb", mode)
    atomic_models = torch.zeros((n_pdbs, *atomic_model.shape))

    for i in range(n_pdbs):
        atomic_models[i] = pdb_parser_(f"{input_file_prefix}{i}.pdb", mode)

    if output_file.endswith("pt"):
        torch.save(atomic_models, output_file)

    else:
        raise ValueError("Model file format not supported. Please use .pt.")

    return
