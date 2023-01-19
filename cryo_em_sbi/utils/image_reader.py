import numpy as np
import torch
import pandas as pd
import mrcfile
import os
from aspire.storage import StarFile

class ImageReader():

    """
    This class is based on ASPIRE's RelionSource class.
    https://github.com/ComputationalCryoEM/ASPIRE-Python/blob/master/src/aspire/source/relion.py
    """
    
    relion_metadata_fields = {
        "_rlnVoltage": float,
        "_rlnDefocusU": float,
        "_rlnDefocusV": float,
        "_rlnDefocusAngle": float,
        "_rlnSphericalAberration": float,
        "_rlnDetectorPixelSize": float,
        "_rlnCtfFigureOfMerit": float,
        "_rlnMagnification": float,
        "_rlnAmplitudeContrast": float,
        "_rlnImageName": str,
        "_rlnOriginalName": str,
        "_rlnCtfImage": str,
        "_rlnCoordinateX": float,
        "_rlnCoordinateY": float,
        "_rlnCoordinateZ": float,
        "_rlnNormCorrection": float,
        "_rlnMicrographName": str,
        "_rlnGroupName": str,
        "_rlnGroupNumber": str,
        "_rlnOriginX": float,
        "_rlnOriginY": float,
        "_rlnAngleRot": float,
        "_rlnAngleTilt": float,
        "_rlnAnglePsi": float,
        "_rlnClassNumber": int,
        "_rlnLogLikeliContribution": float,
        "_rlnRandomSubset": int,
        "_rlnParticleName": str,
        "_rlnOriginalParticleName": str,
        "_rlnNrOfSignificantSamples": float,
        "_rlnNrOfFrames": int,
        "_rlnMaxValueProbDistribution": float,
    }
    def __init__(self, filepath):
        
        self.starfile = self._parse_star_file(filepath)

    def _parse_star_file(self, filepath, data_folder=None):

        starfile = StarFile(filepath).get_block_by_index(0)

        column_types = {
            name: ImageReader.relion_metadata_fields.get(name, str)
            for name in starfile.columns
        }

        starfile = starfile.astype(column_types)

        if data_folder is not None:
            if not os.path.isabs(data_folder):
                data_folder = os.path.join(os.path.dirname(filepath), data_folder)
        else:
            data_folder = os.path.dirname(filepath)

        starfile[["__mrc_index", "__mrc_filename"]] = starfile["_rlnImageName"].str.split(
            "@", 1, expand=True
        )
        # __mrc_index corresponds to the integer index of the particle in the __mrc_filename stack
        # Note that this is 1-based indexing
        starfile["__mrc_index"] = pd.to_numeric(starfile["__mrc_index"])

        # Adding a full-filepath field to the Dataframe helps us save time later
        # Note that os.path.join works as expected when the second argument is an absolute path itself
        starfile["__mrc_filepath"] = starfile["__mrc_filename"].apply(
            lambda filename: os.path.join(data_folder, filename)
        )

        return starfile

    def read_images(self, indices):

        indices = np.asanyarray(indices)

        first_mrc_filepath = self.starfile.loc[0]["__mrc_filepath"]
        mrc = mrcfile.open(first_mrc_filepath)

        # Get the 'mode' (data type) - TODO: There's probably a more direct way to do this.
        mode = int(mrc.header.mode)
        dtypes = {0: "int8", 1: "int16", 2: "float32", 6: "uint16"}
        assert (
            mode in dtypes
        ), f"Only modes={list(dtypes.keys())} in MRC files are supported for now."

        dtype = dtypes[mode]

        shape = mrc.data.shape

        indices = np.array([0, 1, 2])

        images = torch.zeros((indices.shape[0], shape[0] * shape[1]))

        for i in range(indices.shape[0]):

            mrc_filepath = self.starfile.loc[i]["__mrc_filepath"]
            mrc = mrcfile.open(mrc_filepath)

            images[i] = torch.tensor(mrc.data.flatten())

        return images
        

def main():

    filepath = "/path/to/star_file.star"
    image_reader = ImageReader(filepath)

    indices = np.array([0, 1, 2, 3, 4]) # reads the first five images
    images = image_reader.read_images(indices) # shape (5, n_pixels**2)