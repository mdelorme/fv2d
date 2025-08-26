import h5py
import numpy as np
import glob
import os
from core.fields import compute_values


# class MetaData:
#     """ Class to hold the metadata of the simulation."""
#     def __init__(self, file: str) -> None:
#         self.file = file
#         self.metadata = self._extract_metadata()
#         self.Nx = self.metadata['Nx']
#         self.Ny = self.metadata['Ny']
#         self.x  = self.metadata['x']
#         self.y  = self.metadata['y']
#         self.ext = self.metadata['ext']
#         self.problem = self.metadata['problem']
    
#     def _extract_metadata(self) -> dict:
#         with h5py.File(self.file, 'r') as f:
#             Nx = f.attrs['Nx']
#             Ny = f.attrs['Ny']
#             x = np.array(f['x'])
#             y = np.array(f['y'])
#             xmin, xmax = x.min(), x.max()
#             ymin, ymax = y.min(), y.max()
#             dx, dy = x[1] - x[0], y[1] - y[0]
#             ext = [xmin - 0.5 * dx, xmax + 0.5 * dx, ymin - 0.5 * dy, ymax + 0.5 * dy]
#             return {
#                 'Nx': Nx,
#                 'Ny': Ny,
#                 'x': x,
#                 'y': y,
#                 'ext': ext,
#                 'problem': f.attrs['problem'].title()
#             }

# TODO: Ajouter la possibilité de tracer des fonctions customisées (comme dans fv2d utils)
class Fv2dData:
    """
    Class to access simulation data stored in HDF5 files in a unified manner :
    Allows user to interact with a unique file with multiple timesteps or multiple
    files with one timestep.

    Args:
        file_pattern (str or list): Path to .h5 file or glob pattern (ex: run_*.h5).
    """
    def __init__(self, file_pattern) -> None:
        self.files = self._get_simulation_files(file_pattern)
        self.metadata = self._get_metadata()
        self.is_multi_iteration = self._check_multi_iteration()
        self.metadata = self._get_metadata()
        self.Nx = self.metadata['Nx']
        self.Ny = self.metadata['Ny']
        self.x  = self.metadata['x']
        self.y  = self.metadata['y']
        self.ext = self.metadata['ext']
        self.problem = self.metadata['problem']
        
    def _check_multi_iteration(self) -> bool:
        """
        Check if the data is stored in multiple files or a single file with multiple iterations.
        """
        if len(self.files) == 1:
            with h5py.File(self.files[0], 'r') as f:
                return 'ite_0000' in f
        else:
            return False
    
    # Pas besoin d'utiliser glob car déjà pris en charge par argparse. On aura dans tous les cas une liste de fichiers.
    def _get_simulation_files(self, file_pattern) -> list:
        """
        Returns the list of files to handle.
        """
        if isinstance(file_pattern, str): # should not happen. Add an assert?
            assert False, "Unexpected behaviour, misuse of the class FV2dData"
            file_pattern = [file_pattern]
        if len(file_pattern) == 1 and self._check_multi_iteration_for_file(file_pattern[0]):
            return file_pattern # Case 1: one file with multiple iterations
        else:
            # Case 2: Several mono-iteration files
            files = sorted(file_pattern)
            if not files: # should be handled differently
                raise FileNotFoundError(f"No files found matching pattern: {file_pattern[0]}")
            return files
    
    def _check_multi_iteration_for_file(self, file_path) -> bool:
        """
        Check if a single file contains multiple iterations.
        """
        with h5py.File(file_path, 'r') as f:
            return 'ite_0000' in f
    
    def _get_metadata(self) -> dict:
        """
        Extract metadata from the first file.
        """
        with h5py.File(self.files[0], 'r') as f:
            Nx = f.attrs['Nx']
            Ny = f.attrs['Ny']
            x = np.array(f['x'])
            y = np.array(f['y'])
            xmin, xmax = x.min(), x.max()
            ymin, ymax = y.min(), y.max()
            dx, dy = x[1] - x[0], y[1] - y[0]
            ext = [xmin - 0.5 * dx, xmax + 0.5 * dx, ymin - 0.5 * dy, ymax + 0.5 * dy]
            return {
                'Nx': Nx,
                'Ny': Ny,
                'x': x,
                'y': y,
                'ext': ext,
                'problem': f.attrs['problem'].title()
            }
    
    def __len__(self) -> int:
        """
        Returns the number of iterations available.
        """
        if self.is_multi_iteration:
            with h5py.File(self.files[0], 'r') as f:
                return len(f) - 2 # Iterations are ite_0000, ite_0001... minus x and y
        else:
            return len(self.files) # One file = one iteration
    
    def __getitem__(self, i):
        """
        Allows the user to access the data with `data[i]`.
        Returns a dict with all available variables.
        """
        if self.is_multi_iteration:
            with h5py.File(self.files[0], 'r') as f:
                return self._get_iteration_data(f, i)
        else:
            with h5py.File(self.files[i], 'r') as f:
                return self._get_iteration_data(f, i)
    
    def _get_iteration_data(self, f, i):
        data = {}
        if self.is_multi_iteration:
            group = f[f'ite_{i:04d}']
            for var in group:
                data[var] = np.array(group[var]).reshape(self.Ny, self.Nx)
        else:
            for var in f:
                if var not in ['x', 'y']:  # x and y already in metadata
                    data[var] = np.array(f[var]).reshape(self.Ny, self.Nx)
        # Add custom values
        for field, fn in compute_values.items():
            data[field] = fn(data, self.metadata)
        return data
    
    def get_time(self, i):
        """Returns time associated for a given iteration."""
        if self.is_multi_iteration:
            with h5py.File(self.files[0], 'r') as f:
                return f[f'ite_{i:04d}'].attrs['time']
        else:
            with h5py.File(self.files[i], 'r') as f:
                return f.attrs['time']

def get_slice_data(filename, field, xslice=None, yslice=None):
    """Retourne les données pour une slice (x ou y)."""
    metadata = MetaData(filename)
    Nx, Ny = metadata.Nx, metadata.Ny
    x, y = metadata.x, metadata.y

    if yslice is not None:
        y_reshaped = y.reshape((Nx + 1, Ny + 1))
        idxY = np.abs(y_reshaped[0, :] - yslice).argmin()
        return {'x': x[:Nx], 'idx': idxY, 'metadata': metadata}
    elif xslice is not None:
        x_reshaped = x.reshape((Nx + 1, Ny + 1))
        idxX = np.abs(x_reshaped[:, 0] - xslice).argmin()
        return {'y': y[:Ny], 'idx': idxX, 'metadata': metadata}
    else:
        raise ValueError("Soit xslice, soit yslice doit être spécifié.")
