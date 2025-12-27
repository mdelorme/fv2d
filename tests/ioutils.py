import h5py
import numpy as np


def h5_to_numpy(h5file: str, numpyfile: str) -> None:
    with h5py.File(h5file, 'r') as f:
        N = len(f) - 3
        last_ite = f'ite_{N:04d}'
        data = {key: f[last_ite][key][:] for key in f[last_ite].keys()}
        np.savez(numpyfile, **data)
