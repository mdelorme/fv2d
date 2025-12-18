import os
import shutil
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import sys
from fv2d_utils import latexify

# Suppression et création du répertoire de rendu
if os.path.exists('render'):
    shutil.rmtree('render')
os.mkdir('render')

# Paramètres
show_grid = False
field = 'rho'
if '--field' in sys.argv:
    i = sys.argv.index('--field')
    field = sys.argv[i+1]
solver = ''
if '--solver' in sys.argv:
    i = sys.argv.index('--solver')
    solvers = [sys.argv[i+1]]
if '--file' in sys.argv:
    i = sys.argv.index('--file')
    if i+1 >= len(sys.argv):
        print('[ERROR] Please provide a filename after --file')
        sys.exit(1)
    filenames = [sys.argv[i+1]]
elif '--files' in sys.argv:
    i = sys.argv.index('--files')
    if i+1 >= len(sys.argv):
        print('[ERROR] Please provide a list of filenames after --files')
        sys.exit(1)
    if "--solvers" not in sys.argv:
        print('[ERROR] Please provide a list of solvers after --solvers for each file.')
        sys.exit(1)

    filenames = sys.argv[i+1].split(',')
    solvers = sys.argv[sys.argv.index('--solvers') + 1].split(',')
    if not all(os.path.exists(fn) for fn in filenames):
        print(f'[ERROR] One or more files in {filenames} do not exist.')
        sys.exit(1)
else:
    filenames = ['run.h5']
    solvers = ['']

with h5py.File(filenames[0], 'r') as f:
    # on suppose que tous les fichiers ont le même pas de temps
    Nf = len(f) - 2
    is_mhd = True if 'bx' in f['ite_0000'] else False
    Nx = f.attrs['Nx']
    Ny = f.attrs['Ny']
    problem = f.attrs['problem'].title()
    x = np.array(f['x'])
    y = np.array(f['y'])
    Nx = f.attrs['Nx']
    Ny = f.attrs['Ny']
    xmin = x.min()
    xmax = x.max()
    ymin = y.min()
    ymax = y.max()
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    ext = [xmin - 0.5*dx, xmax + 0.5*dx, ymin - 0.5*dy, ymax + 0.5*dy]

print(f'Rendering animation for files: {filenames}')
# Fonction pour tracer les champs
def plot_field(field, cax, i, filename, solver):
    with h5py.File(filename, 'r') as f:
        path = f'ite_{i:04d}/{field}'
        Nx = f.attrs['Nx']
        Ny = f.attrs['Ny']
        arr = np.array(f[path]).reshape((Ny, Nx))
        x_slice = x[:Nx]
        legend = latexify[field]
        y_slice = arr[Ny//2, :]  # For 1D animation, we take a slice at the middle of the y-axis
        cax.plot(x_slice, y_slice, '--', label=solver)
        cax.set_title(legend)
        # cax.legend()
        if show_grid:
            cax.set_xticks(np.arange(ext[0], ext[1], dx), minor=True)
            cax.set_yticks(np.arange(ext[2], ext[3], dy), minor=True)
            cax.grid(which='minor', color='k', linestyle='-', linewidth=1)
            cax.tick_params(which='minor', bottom=False, left=False)

for i in tqdm(range(Nf)):
    fig, ax = plt.subplots(3, 3, figsize=(12, 12)) if is_mhd else plt.subplots(2, 2, figsize=(10, 10))
    for filename, solver in zip(filenames, solvers):
        with h5py.File(filename, 'r') as f:
            t = f[f'ite_{i:04d}'].attrs['time']
            plt.suptitle(f'{problem} - Time: {t:.3f} - $N_x={Nx}$')
            plot_field('rho', ax[0, 0], i, filename, solver)
            plot_field('prs', ax[0, 1], i, filename, solver)
            plot_field('u', ax[1, 0], i, filename, solver)
            plot_field('v', ax[1, 1], i, filename, solver)
            if is_mhd:
                plot_field('bx', ax[0, 2], i, filename, solver)
                plot_field('by', ax[1, 2], i, filename, solver)
                plot_field('bz', ax[2, 2], i, filename, solver)
                plot_field('w', ax[2, 0], i, filename, solver)
                plot_field('divB', ax[2, 1], i, filename, solver)
    plt.legend()
    plt.savefig(f'render/img_{i:04d}.png')
    plt.close('all')
