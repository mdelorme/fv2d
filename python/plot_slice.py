import os
import shutil
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import sys
import argparse

from fv2d_utils import latexify, get_quantity

# Suppression et création du répertoire de rendu
if os.path.exists('render'):
    shutil.rmtree('render')
os.mkdir('render')

# Paramètres
show_grid = False

parser = argparse.ArgumentParser()

parser.add_argument("-f", "--file",
                    nargs='+',
                    # type='checkpath',
                    required=True, 
                    help="Path of the `.h5` file to plot."
                    )

parser.add_argument('-s', "--solver",
                    # default=[" "],
                    # nargs='*',
                    help="Solver associated with the `.h5`file."
                    )

parser.add_argument('-t', '--field',
                    choices=latexify.keys(),
                    default="rho",
                    metavar="FIELD",
                    help="Field to plot."
                    )

parser.add_argument('-y', '--yslice',
                    type=float,
                    help="Y-slice index to plot. If not provided, the middle y-slice will be used.",
                    default=None
                    )

parser.add_argument("-x", "--xslice",
                    type=float,
                    help="X-slice index to plot. If not provided, the middle x-slice will be used.",
                    default=None
                    )

args = parser.parse_args()

xslice = args.xslice
yslice = args.yslice
solvers = args.solver
field = args.field

if args.xslice and args.yslice:
    print("[ERROR] Please select --xslice or --yslice but not both.") # TODO: utiliser les groupes exclusifs
    sys.exit(0)

filenames = args.file
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

def find_index_closest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx

if yslice:
    y = y.reshape((Nx+1, Ny+1)) #access all the y values at x=0
    idxY = find_index_closest(y[0, :], yslice)
    x = x[:Nx]
if xslice:
    x = x.reshape((Nx+1, Ny+1)) #access all the x values at y=0
    print(x)
    idxX = find_index_closest(x[:, 0], xslice) - 1
    y = y[:Ny]

# Fonction pour tracer les champs
def plot_field(field, cax, i, filename, solver):
    with h5py.File(filename, 'r') as f:
        path = f'ite_{i:04d}/{field}'
        arr =  get_quantity(f, i, field)
        y_slice = arr[Ny//2, :] if yslice is None else arr[idxY, :]
        x_slice = x[:Nx] if xslice is None else arr[:, idxX]
        cax.plot(x_slice, y_slice, label=solver)
        cax.set_xlabel(r'$x$')
        cax.set_xlim((xmin-dx, xmax+dx))
        cax.set_ylim((ymin-dy, ymax+dy))
        coords = fr'($x, y={yslice:.2f}$)' if yslice else '$(x,y=cst)$'
        cax.set_ylabel(latexify[field] + coords)
        cax.legend()
    if show_grid:
        cax.set_xticks(np.arange(ext[0], ext[1], dx), minor=True)
        cax.set_yticks(np.arange(ext[2], ext[3], dy), minor=True)
        cax.grid(which='minor', color='k', linestyle='-', linewidth=0.5)
        cax.tick_params(which='minor', bottom=False, left=False)

for i in tqdm(range(Nf)):
    fig, ax = plt.subplots(figsize=(12, 12))
    for filename, solver in zip(filenames, solvers):
        with h5py.File(filename, 'r') as f:
            problem = f.attrs['problem'].title().replace('_', ' ')
            t = f[f'ite_{i:04d}'].attrs['time']
            plt.suptitle(f'{problem} - Time: {t:.3f} - $N_x={Nx}$')
            plot_field(field, ax, i, filename, solver)
    plt.savefig(f'render/img_{i:04d}.png')
    plt.close('all')
