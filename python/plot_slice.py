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

solvers = ['']
if '--solver' in sys.argv:
    i = sys.argv.index('--solver')
    solvers = [sys.argv[i+1]]

yslice = None
if '--yslice' in sys.argv:
    i = sys.argv.index('--yslice')
    if i+1 >= len(sys.argv):
        print('[ERROR] Please provide a y-slice index after --yslice')
        sys.exit(1)
    yslice = float(sys.argv[i+1])

xslice = None
if '--xslice' in sys.argv:
    i = sys.argv.index('--xslice')
    if i+1 >= len(sys.argv):
        print('[ERROR] Please provide an x-slice index after --xslice')
        sys.exit(1)
    xslice = float(sys.argv[i+1])

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
    print('[ERROR] Please provide a file using --file or --files option.')
    sys.exit(1)

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
# print(x)

def find_index_closest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx

if yslice:
    y = y.reshape((Nx+1, Ny+1)) #access all the y values at x=0
    idx = find_index_closest(y[0, :], yslice)
x_slice = x[:Nx]

# Fonction pour tracer les champs
def plot_field(field, cax, i, filename, solver):
    with h5py.File(filename, 'r') as f:
        path = f'ite_{i:04d}/{field}'
        if field == 'Bmag': # for the loop advection i want to check the mag field intensity
            arr = np.sqrt(np.array(f[f'ite_{i:04d}/bx'])**2 + np.array(f[f'ite_{i:04d}/by'])**2)
        elif field == 'divBoverB':
            bx = np.array(f[f'ite_{i:04d}/bx']).reshape((Ny, Nx))
            by = np.array(f[f'ite_{i:04d}/by']).reshape((Ny, Nx))
            bz = np.array(f[f'ite_{i:04d}/bz']).reshape((Ny, Nx))
            divB = np.abs(np.array(f[f'ite_{i:04d}/divB']).reshape((Ny, Nx)))
            Bmag = np.sqrt(bx**2 + by**2 + bz**2)
            arr = np.log(dx * divB / Bmag)
        else:
            arr = np.array(f[path]).reshape((Ny, Nx))
        y_slice = arr[Ny//2, :] if yslice is None else arr[idx, :]
        cax.plot(x_slice, y_slice, label=solver)
        cax.set_xlabel(r'$x$')
        cax.set_xlim((-1, 1))
        cax.set_ylim((-1, 1))
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
