import os
import shutil
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np
import sys
import argparse

from fv2d_utils import latexify, get_quantity, find_tri_layer_Bfield

if os.path.exists('render'):
  shutil.rmtree('render')  
os.mkdir('render')
cwd = os.getcwd()

show_grid = False

def checkfile(filenames):
  if not os.path.exists(filename):
    print(f'[ERROR] File {cwd}/{filename} does not seem to exist.')
    sys.exit(1)


parser = argparse.ArgumentParser()
# parser.register('type', 'checkpath', lambda parser, filenames: checkfile(parser, filenames))
parser.add_argument("-f", "--file",
                    # nargs='+',
                    # type='checkpath',
                    required=True, 
                    help="Path of the `.h5` file to plot."
                    )

parser.add_argument('-s', "--solver",
                    # default=[" "],
                    # nargs='*',
                    help="Solver associated with the `.h5`file."
                  )

parser.add_argument("-t", "--field",
                    choices=latexify.keys(),
                    default="rho",
                    metavar="FIELD",
                    help="Field to plot."
                    )

parser.add_argument('-c', '--colormap',
                    choices=list(colormaps)[:10],
                    default='plasma',
                    help="Matplotlib colormap"
                    )

parser.add_argument("-i", "--init",
                    type=int,
                    default=0,
                    help="Which iteration to start from.")

parser.add_argument("-l", "--last",
                    type=int,
                    default=-1,
                    help="Which iteration to end at.")

args = parser.parse_args()
field = args.field
filename = args.file
print(filename)
checkfile(filename)
solver = args.solver
f = h5py.File(filename, 'r')
Nf = len(f)-2 if args.last == -1 else args.last
Ni = Nf if args.init == -1 else args.init
x = np.array(f['x'])
y = np.array(f['y'])
Nx = int(f.attrs['Nx'])
Ny = int(f.attrs['Ny'])

xmin=x.min()
xmax=x.max()
ymin=y.min()
ymax=y.max()

dx = x[1]-x[0]
dy = y[1]-y[0]

ext = [xmin+0.5*dx, xmax-0.5*dx, ymin+0.5*dy, ymax-0.5*dy]
# vmin, vmax = -15, 0.5
vmin, vmax = None, None
print(f'Rendering animation for file: {cwd}/{filename} and field: {field}')
for i in tqdm(range(Ni, Nf)):
  print("B=", find_tri_layer_Bfield(f, i))
  fig, ax = plt.subplots(figsize=(12, 12))
  t = f['ite_{:04d}'.format(i)].attrs['time']
  problem = f.attrs['problem'].title().replace('_', ' ')
  plt.suptitle(f'{problem} (solver : {solver}) - t={t:.3f} - {Nx=}, {Ny=}')
  path = f'ite_{i:04d}/{field}'
  arr = get_quantity(f, i, field)
  im = ax.imshow(arr, extent=ext, origin='lower', cmap=args.colormap, vmin=vmin, vmax=vmax)
  cbar = fig.colorbar(im, ax=ax,  fraction=0.046, pad=0.04, aspect=20)
  ax.set_title(latexify[field])

  if show_grid:
    ax.set_xticks(np.arange(ext[0], ext[1], dx), minor=True)
    ax.set_yticks(np.arange(ext[2], ext[3], dy), minor=True)
    ax.grid(which='minor', color='k', linestyle='-', linewidth=0.5)
    ax.tick_params(which='minor', bottom=False, left=False)
  plt.savefig(f'render/{field}_{i:04d}.png')
  plt.close('all')
