import os
import shutil
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import sys

from fv2d_utils import latexify, get_quantity

if os.path.exists('render'):
  shutil.rmtree('render')  
os.mkdir('render')
cwd = os.getcwd()

show_grid = False

field = 'rho'
if '--field' in sys.argv:
  i = sys.argv.index('--field')
  field = sys.argv[i+1]

if field not in latexify:
  print(f'[ERROR] Field {field} is not recognized. Available fields: {", ".join(latexify.keys())}')
  sys.exit(1)

colmap = 'inferno'
if '--colormap' in sys.argv:
  i = sys.argv.index('--colormap')
  colmap = sys.argv[i+1]

solver = 'unknown'
if '--solver' in sys.argv:
  i = sys.argv.index('--solver')
  solver = sys.argv[i+1]
if '--file' in sys.argv:
  i = sys.argv.index('--file')
  if i+1 >= len(sys.argv):
    print('[ERROR] Please provide a filename after --file')
    sys.exit(1)
  filename = sys.argv[i+1]
else:
  filename = 'run.h5'
if not os.path.exists(filename):
  print(f'[ERROR] File {cwd}/{filename} does not seem to exist.')
  sys.exit(1)
f = h5py.File(filename, 'r')
Nf = len(f)-2

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
for i in tqdm(range(Nf)):
  fig, ax = plt.subplots(figsize=(12, 12))
  t = f['ite_{:04d}'.format(i)].attrs['time']
  problem = f.attrs['problem'].title().replace('_', ' ')
  plt.suptitle(f'{problem} (solver : {solver}) - t={t:.3f} - {Nx=}, {Ny=}')
  path = f'ite_{i:04d}/{field}'
  arr = get_quantity(f, i, field)
  im = ax.imshow(arr, extent=ext, origin='lower', cmap=colmap, vmin=vmin, vmax=vmax)
  cbar = fig.colorbar(im, ax=ax,  fraction=0.046, pad=0.04, aspect=20)
  ax.set_title(latexify[field])

  if show_grid:
    ax.set_xticks(np.arange(ext[0], ext[1], dx), minor=True)
    ax.set_yticks(np.arange(ext[2], ext[3], dy), minor=True)
    ax.grid(which='minor', color='k', linestyle='-', linewidth=0.5)
    ax.tick_params(which='minor', bottom=False, left=False)
  plt.savefig(f'render/{field}_{i:04d}.png')
  plt.close('all')
