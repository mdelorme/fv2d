import os
import shutil
import h5py
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import sys

# Pass from field name to Latex representation
latexify = {
  'rho': r'$\rho$',
  'prs': r'$p$',
  'u': r'$u$',
  'v': r'$v$',
  'bx': r'$B_x$',
  'by': r'$B_y$',
  'bz': r'$B_z$',
  'psi': r'$\psi$',
  'divB': r'$\nabla \cdot \mathbf{B}$',
  'Bmag': r'$|\mathbf{B}|$',  # Added for magnetic field magnitude
  'divBoverB': r'$\log_{10} \left(dx \frac{|\nabla \cdot \mathbf{B}|}{|\mathbf{B}|}\right)$'  # Added for divergence over magnitude
}


if os.path.exists('render'):
  shutil.rmtree('render')  
os.mkdir('render')
cwd = Path().absolute()

show_grid = False

field = 'rho'
if '--field' in sys.argv:
  i = sys.argv.index('--field')
  field = sys.argv[i+1]

if field not in latexify:
  print(f'[ERROR] Field {field} is not recognized. Available fields: {", ".join(latexify.keys())}')
  sys.exit(1)

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
  print(f'[ERROR] File {cwd/filename} does not seem to exist.')
  sys.exit(1)
f = h5py.File(filename, 'r')
Nf = len(f)-2

x = np.array(f['x'])
y = np.array(f['y'])
Nx = f.attrs['Nx']
Ny = f.attrs['Ny']

xmin=x.min()
xmax=x.max()
ymin=y.min()
ymax=y.max()

dx = x[1]-x[0]
dy = y[1]-y[0]

ext = [xmin-0.5*dx, xmax+0.5*dx, ymin-0.5*dy, ymax+0.5*dy]
# vmin, vmax = -15, 0.5
vmin, vmax = None, None
print(f'Rendering animation for file: {cwd / filename} and field: {field}')
for i in tqdm(range(Nf)):
  fig, ax = plt.subplots(figsize=(12, 12))
  t = f['ite_{:04d}'.format(i)].attrs['time']
  problem = f.attrs['problem'].title()
  plt.suptitle(f'{problem} ({solver=}) - Time: {t:.3f}')
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
  im = ax.imshow(arr, extent=ext, origin='lower', vmin=vmin, vmax=vmax)
  cbar = fig.colorbar(im, ax=ax,  fraction=0.046, pad=0.04, aspect=20)
  ax.set_title(latexify[field])

  if show_grid:
    ax.set_xticks(np.arange(ext[0], ext[1], dx), minor=True)
    ax.set_yticks(np.arange(ext[2], ext[3], dy), minor=True)
    ax.grid(which='minor', color='k', linestyle='-', linewidth=0.5)
    ax.tick_params(which='minor', bottom=False, left=False)
  plt.savefig(f'render/{field}_{i:04d}.png')
  plt.close('all')
