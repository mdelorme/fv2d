import os
import shutil
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import sys

if os.path.exists('render'):
  shutil.rmtree('render')
os.mkdir('render')

show_grid = False

field = 'rho'
if '--field' in sys.argv:
  i = sys.argv.index('--field')
  field = sys.argv[i+1]

f = h5py.File('run.h5', 'r')
Nf = len(f)-2

x = np.array(f['x'])
y = np.array(f['y'])
Nx = x.shape[0]
Ny = y.shape[0]

xmin=x.min()
xmax=x.max()
ymin=y.min()
ymax=y.max()

dx = x[1]-x[0]
dy = y[1]-y[0]

ext = [xmin-0.5*dx, xmax+0.5*dx, ymin-0.5*dy, ymax+0.5*dy]


def plot_field(field, cax, i):
  path = f'ite_{i}/{field}'

  arr = np.array(f[path]).reshape((Ny, Nx))

  cax.imshow(arr, extent=ext, origin='lower')
  cax.set_title(field)

  if show_grid:
    cax.set_xticks(np.arange(ext[0], ext[1], dx), minor=True)
    cax.set_yticks(np.arange(ext[2], ext[3], dy), minor=True)

    # Gridlines based on minor ticks
    cax.grid(which='minor', color='k', linestyle='-', linewidth=1)

    # Remove minor ticks
    cax.tick_params(which='minor', bottom=False, left=False)

  
print('Rendering animation')
for i in tqdm(range(Nf)):
  fig, ax = plt.subplots(2, 2, figsize=(10, 10))
  plot_field('rho', ax[0,0], i)
  plot_field('prs', ax[0,1], i)
  plot_field('u', ax[1,0], i)
  plot_field('v', ax[1,1], i)

  plt.savefig('img_{:04}.png'.format(i))
  plt.close('all')
