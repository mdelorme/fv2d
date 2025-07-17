import os
import shutil
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import sys
from fv2d_utils import latexify


if os.path.exists('render'):
  shutil.rmtree('render')
os.mkdir('render')
cwd = os.getcwd()

show_grid = False

field = 'rho'
if '--field' in sys.argv:
  i = sys.argv.index('--field')
  field = sys.argv[i+1]

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
Nx = f.attrs['Nx']
Ny = f.attrs['Ny']

xmin=x.min()
xmax=x.max()
ymin=y.min()
ymax=y.max()

dx = x[1]-x[0]
dy = y[1]-y[0]

ext = [xmin-0.5*dx, xmax+0.5*dx, ymin-0.5*dy, ymax+0.5*dy]

is_mhd = True if 'bx' in f['ite_0000'] else False

def plot_field(field, cax, i):
  path = f'ite_{i:04d}/{field}'
  arr = np.array(f[path]).reshape((Ny, Nx))
  legend = latexify[field]
  # if field == 'bz': # for the loop advection i want to check the mag field intensity
  #   arr = np.sqrt(np.array(f[f'ite_{i:04d}/bx'])**2 + np.array(f[f'ite_{i:04d}/by'])**2)
  #   legend = r'$\sqrt{B_x^2 + B_y^2}$'
  cax.imshow(arr, extent=ext, origin='lower')
  cax.set_title(legend)

  if show_grid:
    cax.set_xticks(np.arange(ext[0], ext[1], dx), minor=True)
    cax.set_yticks(np.arange(ext[2], ext[3], dy), minor=True)

    # Gridlines based on minor ticks
    cax.grid(which='minor', color='k', linestyle='-', linewidth=1)

    # Remove minor ticks
    cax.tick_params(which='minor', bottom=False, left=False)

  
print(f'Rendering animation for file: {cwd}/{filename}')
for i in tqdm(range(Nf)):
  fig, ax = plt.subplots(3, 3, figsize=(12, 12)) if is_mhd else plt.subplots(2, 2, figsize=(10, 10))
  t = f['ite_{:04d}'.format(i)].attrs['time']
  problem = f.attrs['problem'].title().replace('_', ' ')
  plt.suptitle(rf'{problem} - Time: {t:.3f} - $N_x={Nx} \times N_y={Ny}$')
  plot_field('rho', ax[0,0], i)
  plot_field('prs', ax[0,1], i)
  plot_field('u', ax[1,0], i)
  plot_field('v', ax[1,1], i)
  if is_mhd:
    plot_field('bx', ax[0,2], i)
    plot_field('by', ax[1,2], i)
    plot_field('bz', ax[2,2], i)
    plot_field('psi', ax[2,0], i)
    plot_field('divB', ax[2,1], i)

  plt.savefig('render/img_{:04}.png'.format(i))
  plt.close('all')
