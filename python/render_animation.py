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

field = 'rho'
if '--field' in sys.argv:
  i = sys.argv.index('--field')
  field = sys.argv[i+1]

f = h5py.File('run_bak.h5', 'r')
Nf = len(f)-2

x = np.array(f['x'])
y = np.array(f['y'])
Nx = x.shape[0]
Ny = y.shape[0]

xmin=x.min()
xmax=x.max()
ymin=y.min()
ymax=y.max()

ext = [xmin, xmax, ymin, ymax]


def plot_field(field, cax, i):
  path = f'ite_{i}/{field}'

  arr = np.array(f[path]).reshape((Ny, Nx))

  cax.imshow(arr, extent=ext, origin='lower')
  cax.set_title(field)
  
print('Rendering animation')
for i in tqdm(range(Nf)):
  fig, ax = plt.subplots(2, 2, figsize=(10, 10))
  plot_field('rho', ax[0,0], i)
  plot_field('prs', ax[0,1], i)
  plot_field('u', ax[1,0], i)
  plot_field('v', ax[1,1], i)

  plt.savefig('img_{:04}.png'.format(i))
  plt.close('all')