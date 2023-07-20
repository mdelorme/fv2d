import os
import shutil
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import sys
from mpl_toolkits.axes_grid1 import make_axes_locatable

if os.path.exists('render'):
  shutil.rmtree('render')
os.mkdir('render')

field = 'rho'
if '--field' in sys.argv:
  i = sys.argv.index('--field')
  field = sys.argv[i+1]

no_rewrite = '--no-rewrite' in sys.argv

start_ite = 0 
if no_rewrite:
  for f in os.listdir('.'):
    if f.startswith('img') and f.endswith('.png'):
      ite = int(f.split('_')[1].split('.')[0])
      start_ite = max(start_ite, ite+1)


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

mosaic = '''A
B
C'''

ext = [xmin, xmax, ymin, ymax]

def get_array(field, ite):
  if field == 'T-<T>':
    rho = get_array('rho', ite)
    prs = get_array('prs', ite)
    T = prs / rho
    Tbar = np.average(T, axis=1)
    Tprime = np.tile(Tbar, (Nx, 1)).T
    return T-Tprime
  elif field == 'T':
    rho = get_array('rho', ite)
    prs = get_array('prs', ite)
    return T  
  else:
    path = f'ite_{ite}/{field}'
    return np.array(f[path]).reshape((Ny, Nx))

def plot_field(field, cax, i, clim=None, cmap='viridis'):
  arr = get_array(field, i)
  im = cax.imshow(arr, extent=ext, cmap=cmap, clim=clim)
  cax.axhline(0.0, linestyle='--', color='k')
  cax.axhline(1.0, linestyle='--', color='k')
  cax.set_title(field)
  divider = make_axes_locatable(cax)
  cax = divider.append_axes('right', size='5%', pad=0.05)
  
  fig.colorbar(im, cax=cax, orientation='vertical')

  
print('Rendering animation')
ext[2] -= 0.2
ext[3] -= 0.2 
  
for i in tqdm(range(start_ite, Nf)):
  fig, ax = plt.subplot_mosaic(mosaic, figsize=(10, 10))
  plot_field('T-<T>', ax['A'], i, (-2.0, 2.0), 'seismic')
  plot_field('u',     ax['B'], i, (-1.5, 1.5), 'seismic')
  plot_field('v',     ax['C'], i, (-1.5, 1.5), 'seismic')

  time = f[f'ite_{i}'].attrs['time']
  plt.suptitle('t={:.4f}'.format(time))

  plt.tight_layout()

  plt.savefig('img_{:04}.png'.format(i))
  plt.close('all')
