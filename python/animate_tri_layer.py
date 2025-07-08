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
dy = y[1]-y[0]
Nx = x.shape[0]
Ny = y.shape[0]
g = 20.0
y1 = 1.5
y2 = 2.5

xmin=x.min()
xmax=x.max()
ymin=y.min()
ymax=y.max()

mosaic = '''AB
CD
EF'''

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
    return prs/rho 
  elif field == 'HSE':
    rho = get_array('rho', ite)
    prs = get_array('prs', ite)
    dP = (prs[2:,:] - prs[:-2,:]) / (2.0 * dy)
    return dP - rho[1:-1,:]*g
  else:
    path = f'ite_{ite}/{field}'
    return np.array(f[path]).reshape((Ny, Nx))

def plot_field(field, cax, i, clim=None, cmap='viridis'):
  arr = get_array(field, i)
  im = cax.imshow(arr, extent=ext, clim=clim, cmap=cmap)
  cax.set_title(field)
  cax.axhline(ymax-y1, linestyle='--', color='k')
  cax.axhline(ymax-y2, linestyle='--', color='k')
  divider = make_axes_locatable(cax)
  cax = divider.append_axes('right', size='5%', pad=0.05)
  fig.colorbar(im, cax=cax, orientation='vertical')

  
print('Rendering animation')
for i in tqdm(range(start_ite, Nf)):
  fig, ax = plt.subplot_mosaic(mosaic, figsize=(10, 10))
  Tlim = None
  plot_field('rho',   ax['A'], i)
  plot_field('prs',   ax['B'], i)
  plot_field('T',     ax['C'], i, Tlim, 'YlOrBr_r')
  plot_field('u',     ax['D'], i, (-0.1, 0.1), 'seismic')
  plot_field('v',     ax['E'], i, (-0.1, 0.1), 'seismic')
  plot_field('HSE',   ax['F'], i)
  #fig.delaxes(ax['F'])

  plt.tight_layout()

  plt.savefig('img_{:04}.png'.format(i))
  plt.close('all')
