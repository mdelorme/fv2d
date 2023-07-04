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

mosaic = '''A
B
C
D
E'''

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

def plot_field(field, cax, i):
  arr = get_array(field, i)
  cax.imshow(arr, extent=ext)
  cax.set_title(field)
  
print('Rendering animation')
for i in tqdm(range(Nf)):
  fig, ax = plt.subplot_mosaic(mosaic, figsize=(13, 10))
  plot_field('rho',   ax['A'], i)
  plot_field('prs',   ax['B'], i)
  plot_field('T-<T>', ax['C'], i)
  plot_field('u',     ax['D'], i)
  plot_field('v',     ax['E'], i)

  plt.tight_layout()

  plt.savefig('img_{:04}.png'.format(i))
  plt.close('all')
