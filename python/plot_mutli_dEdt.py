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

folder = '.'
if '--folder' in sys.argv:
    i = sys.argv.index('--folder')
    folder = sys.argv[i+1]

no_rewrite = '--no-rewrite' in sys.argv

start_ite = 0 
if no_rewrite:
  for f in os.listdir('.'):
    if f.startswith('img') and f.endswith('.png'):
      ite = int(f.split('_')[1].split('.')[0])
      start_ite = max(start_ite, ite+1)

all_files = [file for file in os.listdir(folder) if file.endswith('.h5')]
if not all_files:
    print("[ERROR] No files found.")
    sys.exit(1)
Nf = len(all_files)


mosaic = '''A
B
C'''



def get_array(field, file):
  Ny, Nx = file.attrs['Ny'], file.attrs['Nx']
  if field == 'T-<T>':
    rho = get_array('rho', file)
    prs = get_array('prs', file)
    T = prs / rho
    Tbar = np.average(T, axis=1)
    Tprime = np.tile(Tbar, (Nx, 1)).T
    return T-Tprime
  elif field == 'T':
    rho = get_array('rho', file)
    prs = get_array('prs', file)
    return T  
  elif field == 'E':
    rho = get_array('rho', file)
    prs = get_array('prs', file)
    u   = get_array('u', file)
    v   = get_array('v', file)

    Ek = 0.5 * rho * (u**2.0 + v**2.0)
    gamma0 = 5.0/3.0
    E  = Ek + prs / (gamma0-1.0)
    return E
  else:
    path = f'{field}'
    return np.array(file[path]).reshape((Ny, Nx))

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
# ext[2] -= 0.2
# ext[3] -= 0.2 
# f = h5py.File(os.path.join(folder, all_files[0]), 'r')
# cur = get_array('T-<T>', f)
for i in tqdm(range(1, Nf)):
#   prev = cur
  prev = 0
  f = h5py.File(os.path.join(folder, all_files[i]), 'r')
  cur = get_array('T-<T>', f)
  diff = cur - prev
  ext = [np.min(f['x']), np.max(f['x']), np.min(f['y']), np.max(f['y'])]
  fig, ax = plt.subplots(1, 1, figsize=(12, 4))
  im = ax.imshow(diff, extent=ext, cmap='plasma')
  ax.axhline(0.0, linestyle='--', color='k')
  ax.axhline(1.0, linestyle='--', color='k')
#   ax.set_title(r'$E^n - E^{n-1}$')
  ax.set_title(r'$T - <T>$')
  divider = make_axes_locatable(ax)
  ax = divider.append_axes('right', size='5%', pad=0.05)
  fig.colorbar(im, cax=ax, orientation='vertical')

  plt.tight_layout()

  plt.savefig('render/img_{:04}.png'.format(i))
  plt.close('all')
