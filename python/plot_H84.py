import numpy as np
import matplotlib.pyplot as plt
import h5py
from fv2d_utils import *
from tqdm import tqdm

filename = 'run_bak.h5'
f = h5py.File(filename, 'r')

step = 1

N = len(f.keys())-2

x = np.array(f['x'])
y = np.array(f['y'])

ext = [x[0], x[-1], y[0], y[-1]]

for i in tqdm(range(0, N, step)):
  key = f'ite_{i}/'
  get_array = lambda x : np.array(f[key+x])

  u = get_array('u')
  v = get_array('v')

  vel = np.sqrt(u**2.0+v**2.0)

  fig, ax = plt.subplots(1, 1, figsize=(8, 4))
  plt.imshow(vel, cmap='jet', extent=ext, origin='lower')
  plt.tight_layout()
  plt.savefig('img.{:04}.png'.format(i))
  plt.close('all')

f.close()
