import numpy as np
import matplotlib.pyplot as plt
import h5py
from fv2d_utils import *
from tqdm import tqdm


def plot_velocity(f, key):
  get_array = lambda x : np.array(f[key+x])
  u = get_array('u')
  v = get_array('v')  

  x = np.array(f['x'])
  y = np.array(f['y'])
  X, Y = np.meshgrid(x, y)

  plt.quiver(X, Y, u, v)
  plt.show()



filename = 'run_bak.h5'
f = h5py.File(filename, 'r')

Nite = len(f.keys())-2

# Generating fig 4/5 :
plot_velocity(f, f'ite_{Nite-1}/')

f.close()