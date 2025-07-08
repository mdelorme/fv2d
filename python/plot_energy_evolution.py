import numpy as np
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm

filename = 'run_bak.h5'
f = h5py.File(filename, 'r')

N = len(f.keys())-2

x = f['x']
y = f['y']
dx = x[1]-x[0]
dy = y[1]-y[0]
dV = dx*dy

t = []
M = []

gamma0 = 5.0/3.0


mass = []
Ek   = []
e    = []
E    = []
time = []
for i in tqdm(range(N)):
  key = f'ite_{i}/'

  get_array = lambda x: np.array(f[key+x])

  rho = get_array('rho')
  prs = get_array('prs')
  u   = get_array('u')
  v   = get_array('v')

  _Ek = 0.5 * rho * (u**2.0 + v**2.0)
  _e = prs / (gamma0-1.0)
  _E = _e + _Ek
  _mass = rho

  mass.append(dV * _mass.sum())
  Ek.append(dV * _Ek.sum())
  e.append(dV * _e.sum())
  E.append(dV * _E.sum())
  time.append(f[key].attrs['time'])

f.close()

fig, ax = plt.subplots(2, 2, figsize=(10, 10))

ax[0,0].plot(time, mass)
ax[0,1].plot(time, Ek)
ax[1,0].plot(time, e)
ax[1,1].plot(time, E)

ax[0,0].set_xlabel('Time')
ax[0,1].set_xlabel('Time')
ax[1,0].set_xlabel('Time')
ax[1,1].set_xlabel('Time')

ax[0,0].set_ylabel('Mass')
ax[0,1].set_ylabel('Kinetic energy')
ax[1,0].set_ylabel('Internal energy')
ax[1,1].set_ylabel('Total energy')

plt.tight_layout()

plt.show()

