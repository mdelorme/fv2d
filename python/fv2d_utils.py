import numpy as np

get_array = lambda f, x : np.array(f[x])

# Pass from field name to Latex representation
# TODO: split primitive vars and computed values, then add both in one dict
latexify = {
  'rho': r'$\rho$',
  'prs': r'$p$',
  'u': r'$u$',
  'v': r'$v$',
  'w': r'$w$',
  'bx': r'$B_x$',
  'by': r'$B_y$',
  'bz': r'$B_z$',
  'psi': r'$\psi$',
  'divB': r'$\nabla \cdot \mathbf{B}$',
  'Bmag': r'$||\mathbf{B}||$', # = \sqrt{B_x^2 + B_y^2}$',  # Added for magnetic field magnitude
  'divBoverB': r'$\log_{10} \left(\Delta x \cdot \frac{|\nabla \cdot \mathbf{B}|}{|\mathbf{B}|}\right)$',  # Added for divergence over magnitude
  'Bparallele': r'$B_{//}$', # Added for rotated shoc tube
  'T': r'$T$',
  'Tnorm': r'$T - \left<T\right>$',
}


def get_prim_array(f, i, field):
  return np.array(f[f'ite_{i:04d}/{field}'])

def _compute_2D_magnnorm(f, i: int, excludedir: str):
  """Compute 2D magnetic field norm by excluding a specified component."""
  if excludedir == 'x':
    b1 = np.array(f[f'ite_{i:04d}/by'])
    b2 = np.array(f[f'ite_{i:04d}/bz'])
  elif excludedir == 'y':
    b1 = np.array(f[f'ite_{i:04d}/bx'])
    b2 = np.array(f[f'ite_{i:04d}/bz'])
  elif excludedir == 'z':
    b1 = np.array(f[f'ite_{i:04d}/bx'])
    b2 = np.array(f[f'ite_{i:04d}/by'])
  else:
    raise ValueError(f"Direction to exclude must be in ('x', 'y', 'z'), not {excludedir}")
  
  return np.sqrt(b1**2 + b2**2)

def get_BMag(f, i: int):
  """ Compute the norm of the magnetic field from a h5 file containing primitive variables.
      The magnetic field is in 2D and is supposed to have the norm sqrt{B_x^2 + B_y^2}.

    Args:
      - f : h5py.File
      - i : iteration level of the simulation
    
    Returns :
      - Bmag : np.ndarray 
  """
  return _compute_2D_magnnorm(f, i, excludedir='z')


def get_Bperp(f, i: int):
  """ Compute the norm of the perpendicular magnetic components."""
  return _compute_2D_magnnorm(f, i, excludedir='x')


def get_divBoverB(f, i: int):
  Nx = int(f.attrs['Nx'])
  Ny = int(f.attrs['Ny'])
  x = np.array(f['x'])
  dx = x[1]-x[0]
  bx = np.array(f[f'ite_{i:04d}/bx']).reshape((Ny, Nx))
  by = np.array(f[f'ite_{i:04d}/by']).reshape((Ny, Nx))
  bz = np.array(f[f'ite_{i:04d}/bz']).reshape((Ny, Nx))
  divB = np.abs(np.array(f[f'ite_{i:04d}/divB']).reshape((Ny, Nx)))
  Bmag = np.sqrt(bx**2 + by**2 + bz**2)
  arr = np.log(dx * divB / Bmag)
  return arr

def get_temperature(f, i: int):
  cste = 1
  return cste * get_prim_array(f, i, 'prs') / get_prim_array(f, i, 'rho')


def get_norm_temperature(f, i: int):
  Nx = f.attrs['Nx']
  T = get_temperature(f, i)
  averageT = np.tile(np.nanmean(T,axis=1),(Nx, 1)).T
  return (T - averageT)/ averageT * 100


def find_tri_layer_Bfield(f, i):
  Nx, Ny = f.attrs['Nx'], f.attrs['Ny']
  vx = get_prim_array(f, i, 'vx').reshape(Ny, Nx)
  vy = get_prim_array(f, i, 'vy').reshape(Ny, Nx)
  rho = get_prim_array(f, i, 'rho')
  x, y = f['x'], f['y']
  B2_avg = 0
  for i in range(Nx):
    for j in range(Ny):
      v2 = vx[i, j]**2 + vy[i, j]**2
      B2_avg += rho[i, j] * v2
  return np.sqrt(B2_avg)/((x.max() - x.min()) * (y.max() - y.min()))


compute_values = {
  'Bmag': get_BMag,
  'Bperp': get_Bperp,
  'divBoverB': get_divBoverB,
  'T': get_temperature,
  'Tnorm': get_norm_temperature,
}

def get_quantity(f, i, field):
  if field in ('rho', 'prs', 'u', 'v', 'w', 'bx', 'by', 'bz', 'psi', 'divB'):
    return get_prim_array(f, i, field)
  elif field in compute_values:
    return compute_values[field](f, i)
  else:
    raise ValueError(f'No known quantity to evaluate for {field=}.')
