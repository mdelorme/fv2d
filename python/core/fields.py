import numpy as np

# Pass from field name to Latex representation
latexify = {
  'rho': r'$\rho$',
  'prs': r'$p$',
  'u': r'$u$',
  'v': r'$v$',
  'bx': r'$B_x$',
  'by': r'$B_y$',
  'bz': r'$B_z$',
  'psi': r'$\psi$',
  'divB': r'$\nabla \cdot \mathbf{B}$',
  'Bmag': r'$||\mathbf{B}||$', # = \sqrt{B_x^2 + B_y^2}$',  # Added for magnetic field magnitude
  'divBoverB': r'$\log_{10} \left(\Delta x \cdot \frac{|\nabla \cdot \mathbf{B}|}{|\mathbf{B}|}\right)$',  # Added for divergence over magnitude
  'Bperp': r'$B_{//}$', # Added for rotated shoc tube,
  'TT-prime': r'$T - <T>$', # Added for the tri-layer
  'rms': r'$\frac{v}{<v^2> - <v>^2}$',  # Added for the tri-layer
}


# def _path(i, field):
#   path = f'ite_{i:04d}/{field}' if i != -1 else str(field)
#   return path

def get_shape(f) -> tuple[int, int]:
  """Get the shape of the grid from the h5 file."""
  Nx = f.attrs['Nx']
  Ny = f.attrs['Ny']
  return (Ny, Nx)

def get_prim_array(f, i, field):
  Nx = f.attrs['Nx']
  Ny = f.attrs['Ny']
  path = f'ite_{i:04d}/{field}' if i != -1 else str(field)
  return np.array(f[path]).reshape((Ny, Nx))


def _compute_2D_magnnorm(f, i: int, excludedir: str):
  """Compute 2D magnetic field norm by excluding a specified component."""

  if excludedir == 'x':
    b1 = np.array(f[_path(i, 'by')])
    b2 = np.array(f[_path(i, 'bz')])
  elif excludedir == 'y':
    b1 = np.array(f[_path(i, 'bx')])
    b2 = np.array(f[_path(i, 'bz')])
  elif excludedir == 'z':
    b1 = np.array(f[_path(i, 'bx')])
    b2 = np.array(f[_path(i, 'by')])
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
  return _compute_2D_magnnorm(f, i, excludedir='z').reshape((f.attrs['Ny'], f.attrs['Nx']))


def get_rms(f, i: int):
  """
  vy-rms(y) = sqrt(<vy^2> - <vy>^2), calculé à chaque y  avec <> la moyenne horizontale selon x.
  """
  Ny, Nx = get_shape(f)
  v = get_prim_array(f, i, 'v').reshape((Ny, Nx))
  v2 = v**2
  v2_avg = np.mean(v2, axis=0, keepdims=True)
  v_avg = np.mean(v, axis=0, keepdims=True)
  v_rms = np.sqrt(v2_avg - v_avg**2)
  return v / v_rms  # Reshape to match the grid dimensions (Ny, Nx)


def find_tri_layer_Bfield(f, i):
  Nx, Ny = f.attrs['Nx'], f.attrs['Ny']
  vx = get_prim_array(f, i, 'v').reshape(Ny, Nx)
  vy = get_prim_array(f, i, 'v').reshape(Ny, Nx)
  rho = get_prim_array(f, i, 'rho')
  x, y = np.array(f['x']), np.array(f['y'])
  B2_avg = 0
  for i in range(Nx):
    for j in range(Ny):
      v2 = vx[j, i]**2 + vy[j, i]**2
      B2_avg += rho[j, i] * v2
  return np.sqrt(B2_avg)/((x.max() - x.min()) * (y.max() - y.min()))


def get_Bperp(f, i: int):
  """ Compute the norm of the perpendicular magnetic components."""
  return _compute_2D_magnnorm(f, i, excludedir='x')


def get_divBoverB(f, i: int):
  Nx = int(f.attrs['Nx'])
  Ny = int(f.attrs['Ny'])
  x = np.array(f['x'])
  dx = x[1]-x[0]
  bx = np.array(f[_path(i, 'bx')]).reshape((Ny, Nx))
  by = np.array(f[_path(i, 'by')]).reshape((Ny, Nx))
  bz = np.array(f[_path(i, 'bz')]).reshape((Ny, Nx))
  divB = np.abs(np.array(f[_path(i, 'divB')]).reshape((Ny, Nx)))
  Bmag = np.sqrt(bx**2 + by**2 + bz**2)
  arr = np.log(dx * divB / Bmag)
  return arr

def get_TTPrime(f, i: int):
  """ Compute the temperature fluctuation T - <T>."""
  rho = get_prim_array(f, i, 'rho')
  prs = get_prim_array(f, i, 'prs')
  T = prs / rho
  Tbar = np.average(T, axis=1)
  Tprime = np.tile(Tbar, (T.shape[1], 1)).T
  return T - Tprime

compute_values = {
  'Bmag': get_BMag,
  'Bperp': get_Bperp,
  'divBoverB': get_divBoverB,
  'TT-prime': get_TTPrime,
  'rms': get_rms,
}

def get_quantity(f, i, field):
  """set i to -1 to ignore, behaviour is to change"""
  if field in ('rho', 'prs', 'u', 'v', 'w', 'bx', 'by', 'bz', 'psi', 'divB'):
    return get_prim_array(f, i, field)
  elif field in compute_values:
    return compute_values[field](f, i)
  else:
    raise ValueError(f'No known quantity to evaluate for {field=}.')
