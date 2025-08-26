
from typing import Any
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


def get_BMag(data: dict[str, np.ndarray], metadata: dict[str, Any]=None) -> np.ndarray:
  """ Compute the norm of the magnetic field from a h5 file containing primitive variables.
      The magnetic field is in 2D and is supposed to have the norm sqrt{B_x^2 + B_y^2}.

    Args:
      - data : dict containing the primitive values at time i
      - i : iteration level of the simulation
    
    Returns :
      - Bmag : np.ndarray 
  """
  return np.sqrt(data['bx']**2 + data['by']**2 + data['bz']**2) 


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


def get_divBoverB(data: list[int, dict], metadata: dict[str, Any]) -> np.ndarray:
  """Returns a value computed from the primitive variables in data"""
  dx = metadata['x'][1] - metadata['x'][0]
  Bmag = get_BMag(data)
  arr = np.log(dx * data['divB'] / Bmag)
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
  # 'Bperp': get_Bperp,
  # 'divBoverB': get_divBoverB,
  # 'TT-prime': get_TTPrime,
  # 'rms': get_rms,
}
