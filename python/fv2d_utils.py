import numpy as np

get_array = lambda f, x : np.array(f[x])

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
  'divBoverB': r'$\log_{10} \left(\Delta x \cdot \frac{|\nabla \cdot \mathbf{B}|}{|\mathbf{B}|}\right)$'  # Added for divergence over magnitude
}
