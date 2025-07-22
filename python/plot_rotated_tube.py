import numpy as np
import matplotlib.pyplot as plt
import h5py
from matplotlib import rc
rc('text', usetex=True)

def format_title(s):
    s = s.replace('$', '')
    s = s.replace(' ', '_')
    return s

# Parameters
# Data files
f_ar5 = h5py.File('rotated_shock_tube.h5', 'r')

# Test case
test = "Skewed shock"

# Resolution time
time = 0.03
Nf = len(f_ar5) - 3
nx = int(f_ar5.attrs['Nx'])
ny = int(f_ar5.attrs['Ny'])
# Accessing the fields
ar5Bx = np.array(f_ar5[f'ite_{Nf:04d}/bx']).reshape(ny, nx)
ar5By = np.array(f_ar5[f'ite_{Nf:04d}/by']).reshape(ny, nx)

# Squeeze the arrays to remove single-dimensional entries
# ar5Bx1 = np.squeeze(ar5Bx)
# ar5By1 = np.squeeze(ar5By)

x = np.linspace(0, 1, nx)
B_parallel5 = (ar5Bx[ny//2, :] + 2*ar5By[ny//2, :])/np.sqrt(5)
o = -1.1071487  # Angle theta

# Calculate the parallel magnetic field

# Plotting the results
fig, axAR = plt.subplots(1, 1)
axAR.set_title(test + ', ' + r'$t=' + str(time) + '$' + ', ' + r'$n_x=n_y=' + str(nx) + '$' + '\n Horizontal slice of ' + r'$ B_{//}$')

plt.plot(x, B_parallel5, label=r'$B_{//}$' + ' ' + r'$5+1$' + ' waves solver ', color='blue', linewidth=1.25)
plt.plot(x, np.zeros((nx)) + 1.41047393, label=r'$B_{//}$' + '   Reference', color='black', linewidth=0.5)

plt.xlabel(r'$x$')
plt.ylabel(r'$B_{//}$')
plt.legend()

plt.savefig(format_title(str(test)) + 'zoomed_out.png', dpi=200)
plt.ylim([1.1, 1.7])
plt.legend()
plt.savefig(format_title(str(test)) + 'usual_scale.png', dpi=200)

fig.clf()
plt.close()

# Close the HDF5 files
f_ar5.close()
