import numpy as np
import matplotlib.pyplot as plt
import h5py
import os

colella_deform_N = 20
colella_deform_max = 0.9

colella_deform = np.linspace(0, colella_deform_max, colella_deform_N)


with h5py.File('run.h5', 'r') as file:
    N = len(file.keys()) - 4       # number timestamp

    Nx = file.attrs['Nx']
    Ny = file.attrs['Ny']

    cx = np.array(file['center_x']).reshape([Nx, Nx])
    cy = np.array(file['center_y']).reshape([Ny, Ny])
    rho = file[f"ite_{N-1}/rho"]
    
    plt.gca().set_aspect('equal')

    # plt.pcolor(cx, cy, rho)
    plt.contour(cx, cy, rho, 30, linewidths=0.4)

    plt.show()


# for i in range(N):
#     rho = file[f"ite_{i}/prs"]
#     plt.clf()
#     plt.scatter(cx, rho, marker='.', s=0.7)
#     plt.pause(0.1)
