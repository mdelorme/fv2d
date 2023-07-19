import numpy as np
import matplotlib.pyplot as plt
import h5py
import os

colella_deform_N = 20
colella_deform_max = 0.9

colella_deform = np.linspace(0, colella_deform_max, colella_deform_N)


fig, ax = plt.subplots(2, 1)

for i, cd in enumerate(colella_deform):
    os.system(f'sed "s/geometry_colella_param.*/geometry_colella_param={cd/ (2*np.pi)}/" sod_x.ini > tmp.ini')
    os.system('./fv2d tmp.ini')

    with h5py.File('run.h5', 'r') as file:
        N = len(file.keys()) - 4       # number timestamp

        Nx = file.attrs['Nx']
        Ny = file.attrs['Ny']

        cx = np.array(file['center_x']).reshape([Nx, Nx])
        cy = np.array(file['center_y']).reshape([Ny, Ny])
        rho = file[f"ite_{N-1}/rho"]
        prs = file[f"ite_{N-1}/prs"]

        for a in ax:
            a.clear()
            a.grid()
            a.set_xlim([-1, 1])
            a.set_xlabel("x")

        ax[0].set_title(f"colella {cd:1.3f}")
        ax[0].set_ylim([ 0.1, 1.03])
        ax[0].set_ylabel("rho")
        ax[0].scatter(cx, rho, marker='.', s=0.7, color="blue")
        ax[1].set_ylim([ 0.1, 1.03])
        ax[1].set_ylabel("prs")
        ax[1].scatter(cx, prs, marker='.', s=0.7, color="red")

        plt.savefig(f"tmp/{i:03}.png")
        # plt.show()

os.system('ffmpeg -y -r 10 -pattern_type glob -i "tmp/*.png" -pix_fmt yuv420p -c:v libx264 sod_colella.mkv')


# for i in range(N):
#     rho = file[f"ite_{i}/prs"]
#     plt.clf()
#     plt.scatter(cx, rho, marker='.', s=0.7)
#     plt.pause(0.1)
