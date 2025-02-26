import h5py
import matplotlib.pyplot as plt

Nx = 64
Ny = 1
with h5py.File('build/run.h5', 'r') as fichier:
    print("Clés disponibles :", list(fichier.keys()))
    Nite = len(fichier.keys())-2 # removing x and y
    if Ny < 2:
        x = fichier['x'][:Nx]
        for num in range(Nite):
            p = fichier[f'ite_{num:04d}']['prs'][:][0]
            r = fichier[f'ite_{num:04d}']['rho'][:][0]
            u = fichier[f'ite_{num:04d}']['u'][:][0]
            v = fichier[f'ite_{num:04d}']['v'][:][0]
            w = fichier[f'ite_{num:04d}']['w'][:][0]
            by = fichier[f'ite_{num:04d}']['by'][:][0]
            bz = fichier[f'ite_{num:04d}']['bz'][:][0]

            fig, ax = plt.subplots(3, 2, figsize=(15, 10))
            ax[0,0].plot(x, r)
            ax[0,1].plot(x, p)
            ax[1,0].plot(x, u)
            ax[1,1].plot(x, v)
            ax[2,0].plot(x, by)
            ax[2,1].plot(x, bz)
            ax[0,0].set_title('Density')
            ax[0,1].set_title('Pressure')
            ax[1,0].set_title('$v_x$')
            ax[1,1].set_title('$v_y$')
            ax[2,0].set_title('$B_y$')
            ax[2,1].set_title('$B_z$')
            plt.tight_layout()
            plt.savefig(f"ite__{num}.png")
            plt.close()
    else:
        fig, ax = plt.subplots(3, 2, figsize=(10, 15))
        Nite -= 1
        ax[0,0].imshow(fichier[f'ite_{Nite:04d}']['rho'])
        ax[0,1].imshow(fichier[f'ite_{Nite:04d}']['prs'])
        ax[1,0].imshow(fichier[f'ite_{Nite:04d}']['u'])
        ax[1,1].imshow(fichier[f'ite_{Nite:04d}']['v'])
        ax[2,0].imshow(fichier[f'ite_{Nite:04d}']['by'])
        ax[2,1].imshow(fichier[f'ite_{Nite:04d}']['bz'])

        ax[0,0].set_title('Density')
        ax[0,1].set_title('Pressure')
        ax[1,0].set_title('$v_x$')
        ax[1,1].set_title('$v_y$')
        ax[2,0].set_title('$B_y$')
        ax[2,1].set_title('$B_z$')
        plt.tight_layout()
        plt.show()
        plt.close()