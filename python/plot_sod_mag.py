import h5py
import matplotlib.pyplot as plt


with h5py.File('build/run.h5', 'r') as fichier:
    print("Clés disponibles :", list(fichier.keys()))
    Nite = len(fichier.keys())-2 # removing x and y

    x = fichier['x'][:64]
    for num in range(Nite):
        p = fichier[f'ite_{num:04d}']['prs'][:].reshape(64)
        r = fichier[f'ite_{num:04d}']['rho'][:].reshape(64)
        u = fichier[f'ite_{num:04d}']['u'][:].reshape(64)
        v = fichier[f'ite_{num:04d}']['v'][:].reshape(64)
        w = fichier[f'ite_{num:04d}']['w'][:].reshape(64)
        by = fichier[f'ite_{num:04d}']['by'][:].reshape(64)
        bz = fichier[f'ite_{num:04d}']['bz'][:].reshape(64)

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