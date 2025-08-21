import os
import shutil
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import colormaps
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import sys
import argparse
import subprocess
import concurrent.futures
from fv2d_utils import latexify, get_quantity, find_tri_layer_Bfield

if os.path.exists('render'):
    shutil.rmtree('render')
os.mkdir('render')
cwd = os.getcwd()

show_grid = False

def checkfile(filename):
    if not os.path.exists(filename):
        print(f'[ERROR] File {filename} does not seem to exist.')
        sys.exit(1)

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", nargs='+', required=True, help="Path of the `.h5` file(s) to plot.")
parser.add_argument('-s', "--solver", help="Solver associated with the `.h5` file(s).")
parser.add_argument("-t", "--field", choices=latexify.keys(), default="rho", metavar="FIELD", help="Field to plot.")
parser.add_argument('-c', '--colormap', choices=list(colormaps), default='plasma', help="Matplotlib colormap")
parser.add_argument("-i", "--init", type=int, default=0, help="Which iteration to start from.")
parser.add_argument("-l", "--last", type=int, default=-1, help="Which iteration to end at.")
parser.add_argument('--flipy', action='store_true', help="Flip the y-axis (useful for some simulations).")
parser.add_argument('--fps', type=int, default=25, help="Number of frames per second for the animation.")
parser.add_argument('--saveMP4', action='store_true', help="Save the animation as an MP4 file.")

args = parser.parse_args()
field = args.field
filenames = args.file
solver = args.solver

# file_iterator = tqdm(filenames, desc="Processing files") if len(filenames) > 1 else filenames

def plot_snapshot(filename):
    checkfile(filename)
    with h5py.File(filename, 'r') as f:
        Nf = len(f) - 2 if args.last == -1 else args.last
        Ni = Nf if args.init == -1 else args.init
        x = np.array(f['x'])
        y = np.array(f['y'])
        Nx = int(f.attrs['Nx'])
        Ny = int(f.attrs['Ny'])
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
        dx, dy = x[1] - x[0], y[1] - y[0]
        ext = [xmin + 0.5 * dx, xmax - 0.5 * dx, ymin + 0.5 * dy, ymax - 0.5 * dy]

        if args.flipy:
            ext[2], ext[3] = ymax - 0.5 * dy, ymin + 0.5 * dy

        is_multiple = 'ite_0000' in f

        if not is_multiple:
            Nf = 1
            Ni = 0

        time_iterator = tqdm(range(Ni, Nf), desc="Processing iterations") if len(filenames) == 1 else range(Ni, Nf)

        for i in time_iterator:
            if i % args.fps == 0:
                fig, ax = plt.subplots(figsize=(16, 8))
                t = f[f'ite_{i:04d}'].attrs['time'] if is_multiple else f.attrs['time']
                problem = f.attrs['problem'].title().replace('_', ' ')
                plt.suptitle(f'{problem} (solver: {solver}) - t={t:.3f} - {Nx=}, {Ny=}')
                i = i if is_multiple else -1
                arr = np.flipud(get_quantity(f, i, field)) if args.flipy else get_quantity(f, i, field)
                im = ax.imshow(arr, extent=ext, origin='lower', cmap=args.colormap, aspect='auto')
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax, boundaries=(-0.006, 0.006))
                ax.set_title(latexify[field])

                if show_grid:
                    ax.set_xticks(np.arange(ext[0], ext[1], dx), minor=True)
                    ax.set_yticks(np.arange(ext[2], ext[3], dy), minor=True)
                    ax.grid(which='minor', color='k', linestyle='-', linewidth=0.5)
                    ax.tick_params(which='minor', bottom=False, left=False)

                plt.savefig(f'render/img_{int(t):04d}.png')
                plt.close('all')


if __name__ == "__main__":
    with concurrent.futures.ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(plot_snapshot, filenames), desc="Processing files", total=len(filenames)))

    if args.saveMP4:
        command = [
            "ffmpeg",
            "-framerate", str(args.fps),
            "-pattern_type", "glob",
            "-i", "render/img_*.png",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            f"{field}.mp4"
        ]
        subprocess.run(command, cwd=cwd)
        print(f'[INFO] Animation saved as {field}.mp4')
