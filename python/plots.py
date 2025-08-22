#!/usr/bin/env python3
# scripts/plot.py
import os
import shutil
import subprocess
import concurrent.futures
from tqdm import tqdm
from core.h5 import Fv2dData, get_slice_data
from core.plotting import (
    setup_figure, plot_2d_field, plot_1d_slice,
    plot_side_by_side, plot_overlay,
    plot_multi_field_side_by_side, plot_multi_field_overlay,
    add_colorbar, add_grid
)
from core.cli import PlotCLI

def generate_video(output_pattern, output_file, fps):
    """Génère une vidéo MP4 à partir des images."""
    command = [
        "ffmpeg", "-framerate", str(fps), "-pattern_type", "glob", "-i", output_pattern,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        output_file
    ]
    subprocess.run(command, check=True)
    print(f"[INFO] Vidéo sauvegardée sous {output_file}")

def plot_field(filename, args):
    """Trace un champ 2D pour un fichier donné."""
    data = Fv2dData([filename])
    output_dir = f"render/{os.path.basename(filename)}"
    os.makedirs(output_dir, exist_ok=True)

    for i in range(len(data)):
        field_data = data[i][args.field]
        t = data.get_time(i)

        fig, ax = setup_figure(
            f"{data.metadata.problem} - t={t:.3f} - {args.field}",
            figsize=(16, 8)
        )
        im = plot_2d_field(ax, data.metadata.x, data.metadata.y, field_data, args.colormap, args.flipy)
        add_colorbar(fig, ax, im)
        if args.show_grid:
            dx = data.metadata.x[1] - data.metadata.x[0]
            dy = data.metadata.y[1] - data.metadata.y[0]
            add_grid(ax, data.metadata.ext, dx, dy)
        plt.savefig(f"{output_dir}/img_{i:04d}.png")
        plt.close()

def plot_slice(filename, args):
    """Trace une slice 1D pour un fichier donné."""
    data = SimulationData([filename])
    slice_data = get_slice_data(filename, args.field, args.xslice, args.yslice)
    output_dir = f"render/{os.path.basename(filename)}"
    os.makedirs(output_dir, exist_ok=True)

    for i in range(len(data)):
        field_data = data[i][args.field]
        t = data.get_time(i)

        fig, ax = setup_figure(
            f"{data.metadata.problem} - t={t:.3f} - Slice {'y' if args.yslice else 'x'}={args.yslice if args.yslice else args.xslice}",
            figsize=(12, 8)
        )

        if args.yslice:  # Slice horizontale (y=cste)
            idx = slice_data['idx']
            y_slice = field_data[idx, :]
            plot_1d_slice(ax, slice_data.x, y_slice, args.field, coords=fr'$(x, y={args.yslice:.2f})$')
        else:  # Slice verticale (x=cste)
            idx = slice_data['idx']
            x_slice = field_data[:, idx]
            plot_1d_slice(ax, x_slice, slice_data.y, args.field, coords=fr'$(x={args.xslice:.2f}, y)$')

        plt.savefig(f"{output_dir}/img_{i:04d}.png")
        plt.close()

def compare_fields_in_simulation(filename, args):
    """Compare plusieurs champs d'une même simulation."""
    data = SimulationData([filename])
    field_names = args.fields if args.fields else [args.field]
    labels = args.labels if args.labels else field_names
    output_dir = f"render/compare_fields_{os.path.basename(filename)}"
    os.makedirs(output_dir, exist_ok=True)

    for i in range(len(data)):
        field_datas = [data[i][field] for field in field_names]
        t = data.get_time(i)

        if args.mode == "side-by-side":
            fig, axes = plt.subplots(1, len(field_datas), figsize=(5 * len(field_datas), 8))
            if len(field_datas) == 1:
                axes = [axes]
            im = plot_multi_field_side_by_side(
                axes, data.metadata.x, data.metadata.y,
                field_datas, field_names, args.colormap, args.flipy
            )
            add_colorbar(fig, axes[0], im, orientation='horizontal')
        else:  # mode == "overlay"
            fig, ax = plt.subplots(figsize=(12, 8))
            im = plot_multi_field_overlay(
                ax, data.metadata.x, data.metadata.y,
                field_datas, field_names, args.colormap, args.flipy
            )
            add_colorbar(fig, ax, im)

        plt.suptitle(f"Comparaison des champs à t={t:.3f}")
        plt.savefig(f"{output_dir}/compare_{i:04d}.png")
        plt.close()

def compare_fields_between_simulations(filenames, args):
    """Compare un même champ entre plusieurs simulations."""
    datasets = [SimulationData([f]) for f in filenames]
    labels = args.labels if args.labels else [os.path.basename(f) for f in filenames]
    field_name = args.fields[0] if args.fields else args.field
    output_dir = f"render/compare_{field_name}"
    os.makedirs(output_dir, exist_ok=True)

    n_iterations = len(datasets[0])
    for i in range(n_iterations):
        field_datas = [data[i][field_name] for data in datasets]
        times = [data.get_time(i) for data in datasets]

        if args.mode == "side-by-side":
            fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 8))
            if len(datasets) == 1:
                axes = [axes]
            im = plot_side_by_side(axes, datasets[0].metadata.x, datasets[0].metadata.y, field_datas, labels, args.colormap, args.flipy)
            add_colorbar(fig, axes[0], im, orientation='horizontal')
        else:  # mode == "overlay"
            fig, ax = plt.subplots(figsize=(12, 8))
            im = plot_overlay(ax, datasets[0].metadata.x, datasets[0].metadata.y, field_datas, labels, args.colormap, args.flipy)
            add_colorbar(fig, ax, im)

        plt.suptitle(f"Comparaison de {field_name} à t={times[0]:.3f}")
        plt.savefig(f"{output_dir}/compare_{i:04d}.png")
        plt.close()

def run_field_command(args):
    """Exécute la commande `field` avec `submit()`."""
    if os.path.exists('render'):
        shutil.rmtree('render')
    os.mkdir('render')

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for filename in args.file:
            future = executor.submit(plot_field, filename, args)
            futures.append(future)

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing files"):
            future.result()  # Attend la fin du traitement

    if args.save_mp4:
        generate_video("render/*/img_*.png", f"{args.field}.mp4", args.fps)

def run_slice_command(args):
    """Exécute la commande `slice` avec `submit()`."""
    if os.path.exists('render'):
        shutil.rmtree('render')
    os.mkdir('render')

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for filename in args.file:
            future = executor.submit(plot_slice, filename, args)
            futures.append(future)

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing files"):
            future.result()

def run_compare_command(args):
    """Exécute la commande `compare`."""
    if os.path.exists('render'):
        shutil.rmtree('render')
    os.mkdir('render')

    if len(args.file) == 1 and args.fields and len(args.fields) > 1:
        compare_fields_in_simulation(args.file[0], args)
        if args.save_mp4:
            generate_video("render/compare_fields_*/compare_*.png", "compare_fields.mp4", args.fps)
    else:
        compare_fields_between_simulations(args.file, args)
        if args.save_mp4:
            field_name = args.fields[0] if args.fields else args.field
            generate_video(f"render/compare_{field_name}/compare_*.png", f"compare_{field_name}.mp4", args.fps)

def main():
    cli = PlotCLI()
    args = cli.parse_args()

    if args.command == 'field':
        run_field_command(args)
    elif args.command == 'slice':
        run_slice_command(args)
    elif args.command == 'compare':
        run_compare_command(args)

if __name__ == "__main__":
    main()
