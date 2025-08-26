import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from mpl_toolkits.axes_grid1 import make_axes_locatable
from .fields import latexify

# TODO: Typer les objets matplotlib

def setup_figure(title: str, figsize=(12, 12)):
    fig, ax = plt.subplots(figsize=figsize)
    plt.suptitle(title)
    return fig, ax


def add_colorbar(fig, ax, im, boundaries=(-0.006, 0.006)):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax, boundaries=boundaries)


def add_grid(ax, ext, dx, dy):
    ax.set_xticks(np.arange(ext[0], ext[1], dx), minor=True)
    ax.set_yticks(np.arange(ext[2], ext[3], dy), minor=True)
    ax.grid(which="minor", color='w', linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)


def plot_2d_field(ax, ext: list, field_data, colormap: str='plasma', flipy: bool=False):
    im = ax.imshow(
        np.flipud(field_data) if flipy else field_data,
        extent=ext,
        origin='lower',
        cmap=colormap,
        aspect='auto'
    )
    return im


def plot_1d_slice(ax, x_or_y, slice_data, field, solver=None, coords=None):
    ax.plot(x_or_y, slice_data, label=solver)
    ax.set_xlabel(r'$x$' if coords.startswith('$(x,') else r'$y$')
    ax.set_ylabel(latexify[field] + coords)
    ax.legend()

def plot_side_by_side(axes, x, y, field_datas, labels, colormap='plasma', flipy=False):
    """
    Trace plusieurs champs/simulations côte à côte.
    Args:
        axes: Liste d'axes (un par champ/simulation).
        x, y: Coordonnées spatiales.
        field_datas: Liste des données à tracer.
        labels: Liste des labels pour chaque sous-figure.
        colormap: Colormap à utiliser.
        flipy: Si True, inverse l'axe y.
    """
    ims = []
    for ax, data, label in zip(axes, field_datas, labels):
        im = ax.imshow(
            np.flipud(data) if flipy else data,
            extent=[x.min(), x.max(), y.min(), y.max()],
            origin='lower',
            cmap=colormap,
            aspect='auto'
        )
        ax.set_title(label)
        ims.append(im)
    return ims[0]  # Retourne le premier im pour la colorbar


def plot_overlay(ax, x, y, field_datas, labels, colormap='plasma', flipy=False):
    """
    Trace plusieurs champs/simulations superposés avec transparence.
    Args:
        ax: Axe matplotlib.
        x, y: Coordonnées spatiales.
        field_datas: Liste des données à tracer.
        labels: Liste des labels pour la légende.
        colormap: Colormap de base (d'autres seront dérivées).
        flipy: Si True, inverse l'axe y.
    """
    cmaps = [colormap, 'viridis', 'inferno', 'magma', 'cividis', 'plasma', 'coolwarm']
    for i, (data, label) in enumerate(zip(field_datas, labels)):
        im = ax.imshow(
            np.flipud(data) if flipy else data,
            extent=[x.min(), x.max(), y.min(), y.max()],
            origin='lower',
            cmap=cmaps[i % len(cmaps)],
            aspect='auto',
            alpha=0.7,
            label=label
        )
    ax.legend(loc='upper right')
    return im


def plot_multi_field_side_by_side(axes, x, y, field_datas, field_names, colormap='plasma', flipy=False):
    """
    Trace plusieurs champs d'une même simulation côte à côte.
    """
    ims = []
    for ax, data, name in zip(axes, field_datas, field_names):
        im = ax.imshow(
            np.flipud(data) if flipy else data,
            extent=[x.min(), x.max(), y.min(), y.max()],
            origin='lower',
            cmap=colormap,
            aspect='auto'
        )
        ax.set_title(f"{name} ({latexify.get(name, name)})")
        ims.append(im)
    return ims[0]  # Retourne le premier im pour la colorbar


def plot_multi_field_overlay(ax, x, y, field_datas, field_names, colormap='plasma', flipy=False):
    """
    Trace plusieurs champs d'une même simulation superposés.
    """
    cmaps = [colormap, 'viridis', 'inferno', 'magma', 'cividis', 'plasma', 'coolwarm']
    for i, (data, name) in enumerate(zip(field_datas, field_names)):
        im = ax.imshow(
            np.flipud(data) if flipy else data,
            extent=[x.min(), x.max(), y.min(), y.max()],
            origin='lower',
            cmap=cmaps[i % len(cmaps)],
            aspect='auto',
            alpha=0.7,
            label=f"{name} ({latexify.get(name, name)})"
        )
    ax.legend(loc='upper right')
    return im
