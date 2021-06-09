#!/usr/bin/env python3
"""plot.py: Useful functions for convenient plotting."""
__author__ = "Joha Park"
__all__ = ["custom_cmap", "sequential_palette"]

from matplotlib.colors import ColorConverter, LinearSegmentedColormap
import seaborn as sns
import numpy as np

def sequential_palette(color, n_colors=6, mode='light', **kwargs):
    if mode == 'light': return sns.light_palette(color, n_colors, **kwargs)
    elif mode == 'dark': return sns.dark_palette(color, n_colors, **kwargs)
    else: raise ValueError("Only 'light' or 'dark' modes are accepted")

def custom_cmap(colors, cmap_name="newmap", nspace=11, linear=True):
    to_rgb = ColorConverter().to_rgb

    if (type(colors) == str) or (len(colors) == 1):
        colors = [colors, "white"]

    ncolors = len(colors)
    sidx = list(map(int, map(np.around, np.linspace(0, nspace-1, num=ncolors))))
    intervals = np.linspace(0, 1.0, num=nspace)
    rgb = ["red", "green", "blue"]
    cdict = {e:None for e in rgb}


    for element, components in zip(rgb, zip(*[to_rgb(c) for c in colors])):
        intensities = [components[0]]

        for i, value in enumerate(components):
            if i + 1 == len(components): break
            v1, v2 = components[i:i+2]
            intensities += list(np.linspace(v1, v2, num=sidx[i+1] - sidx[i] + 1))[1:]

        cdict[element] =  list(zip(intervals, intensities, intensities))

    return LinearSegmentedColormap(cmap_name, cdict)
