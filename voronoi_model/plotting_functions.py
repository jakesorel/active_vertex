import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
import os
def make_colormap_white(col):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).

    https://stackoverflow.com/questions/16834861/create-own-colormap-using-matplotlib-and-plot-color-scale
    """
    c = mcolors.ColorConverter().to_rgb
    seq = [c("white"),c(col)]
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)

def make_colormap_alpha(col):
    rgb_col = np.array(list(mcolors.ColorConverter().to_rgb(col)))
    N = 256
    rgba_map = np.zeros((N,4))
    rgba_map[:,:3] = rgb_col
    rgba_map[:,3] = np.linspace(0,1,N)
    cmap = ListedColormap(rgba_map)
    return cmap

def make_directory(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def flipim(x):
    return np.flip(x.T,axis=0)

def make_extent(x_range,y_range,xscale="linear",yscale="linear"):
    if xscale == "log":
        x_range = np.log10(x_range)
    if yscale == "log":
        y_range = np.log10(y_range)
    extent = [x_range[0],x_range[-1]+x_range[1]-x_range[0],y_range[0],y_range[-1]+y_range[1]-y_range[0]]
    aspect = (extent[1]-extent[0])/(extent[3]-extent[2])
    return extent,aspect