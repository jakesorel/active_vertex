import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"
from scipy.interpolate import bisplrep,bisplev
import dask
from dask.distributed import Client
from scipy.interpolate import bisplrep,bisplev
from matplotlib.colors import ListedColormap


"""
For Figure 2. 

2D plot of n_islands across parameters. from sorted state
"""

###1. Load the data

n_slurm_tasks = 8
client = Client(threads_per_worker=1, n_workers=n_slurm_tasks, memory_limit="1GB")
N = 10
rep = 8
p0_range = np.linspace(3.5, 4, N)
v0_range = np.linspace(5e-3, 1e-1, N)
beta_range = np.logspace(-3, -1, N)
rep_range = np.arange(rep)
PP, VV, BB,RR = np.meshgrid(p0_range, v0_range, beta_range,rep_range, indexing="ij")
ID_mat = np.arange(N**3).astype(int).reshape(N,N,N)
ID_mat = np.stack([ID_mat for i in range(rep)],axis=3)


def get_n_islands(X):
    Id, Rep = X
    try:
        FILE = np.load("analysis_fsorted/%d_%d.npz" % (Id,Rep))
        return FILE["n_islands"]
    except FileNotFoundError:
        return np.ones(100)*np.nan


inputs = np.array([ID_mat.ravel(),RR.ravel()]).T
inputs = inputs.astype(np.int64)
lazy_results = []
for inputt in inputs:
    lazy_result = dask.delayed(get_n_islands)(inputt)
    lazy_results.append(lazy_result)
out_nislands = dask.compute(*lazy_results)
out_nislands = np.array(out_nislands).reshape(RR.shape[0],RR.shape[1],RR.shape[2],RR.shape[3],2,100)
n_islands_tot = out_nislands.sum(axis=-2)

##2. Calculate the mean number of islands

n_islands_tot_mean = n_islands_tot[:,:,:,:,-1].mean(axis=-1)


def normalise(x, xmin, xmax):
    return (x - xmin) / (xmax - xmin)


for iv in range(10):
    x,y,z = PP[:,iv,:,0].ravel(),np.log10(BB[:,iv,:,0]).ravel(),n_islands_tot_mean[:,iv,:].ravel()
    ni_min,ni_max = np.percentile(n_islands_tot_mean,5),np.percentile(n_islands_tot_mean,95)
    ni_mid = (ni_max+ni_min)/2

    sort_mask = z<ni_mid

    vmin,vmax = 2,5.5
    vmax2 = vmax*1.2
    cmap = plt.cm.inferno(normalise(np.linspace(vmin,vmax,100),z.min(),vmax2))
    mycmap = ListedColormap(cmap)

    fig, ax = plt.subplots(figsize=(3.5,3))
    ax.scatter(x[sort_mask],y[sort_mask],c=plt.cm.inferno(normalise(z[sort_mask],vmin,vmax2)))
    ax.scatter(x[~sort_mask],y[~sort_mask],c=plt.cm.inferno(normalise(z[~sort_mask],vmin,vmax2)),marker=",")
    sm = plt.cm.ScalarMappable(cmap=mycmap)
    sm._A = []
    cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.095, aspect=12, orientation="vertical",ticks=np.linspace(0,1,6))
    cl.set_label(r"$\langle n_{clust} \rangle$")
    cl.ax.set_yticklabels(np.round(np.linspace(vmin,vmax,6),1))
    ax.set(xlabel=r"$p_0$",ylabel=r"$log_{10} \ \beta$")
    ax.set_title(r"$v_0 = %.3f$"%v0_range[iv])
    fig.subplots_adjust(top=0.8, bottom=0.25, left=0.23, right=0.8)
    fig.savefig("analysis_plots/fsorted p0 beta v0=%.3f.pdf"%v0_range[iv],dpi=300)

    plt.close("all")

