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
For figure 1. Extracts the mean self-self interaction length (check) for different parameter regimes. 

Characteristic examples across the three dimensions around a pivot point. 

"""

def normalise(x, xmin, xmax):
    return (x - xmin) / (xmax - xmin)

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


def get_mean_self(X):
    Id, Rep = X
    try:
        FILE = np.load("analysis/%d_%d.npz" % (Id,Rep))
        return FILE["mean_self"]
    except FileNotFoundError:
        return np.ones(100)*np.nan


inputs = np.array([ID_mat.ravel(),RR.ravel()]).T
inputs = inputs.astype(np.int64)
lazy_results = []
for inputt in inputs:
    lazy_result = dask.delayed(get_mean_self)(inputt)
    lazy_results.append(lazy_result)
out_meanself = dask.compute(*lazy_results)
out_meanself = np.array(out_meanself).reshape(RR.shape[0],RR.shape[1],RR.shape[2],RR.shape[3],100)

plt.imshow(out_meanself[:,:,6,0,40])
plt.show()

def rolling_mean(x,n):
    N = x.size
    X = np.empty((N-n,n))
    for i in range(n):
        X[:,i] = x[i:N-n+i]
    return X.mean(axis=1)

cols = plt.cm.viridis(np.linspace(0,1,10))
fig, ax = plt.subplots(figsize=(3.5,3))
for j in range(10):
    i = 9-j
    ax.plot(np.arange(0,500,5),out_meanself[i,4,4,:,:].mean(axis=-2),color=cols[i])

cmap = plt.cm.viridis(np.linspace(0,1,200))
mycmap = ListedColormap(cmap)
sm = plt.cm.ScalarMappable(cmap=mycmap)
sm._A = []
ntick = 5
cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.095, aspect=12, orientation="vertical", ticks=np.linspace(0, 1, ntick))
cl.set_label(r"$p_0$")
cl.ax.set_yticklabels(np.linspace(p0_range.min(),p0_range.max(),ntick))
ax.set(xlabel="Time",ylabel=r"$\langle \phi_{self} \rangle$",xlim=(0,500),ylim=(0.48,0.8))
fig.subplots_adjust(top=0.8, bottom=0.25, left=0.23, right=0.8)
fig.savefig("analysis_plots/p0 vs time v0=4,beta=4.pdf")


fig, ax = plt.subplots(figsize=(3.5,3))
for j in range(10):
    i = 9-j
    ax.plot(np.arange(0,500,5),out_meanself[6,i,4,:,:].mean(axis=-2),color=plt.cm.viridis(normalise(v0_range[i],0,v0_range.max())))

cmap = plt.cm.viridis(np.linspace(0,1,200))
mycmap = ListedColormap(cmap)
sm = plt.cm.ScalarMappable(cmap=mycmap)
sm._A = []
ntick = 6
cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.095, aspect=12, orientation="vertical", ticks=np.linspace(0, 1, ntick))
cl.set_label(r"$v_0$")
cl.ax.set_yticklabels(np.round(np.linspace(0,v0_range.max(),ntick),2))
ax.set(xlabel="Time",ylabel=r"$\langle \phi_{self} \rangle$",xlim=(0,500),ylim=(0.48,0.8))
fig.subplots_adjust(top=0.8, bottom=0.25, left=0.23, right=0.8)
fig.savefig("analysis_plots/v0 vs time p0=6,beta=4.pdf")


fig, ax = plt.subplots(figsize=(3.5,3))
for j in range(10):
    i = 9-j
    ax.plot(np.arange(0,500,5),out_meanself[6,4,i,:,:].mean(axis=-2),color=cols[i])

cmap = plt.cm.viridis(np.linspace(0,1,200))
mycmap = ListedColormap(cmap)
sm = plt.cm.ScalarMappable(cmap=mycmap)
sm._A = []
ntick = 6
cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.095, aspect=12, orientation="vertical", ticks=np.linspace(0, 1, ntick))
cl.set_label(r"$\beta$")
cl.ax.set_yticklabels(np.round(np.linspace(np.log10(beta_range.min()),np.log10(beta_range.max()),ntick),2))
ax.set(xlabel="Time",ylabel=r"$\langle \phi_{self} \rangle$",xlim=(0,500),ylim=(0.48,0.8))
fig.subplots_adjust(top=0.8, bottom=0.25, left=0.23, right=0.8)
fig.savefig("analysis_plots/beta vs time p0=6,v0=4.pdf")
