import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from joblib import Parallel, delayed
from voronoi_model.plotting_functions import *
plt.rcParams.update({'pdf.fonttype': 42})
import seaborn as sb
import pandas as pd

N = 12
rep = 24
# p0_range = np.linspace(3.5, 4, N)
# v0_range = np.linspace(5e-3, 1e-1, N)
# beta_range = np.linspace(0, 0.3)
v0_range = np.linspace(5e-3, 1e-1, N)
beta_range = np.logspace(-3, -1, N)
rep_range = np.arange(rep)
BB,RR = np.meshgrid(beta_range,rep_range, indexing="ij")
ID_mat = np.arange(N).astype(int)
ID_mat = np.stack([ID_mat for i in range(rep)],axis=1)

inputs = np.array([ID_mat.ravel(),RR.ravel()]).T


def get_n_islands(X):
    Id, Rep = X
    try:
        FILE = np.load("from_unsorted_beta_only/analysis/%d_%d.npz" % (Id,Rep))
        return FILE["n_islands"]
    except FileNotFoundError:
        return np.nan

num_cores = multiprocessing.cpu_count()
n_islands = Parallel(n_jobs=num_cores)(delayed(get_n_islands)(inputt) for inputt in inputs)
n_islands = np.array(n_islands).reshape(N,rep,2,100)

n_islands_fin = n_islands.sum(axis=2)[:,:,-1]
n_islands_init = n_islands.sum(axis=2)[:,:,0]
LQ = np.percentile(n_islands_fin,25,axis=1)
UQ = np.percentile(n_islands_fin,75,axis=1)
mean = np.mean(n_islands_fin,axis=1)

make_directory("paper_plots")
make_directory("paper_plots/Fig1")
mult = 4
fig, ax = plt.subplots(figsize=(3,2.5))
for i in range(N):
    bincount = np.bincount(n_islands_fin[i])
    # for j in range(bincount.size):
    #     ax.add_patch(plt.Circle((np.log10(beta_range)[i], j), bincount[j]*mult, color='r'))
    ax.scatter(np.repeat(np.log10(beta_range)[i],bincount.size),np.arange(np.amax(n_islands_fin[i]+1)),s=bincount*mult,color="grey")
ax.plot(np.log10(beta_range),mean,color="k")
# ax.plot(np.linspace(-3.5,-0.5,N),2*np.ones_like(beta_range),color="grey",linestyle="--")
ax.set(xlabel=r"$log_{10} \ \beta$",ylabel=r"$\langle N_{clust} \rangle$",ylim=(1,10.7),xlim=(-3.1,-0.9))
# sb.lineplot(data=df,x="logbeta",y="n_islands",ax = ax)
fig.subplots_adjust(top=0.8, bottom=0.2, left=0.25, right=0.8)
fig.savefig("paper_plots/Fig1/N_clust_beta_only.pdf",dpi=300)

fig, ax = plt.subplots(figsize=(3,2.5))
n = 6
sizes = np.linspace(0,24,n)
ax.scatter(np.repeat(-2, n), 2+np.arange(0,n)*1.5,
           s=sizes * mult, color="grey")
ax.set(xlabel=r"$log_{10} \ \beta$",ylabel=r"$\langle N_{clust} \rangle$",ylim=(1,10.7),xlim=(-3.1,-0.9))
# sb.lineplot(data=df,x="logbeta",y="n_islands",ax = ax)
fig.subplots_adjust(top=0.8, bottom=0.2, left=0.25, right=0.8)
fig.savefig("paper_plots/Fig1/N_clust_beta_only_key.pdf",dpi=300)


def get_phi_self(X):
    Id, Rep = X
    try:
        FILE = np.load("from_unsorted_beta_only/analysis/%d_%d.npz" % (Id,Rep))
        return FILE["mean_self"]
    except FileNotFoundError:
        return np.nan

num_cores = multiprocessing.cpu_count()
L = Parallel(n_jobs=num_cores)(delayed(get_phi_self)(inputt) for inputt in inputs)
L = np.array(L).reshape(N,rep,100)

L_fin = L[:,:,-1]
mean = np.mean(L_fin,axis=1)

make_directory("paper_plots")
make_directory("paper_plots/Fig1")
mult = 4
fig, ax = plt.subplots(figsize=(3,2.5))
df = pd.DataFrame({"b":np.log10(BB).ravel(),"L":L_fin.ravel()})
sb.lineplot(data = df,x="b",y="L",ax=ax,ci="sd",color="black")
ax.set(xlabel=r"$log_{10} \ \beta$",ylabel=r"$\langle \phi_{self} \rangle$",xlim=(-3,-1))
fig.subplots_adjust(top=0.8, bottom=0.2, left=0.25, right=0.8)
fig.savefig("paper_plots/Fig1/phi_beta_only.pdf",dpi=300)

fig, ax = plt.subplots(figsize=(3,2.5))
n = 6
sizes = np.linspace(0,24,n)
ax.scatter(np.repeat(-2, n), 2+np.arange(0,n)*1.5,
           s=sizes * mult, color="grey")
ax.set(xlabel=r"$log_{10} \ \beta$",ylabel=r"$\langle N_{clust} \rangle$",ylim=(1,10.7),xlim=(-3.1,-0.9))
# sb.lineplot(data=df,x="logbeta",y="n_islands",ax = ax)
fig.subplots_adjust(top=0.8, bottom=0.2, left=0.25, right=0.8)
fig.savefig("paper_plots/Fig1/N_clust_beta_only_key.pdf",dpi=300)





def get_autocorr(X):
    Id, Rep = X
    try:
        FILE = np.load("from_unsorted_beta_only/autocorr/%d_%d.npz" % (Id,Rep))
        return FILE["ds"],FILE["radialprofile"]
    except FileNotFoundError:
        return np.nan


autocorr = Parallel(n_jobs=num_cores)(delayed(get_autocorr)(inputt) for inputt in inputs)
autocorr = np.array(autocorr).reshape(N,rep,2,64)

fig, ax = plt.subplots()
cols = plt.cm.plasma(np.linspace(0,1,N))
for i in [0,7,-1]:
    ax.plot(autocorr[i,:,0].mean(axis=0),autocorr[i,:,1].mean(axis=0),color=cols[i])
fig.show()

from scipy.interpolate import UnivariateSpline
x = autocorr[0,0,0]
fig, ax = plt.subplots()
cols = plt.cm.plasma(np.linspace(0,1,N))
for i in [0,6,-1]:
    autoc = autocorr[i,12,1]
    spl = UnivariateSpline(x[~np.isnan(autoc)],autoc[~np.isnan(autoc)],k = 5)
    ax.plot(x,autoc,color=cols[i],alpha=0.5)
    # ax.plot(x,spl(x),color=cols[i])
fig.show()



