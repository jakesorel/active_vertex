import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from joblib import Parallel, delayed
from voronoi_model.plotting_functions import *
plt.rcParams.update({'pdf.fonttype': 42})

N = 12
rep = 12
# p0_range = np.linspace(3.5, 4, N)
# v0_range = np.linspace(5e-3, 1e-1, N)
# beta_range = np.linspace(0, 0.3)
v0_range = np.linspace(5e-3, 1e-1, N)
beta_range = np.logspace(-3, -1, N)
rep_range = np.arange(rep)
VV, BB,RR = np.meshgrid(v0_range, beta_range,rep_range, indexing="ij")
ID_mat = np.arange(N**2).astype(int).reshape(N,N)
ID_mat = np.stack([ID_mat for i in range(rep)],axis=2)

inputs = np.array([ID_mat.ravel(),RR.ravel()]).T


def get_n_islands(X):
    Id, Rep = X
    try:
        FILE = np.load("from_sorted/analysis/%d_%d.npz" % (Id,Rep))
        return FILE["n_islands"]
    except FileNotFoundError:
        return np.nan

num_cores = multiprocessing.cpu_count()
n_islands = Parallel(n_jobs=num_cores)(delayed(get_n_islands)(inputt) for inputt in inputs)
n_islands = np.array(n_islands).reshape(N,N,rep,2,100)

final_mean_n_islands = n_islands.sum(axis=3).mean(axis=2)[:,:,-1]

make_directory("paper_plots")
make_directory("paper_plots/Fig2")
vmax,vmin = 9.6,2

fig, ax = plt.subplots(figsize=(3,2.5))
extent, aspect = make_extent(v0_range,np.log10(beta_range))
ax.imshow(flipim(final_mean_n_islands),extent=extent,aspect=aspect,cmap=plt.cm.inferno,vmin=vmin,vmax=vmax)
sm = plt.cm.ScalarMappable(cmap=plt.cm.inferno, norm=plt.Normalize(vmax=vmax, vmin=vmin))
sm._A = []
cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.085, aspect=10, orientation="vertical")
cl.set_label(r"$N_{clust}$")
ax.set(xlabel=r"$v_0$",ylabel=r"$log_{10} \ \beta$")
fig.subplots_adjust(top=0.8, bottom=0.2, left=0.25, right=0.8)
fig.savefig("paper_plots/Fig2/N_clust_phase_diag.pdf",dpi=300)


def get_L_star(X):
    Id, Rep = X
    try:
        FILE = np.load("from_sorted/analysis/%d_%d.npz" % (Id,Rep))
        return FILE["L_star"]
    except FileNotFoundError:
        return np.nan


L_star = Parallel(n_jobs=num_cores)(delayed(get_L_star)(inputt) for inputt in inputs)
L_star = np.array(L_star).reshape(N,N,rep,100)
dL_star = L_star[:,:,:,-1] - L_star[:,:,:,0]

final_mean_L_star = L_star.mean(axis=2)[:,:,-1]
final_mean_dL_star = dL_star.mean(axis=2)

cmap = plt.cm.plasma
fig, ax = plt.subplots(figsize=(3,2.5))
extent, aspect = make_extent(v0_range,np.log10(beta_range))
ax.imshow(flipim(final_mean_L_star),extent=extent,aspect=aspect,cmap=cmap)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmax=final_mean_L_star.max(), vmin=final_mean_L_star.min()))
sm._A = []
cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.085, aspect=10, orientation="vertical")
cl.set_label(r"$L^*$")
ax.set(xlabel=r"$v_0$",ylabel=r"$log_{10} \ \beta$")
fig.subplots_adjust(top=0.8, bottom=0.2, left=0.25, right=0.8)
fig.savefig("paper_plots/Fig2/L_star_phase_diag.pdf",dpi=300)


cmap = plt.cm.plasma
fig, ax = plt.subplots(figsize=(3,2.5))
extent, aspect = make_extent(v0_range,np.log10(beta_range))
ax.imshow(flipim(final_mean_dL_star),extent=extent,aspect=aspect,cmap=cmap)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmax=final_mean_dL_star.max(), vmin=final_mean_dL_star.min()))
sm._A = []
cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.085, aspect=10, orientation="vertical")
cl.set_label(r"$\Delta L^*$")
ax.set(xlabel=r"$v_0$",ylabel=r"$log_{10} \ \beta$")
fig.subplots_adjust(top=0.8, bottom=0.2, left=0.25, right=0.8)
fig.savefig("paper_plots/Fig2/dL_star_phase_diag.pdf",dpi=300)




def get_mean_self(X):
    Id, Rep = X
    try:
        FILE = np.load("from_sorted/analysis/%d_%d.npz" % (Id,Rep))
        return FILE["mean_self"]
    except FileNotFoundError:
        return np.nan


mean_self = Parallel(n_jobs=num_cores)(delayed(get_mean_self)(inputt) for inputt in inputs)
mean_self = np.array(mean_self).reshape(N,N,rep,100)

dmean_self = mean_self[:,:,:,-1] - mean_self[:,:,:,0]

final_mean_dmean_self = dmean_self.mean(axis=2)
final_mean_mean_self = mean_self.mean(axis=2)[:,:,-1]

cmap = plt.cm.viridis
vmax = 0.85
vmin = 0.5

fig, ax = plt.subplots(figsize=(3,2.5))
extent, aspect = make_extent(v0_range,np.log10(beta_range))
ax.imshow(flipim(final_mean_mean_self),extent=extent,aspect=aspect,cmap=cmap,vmin=vmin,vmax=vmax)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmax=vmax,vmin=vmin))
sm._A = []
cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.085, aspect=10, orientation="vertical")
cl.set_label(r"$\phi_{self}$")
ax.set(xlabel=r"$v_0$",ylabel=r"$log_{10} \ \beta$")
fig.subplots_adjust(top=0.8, bottom=0.2, left=0.25, right=0.8)
fig.savefig("paper_plots/Fig2/mean_self_phase_diag.pdf",dpi=300)


cmap = plt.cm.viridis


fig, ax = plt.subplots(figsize=(3,2.5))
extent, aspect = make_extent(v0_range,np.log10(beta_range))
ax.imshow(flipim(final_mean_dmean_self),extent=extent,aspect=aspect,cmap=cmap)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmax=final_mean_dmean_self.max(), vmin=final_mean_dmean_self.min()))
sm._A = []
cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.085, aspect=10, orientation="vertical")
cl.set_label(r"$\Delta \phi_{self}$")
ax.set(xlabel=r"$v_0$",ylabel=r"$log_{10} \ \beta$")
fig.subplots_adjust(top=0.8, bottom=0.2, left=0.25, right=0.8)
fig.savefig("paper_plots/Fig2/dmean_self_phase_diag.pdf",dpi=300)


"""Time courses"""

tspan_sample = np.linspace(0,500,100)

cmap = plt.cm.plasma
cols = cmap(np.linspace(0,1,N))
fig, ax = plt.subplots(figsize=(3,2.5))
for i in range(N):
    ax.plot(tspan_sample,mean_self[5,i].mean(axis=0),color=cols[i])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmax=np.log10(beta_range.max()), vmin=np.log10(beta_range).min()))
sm._A = []
cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.085, aspect=10, orientation="vertical")
cl.set_label(r"$log_{10} \ \beta$")
ax.set(xlabel=r"$t$",ylabel=r"$\phi_{self}$")
fig.subplots_adjust(top=0.8, bottom=0.25, left=0.20, right=0.75)
fig.savefig("paper_plots/Fig1/mean_self_timecourse.pdf",dpi=300)


cmap = plt.cm.plasma
cols = cmap(np.linspace(0,1,N))
fig, ax = plt.subplots(figsize=(3,2.5))
for i in range(N):
    ax.plot(tspan_sample,n_islands[5,i].sum(axis=1).mean(axis=0),color=cols[i])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmax=np.log10(beta_range.max()), vmin=np.log10(beta_range).min()))
sm._A = []
cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.085, aspect=10, orientation="vertical")
cl.set_label(r"$log_{10} \ \beta$")
ax.set(xlabel=r"$t$",ylabel=r"$N_{clust}$")
fig.subplots_adjust(top=0.8, bottom=0.25, left=0.20, right=0.75)
fig.savefig("paper_plots/Fig1/n_islands_timecourse.pdf",dpi=300)


t0 = 10
t1 = -1

from scipy.stats import linregress

ms_grad = np.zeros((N,N,rep))
for i in range(N):
    for j in range(N):
        for k in range(rep):
            ms_grad[i,j,k] = linregress(tspan_sample[t0:t1],mean_self[i,j,k,t0:t1])[0]

fig, ax = plt.subplots()
ax.imshow(flipim(ms_grad.mean(axis=2)))
fig.show()



####################################
#variation in v0
####################################



N = 12
rep = 12
# p0_range = np.linspace(3.5, 4, N)
# v0_range = np.linspace(5e-3, 1e-1, N)
# beta_range = np.linspace(0, 0.3)
sv0_range = np.linspace(0, 0.2, N)
beta_range = np.logspace(-3, -1, N)
rep_range = np.arange(rep)
sVV, BB,RR = np.meshgrid(sv0_range, beta_range,rep_range, indexing="ij")
ID_mat = np.arange(N**2).astype(int).reshape(N,N)
ID_mat = np.stack([ID_mat for i in range(rep)],axis=2)
inputs = np.array([ID_mat.ravel(),RR.ravel()]).T

def get_n_islands(X):
    Id, Rep = X
    try:
        FILE = np.load("from_sorted_v0_vary/analysis/%d_%d.npz" % (Id,Rep))
        return FILE["mean_self"]
    except FileNotFoundError:
        return np.nan

num_cores = multiprocessing.cpu_count()
n_islands = Parallel(n_jobs=num_cores)(delayed(get_n_islands)(inputt) for inputt in inputs)
n_islands = np.array(n_islands).reshape(N,N,rep,100)
final_mean_n_islands = n_islands.mean(axis=2)[:,:,-1]



fig, ax = plt.subplots(figsize=(3,2.5))
extent, aspect = make_extent(sv0_range,np.log10(beta_range))
ax.imshow(flipim(final_mean_n_islands),extent=extent,aspect=aspect,cmap=plt.cm.inferno)#,vmin=vmin,vmax=vmax)
sm = plt.cm.ScalarMappable(cmap=plt.cm.inferno, norm=plt.Normalize(vmax=vmax, vmin=vmin))
sm._A = []
cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.085, aspect=10, orientation="vertical")
cl.set_label(r"$N_{clust}$")
ax.set(xlabel=r"$s_{v_0}$",ylabel=r"$log_{10} \ \beta$")
fig.subplots_adjust(top=0.8, bottom=0.2, left=0.25, right=0.8)
fig.show()
# fig.savefig("paper_plots/Fig2/N_clust_phase_diag.pdf",dpi=300)
