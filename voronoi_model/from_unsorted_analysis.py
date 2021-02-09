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
        FILE = np.load("from_unsorted/analysis/%d_%d.npz" % (Id,Rep))
        return FILE["n_islands"]
    except FileNotFoundError:
        return np.nan

num_cores = multiprocessing.cpu_count()
n_islands = Parallel(n_jobs=num_cores)(delayed(get_n_islands)(inputt) for inputt in inputs)
n_islands = np.array(n_islands).reshape(N,N,rep,2,100)

final_mean_n_islands = n_islands.sum(axis=3).mean(axis=2)[:,:,-1]

vmax,vmin = 9.6,2
make_directory("paper_plots")
make_directory("paper_plots/Fig1")

fig, ax = plt.subplots(figsize=(3,2.5))
extent, aspect = make_extent(v0_range,np.log10(beta_range))
ax.imshow(flipim(final_mean_n_islands),extent=extent,aspect=aspect,cmap=plt.cm.inferno,vmin=vmin,vmax=vmax)
sm = plt.cm.ScalarMappable(cmap=plt.cm.inferno, norm=plt.Normalize(vmax=vmax, vmin=vmin))
sm._A = []
cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.085, aspect=10, orientation="vertical")
cl.set_label(r"$N_{clust}$")
ax.set(xlabel=r"$v_0$",ylabel=r"$log_{10} \ \beta$")
fig.subplots_adjust(top=0.8, bottom=0.2, left=0.25, right=0.8)
fig.savefig("paper_plots/Fig1/N_clust_phase_diag.pdf",dpi=300)


def get_L_star(X):
    Id, Rep = X
    try:
        FILE = np.load("from_unsorted/analysis/%d_%d.npz" % (Id,Rep))
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
fig.savefig("paper_plots/Fig1/L_star_phase_diag.pdf",dpi=300)


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
fig.savefig("paper_plots/Fig1/dL_star_phase_diag.pdf",dpi=300)




def get_mean_self(X):
    Id, Rep = X
    try:
        FILE = np.load("from_unsorted/analysis/%d_%d.npz" % (Id,Rep))
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
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmax=vmax, vmin=vmin))
sm._A = []
cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.085, aspect=10, orientation="vertical")
cl.set_label(r"$\phi_{self}$")
ax.set(xlabel=r"$v_0$",ylabel=r"$log_{10} \ \beta$")
fig.subplots_adjust(top=0.8, bottom=0.2, left=0.25, right=0.8)
fig.savefig("paper_plots/Fig1/mean_self_phase_diag.pdf",dpi=300)


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
fig.savefig("paper_plots/Fig1/dmean_self_phase_diag.pdf",dpi=300)


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
    ax.plot(tspan_sample,n_islands[7,i].sum(axis=1).mean(axis=0),color=cols[i])
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


dL_star = (L_star.T - L_star[:,:,:,0].T).T

cmap = plt.cm.plasma
cols = cmap(np.linspace(0,1,N))
fig, ax = plt.subplots(figsize=(3,2.5))
for i in range(N):
    ax.plot(tspan_sample,dL_star[5,i,:].mean(axis=0),color=cols[i])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmax=np.log10(beta_range.max()), vmin=np.log10(beta_range).min()))
sm._A = []
cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.085, aspect=10, orientation="vertical")
cl.set_label(r"$log_{10} \ \beta$")
ax.set(xlabel=r"$t$",ylabel=r"$\phi_{self}$")
fig.subplots_adjust(top=0.8, bottom=0.25, left=0.20, right=0.75)
fig.savefig("paper_plots/Fig1/L_star_timecourse.pdf",dpi=300)





def get_n_het_swap_tot(X):
    Id, Rep = X
    try:
        FILE = np.load("from_unsorted/het_swaps/%d_%d.npz" % (Id,Rep))
        return FILE["n_het_swap_tot"]
    except FileNotFoundError:
        return np.nan




def get_n_het_swap_tot(X,t0=2000):
    Id, Rep = X
    try:
        FILE = np.load("from_unsorted/het_swaps/%d_%d.npz" % (Id,Rep))
        n_het_swap,changed_t = FILE["n_het_swap"], FILE["changed_t"]
        return n_het_swap[changed_t>t0].sum()
    except FileNotFoundError:
        return np.nan




num_cores = multiprocessing.cpu_count()
n_het_swap_tot = Parallel(n_jobs=num_cores)(delayed(get_n_het_swap_tot)(inputt) for inputt in inputs)
n_het_swap_tot = np.array(n_het_swap_tot).reshape(N,N,rep)


fig, ax = plt.subplots()
ax.imshow(flipim(n_het_swap_tot.mean(axis=2)))
fig.show()

mean_swap_rate = n_het_swap_tot.mean(axis=2)/((20000-2000)*(0.025))

vmin = mean_swap_rate.min()
vmax = 0.005#np.percentile(n_het_swap_tot.mean(axis=2),50)

cmap = plt.cm.Reds
fig, ax = plt.subplots(figsize=(3,2.5))
extent, aspect = make_extent(v0_range,np.log10(beta_range))
ax.imshow(flipim(mean_swap_rate),extent=extent,aspect=aspect,cmap=cmap,vmax = vmax)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmax=vmax, vmin=vmin))
sm._A = []
cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.085, aspect=10, orientation="vertical")
cl.formatter.set_powerlimits((0, 0))
cl.set_label(r"$\frac{dN_{swap}}{dt}$")
ax.set(xlabel=r"$v_0$",ylabel=r"$log_{10} \ \beta$")
fig.subplots_adjust(top=0.8, bottom=0.2, left=0.25, right=0.8)
# fig.show()
fig.savefig("paper_plots/Fig1/n_het_swap.pdf",dpi=300)


vals = np.log10(mean_swap_rate).copy()
vals[np.isinf(vals)] = np.nan

vmin = np.nanmin(vals)
vmax = np.nanpercentile(vals,90)
cmap = plt.cm.Reds
fig, ax = plt.subplots(figsize=(3.15,2.5))
extent, aspect = make_extent(v0_range,np.log10(beta_range))
ax.imshow(flipim(np.log10(mean_swap_rate)),extent=extent,aspect=aspect,cmap=cmap,vmin=vmin,vmax=vmax)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmax=vmax, vmin=vmin))
sm._A = []
cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.085, aspect=10, orientation="vertical")
cl.formatter.set_powerlimits((0, 0))
cl.set_label(r"$log_{10} \left(\frac{dN_{swap}}{dt}\right)$")
ax.set(xlabel=r"$v_0$",ylabel=r"$log_{10} \ \beta$")
fig.subplots_adjust(top=0.8, bottom=0.2, left=0.25, right=0.75)
# fig.show()
fig.savefig("paper_plots/Fig1/n_het_swap_log.pdf",dpi=300)
