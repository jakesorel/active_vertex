import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from joblib import Parallel, delayed
from voronoi_model.plotting_functions import *
plt.rcParams.update({'pdf.fonttype': 42})


@jit(nopython=True, cache=True)
def _beta_t(beta_max, beta_min, tau, t):
    if t <= tau:
        return beta_min + (beta_max - beta_min) * t / tau
    else:
        return beta_max

t_span = np.linspace(0,499,200)
fig, ax = plt.subplots(figsize=(2.5,1.5))
tau_range = np.linspace(50, 500, 5)
cmap = plt.cm.cividis(np.linspace(0,1,5))
for i, tau in enumerate(tau_range):
    ax.plot(t_span,np.log10(np.array([_beta_t(10**-1,10**-2.5,tau,t) for t in t_span])),color=cmap[i])
ax.set(xlabel="t",ylabel=r"$\beta(t)$",ylim=(-2.6,-0.9))
sm = plt.cm.ScalarMappable(cmap=plt.cm.cividis, norm=plt.Normalize(vmax=tau_range.max(), vmin=tau_range.min()))
sm._A = []
cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.11, aspect=10, orientation="vertical")
cl.set_label(r"$\tau$")
fig.subplots_adjust(top=0.8, bottom=0.3, left=0.25, right=0.8)
# ax.set(yscale="log")
fig.show()
fig.savefig("paper_plots/Fig5/beta_t_tau_log.pdf",dpi=300)

fig, ax = plt.subplots(figsize=(2.5,1.5))
beta_max_range = np.logspace(-2.5, -1, 5)
cmap = plt.cm.plasma((np.linspace(-2.5,-1,5) + 3)/2)
for i, beta_max in enumerate(beta_max_range):
    ax.plot(t_span,np.log10(np.array([_beta_t(beta_max,10**-2.5,300,t) for t in t_span])),color=cmap[i])
ax.set(xlabel="t",ylabel=r"$\beta(t)$",ylim=(-2.6,-0.9))
sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(vmax=-1, vmin=-3))
sm._A = []
cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.11, aspect=10, orientation="vertical")
cl.set_label(r"$\beta$")
fig.subplots_adjust(top=0.8, bottom=0.3, left=0.25, right=0.8)
fig.show()
fig.savefig("paper_plots/Fig5/beta_t_beta_log.pdf",dpi=300)


N = 12
rep = 12
# p0_range = np.linspace(3.5, 4, N)
# v0_range = np.linspace(5e-3, 1e-1, N)
# beta_range = np.linspace(0, 0.3)
tau_range = np.linspace(50, 500, N)
beta_max_range = np.logspace(-2.5, -1, N)

rep_range = np.arange(rep)
TT, BB,RR = np.meshgrid(tau_range, beta_max_range,rep_range, indexing="ij")
ID_mat = np.arange(N**2).astype(int).reshape(N,N)
ID_mat = np.stack([ID_mat for i in range(rep)],axis=2)

inputs = np.array([ID_mat.ravel(),RR.ravel()]).T

BBt0,RRt0 = np.meshgrid(beta_max_range,rep_range, indexing="ij")
ID_matt0 = np.stack([np.arange(N) for i in range(rep)],axis=1)
inputs_tau0 = np.array([ID_matt0.ravel(),RRt0.ravel()]).T


def get_n_islands(X):
    Id, Rep = X
    try:
        FILE = np.load("from_unsorted_control/analysis/%d_%d.npz" % (Id,Rep))
        return FILE["n_islands"]
    except FileNotFoundError:
        return np.nan

def get_n_islands_tau0(X):
    Id, Rep = X
    try:
        FILE = np.load("from_unsorted_control/analysis_tau0/%d_%d.npz" % (Id,Rep))
        return FILE["n_islands"]
    except FileNotFoundError:
        return np.nan
num_cores = multiprocessing.cpu_count()
n_islands = Parallel(n_jobs=num_cores)(delayed(get_n_islands)(inputt) for inputt in inputs)
n_islands = np.array(n_islands).reshape(N,N,rep,2,100)
n_islands_tau0 = Parallel(n_jobs=num_cores)(delayed(get_n_islands_tau0)(inputt) for inputt in inputs_tau0)
n_islands_tau0 = np.array(n_islands_tau0).reshape(N,rep,2,100)


final_mean_n_islands = n_islands.sum(axis=3).mean(axis=2)[:,:,-1]

vmax,vmin = 7,final_mean_n_islands.min()
make_directory("paper_plots")
make_directory("paper_plots/Fig5")

fig, ax = plt.subplots(figsize=(3,2.5))
extent, aspect = make_extent(tau_range,np.log10(beta_max_range))
ax.imshow(flipim(final_mean_n_islands),extent=extent,aspect=aspect,cmap=plt.cm.inferno,vmin=vmin,vmax=vmax)
sm = plt.cm.ScalarMappable(cmap=plt.cm.inferno, norm=plt.Normalize(vmax=vmax, vmin=vmin))
sm._A = []
cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.085, aspect=10, orientation="vertical")
cl.set_label(r"$N_{clust}$")
ax.set(xlabel=r"$\tau$",ylabel=r"$log_{10} \ \beta_{max}$")
fig.subplots_adjust(top=0.8, bottom=0.2, left=0.25, right=0.8)
fig.savefig("paper_plots/Fig1/N_clust_phase_diag.pdf",dpi=300)
fig.show()




vmax,vmin = 9,2

def make_N_plot(ti,ax,i,i_max):
    final_mean_n_islands = n_islands.sum(axis=3).mean(axis=2)[:, :, ti]


    extent, aspect = make_extent(tau_range,np.log10(beta_max_range))
    ax.imshow(flipim(final_mean_n_islands),extent=extent,aspect=aspect,cmap=plt.cm.inferno,vmin=vmin,vmax=vmax)
    if i == i_max:
        sm = plt.cm.ScalarMappable(cmap=plt.cm.inferno, norm=plt.Normalize(vmax=vmax, vmin=vmin))
        sm._A = []
        cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.085, aspect=10, orientation="vertical")
        cl.set_label(r"$N_{clust}$")
    if i == 0:
        ax.set(xlabel=r"$\tau$",ylabel=r"$log_{10} \ \beta_{max}$")
    else:
        ax.set(xlabel=r"$\tau$")
        ax.set_yticks([])



T_range = np.linspace(0,99,5).astype(np.int64)
nt = T_range.size
fig, ax = plt.subplots(1,nt)
for i, ti in enumerate(T_range):
    make_N_plot(ti,ax[i],i,nt-1)
# fig.subplots_adjust(top=0.8, bottom=0.2, left=0.25, right=0.8)
# fig.savefig("paper_plots/Fig1/N_clust_phase_diag.pdf",dpi=300)
fig.savefig("paper_plots/Fig5/N_clust_vs_time.pdf",dpi=300)


vmax = 5
def make_N_plot_T(ax,i,i_max,T_range):
    final_mean_n_islands = np.row_stack([n_islands.sum(axis=3).mean(axis=2)[:, i, t] for t in T_range])


    extent, aspect = make_extent(T_range*5,tau_range)
    ax.imshow(flipim(final_mean_n_islands),extent=extent,aspect=aspect,cmap=plt.cm.inferno,vmin=vmin,vmax=vmax)
    ax.set_title("lgb = %.1f"%np.log10(beta_max_range[i]),fontsize=2)
    if i == i_max:
        sm = plt.cm.ScalarMappable(cmap=plt.cm.inferno, norm=plt.Normalize(vmax=vmax, vmin=vmin))
        sm._A = []
        cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.085, aspect=10, orientation="vertical")
        cl.set_label(r"$N_{clust}$")
    if i == 0:
        ax.set(xlabel=r"$T$",ylabel=r"$\tau$")
    else:
        ax.set(xlabel=r"$T$")
        ax.set_yticks([])


T_range = np.linspace(0,99,20).astype(np.int64)
nt = T_range.size
fig, ax = plt.subplots(1,N)
for i in range(N):
    make_N_plot_T(ax[i],i,N-1,T_range)
# fig.subplots_adjust(top=0.8, bottom=0.2, left=0.25, right=0.8)
# fig.savefig("paper_plots/Fig1/N_clust_phase_diag.pdf",dpi=300)
fig.savefig("paper_plots/Fig5/N_clust_vs_time_vs_tau.pdf",dpi=300)

nis = n_islands.sum(axis=-2).mean(axis=-2)
nist0 = n_islands_tau0.sum(axis=-2).mean(axis=-2)

fig, ax = plt.subplots(1,6,figsize=(12,2),sharey=True)
cmap = plt.cm.cividis(np.linspace(0,1,12))
ax = ax.ravel()
for i in range(12):
    for ji, j in enumerate(np.arange(1,12,2)):
        ax[ji].plot(np.linspace(0,499,100).astype(np.int64),nis[i,j],color=cmap[i],linewidth=1)
        ax[ji].plot(np.linspace(0,499,100).astype(np.int64),nist0[j],color="darkred",linewidth=1.3,alpha=0.5)
        ax[ji].set_title("%.1f"%np.log10(beta_max_range[j]))
        ax[ji].set(xlabel="t")
ax[0].set(ylabel=r"$N_{clust}$")
sm = plt.cm.ScalarMappable(cmap=plt.cm.cividis, norm=plt.Normalize(vmax=tau_range.max(), vmin=tau_range.min()))
sm._A = []
cl = plt.colorbar(sm, ax=ax[-1], pad=0.05, fraction=0.11, aspect=10, orientation="vertical")
cl.set_label(r"$\tau$")
fig.subplots_adjust(top=0.8, bottom=0.3, left=0.25, right=0.8)
fig.show()
fig.savefig("paper_plots/Fig5/line_plots_N_clust_scalebar.pdf",dpi=300)


fig, ax = plt.subplots(figsize=(2.5,2),sharey=True)
cmap = plt.cm.cividis(np.linspace(0,1,12))
ji,j = 5,11
for i in range(12):
    ax.plot(np.linspace(0,499,100).astype(np.int64),nis[i,j],color=cmap[i],linewidth=1)
    ax.plot(np.linspace(0,499,100).astype(np.int64),nist0[j],color="darkred",linewidth=1.3,alpha=0.5)
    ax.set_title("%.1f"%np.log10(beta_max_range[j]))
    ax.set(xlabel="t")
ax.set(ylabel=r"$N_{clust}$")
sm = plt.cm.ScalarMappable(cmap=plt.cm.cividis, norm=plt.Normalize(vmax=tau_range.max(), vmin=tau_range.min()))
sm._A = []
cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.11, aspect=10, orientation="vertical")
cl.set_label(r"$\tau$")
fig.subplots_adjust(top=0.8, bottom=0.3, left=0.25, right=0.8)
fig.show()
fig.savefig("paper_plots/Fig5/line_plots_N_clust_scalebar_single.pdf",dpi=300)




df = pd.DataFrame({"N_clust":n_islands.sum(axis=-2)[:,:,:,-1].ravel(),"beta":BB.ravel(),"tau":TT.ravel(),"rep":RR.ravel(),"logbeta":np.log10(BB.ravel())})
df = pd.concat([df,pd.DataFrame({"N_clust":n_islands_tau0.sum(axis=-2)[:,:,-1].ravel(),"beta":BBt0.ravel(),"tau":np.zeros_like(BBt0).ravel(),"rep":RRt0.ravel(),"logbeta":np.log10(BBt0.ravel())})])

import seaborn as sb

fig, ax = plt.subplots(figsize=(5,2))
sb.lineplot(data=df.loc[df["tau"]==0],x="logbeta",y="N_clust",color=plt.cm.cividis(0.1),ax=ax,label=r"$\tau$"" =0")
sb.lineplot(data=df.loc[df["tau"]==500],x="logbeta",y="N_clust",color=plt.cm.plasma(0.85),ax=ax,label=r"$\tau$"" =500")
ax.set(xlabel=r"$log_{10} \ \beta_{max}$",ylabel=r"$N_{clust}$")
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),ncol=2)

fig.subplots_adjust(top=0.8, bottom=0.3, left=0.25, right=0.5)
fig.show()
fig.savefig("paper_plots/Fig5/final_N_clust_lowhigh_tau.pdf",dpi=300)
#
# from scipy.optimize import curve_fit
#
# t_span = np.linspace(0,499,100)
#
# nis_r = n_islands.sum(axis=-2)
#
# plt.plot(np.log((nis_r-2).mean(axis=-2))[:,-3].T)
# plt.show()
# nist0_r = n_islands_tau0.sum(axis=-2)
#
# tau_sort = np.zeros((12,12,12))
# for i in range(12):
#     for j in range(12):
#         for k in range(12):
#             init = nis_r[i,j,k,0]
#             fin = 2
#             def fit_tau_sort(t_span, tau_sort):
#                 return init + (fin - init) * np.exp(-t_span / tau_sort)
#             tau_sort[i,j,k] = curve_fit(fit_tau_sort,t_span,nis_r[i,j,k])[0]
#
#
# tau_sort = np.zeros((12,12))
# for i in range(12):
#     for j in range(12):
#         init = nis[i,j,0]
#         fin = 2
#         def fit_tau_sort(t_span, tau_sort):
#             return init + (fin - init) * np.exp(-t_span / tau_sort)
#         tau_sort[i,j] = curve_fit(fit_tau_sort,t_span,nis[i,j])[0][0]
#
#
# plt.imshow(tau_sort)
# plt.show()


def get_mean_self(X):
    Id, Rep = X
    try:
        FILE = np.load("from_unsorted_control/analysis/%d_%d.npz" % (Id,Rep))
        return FILE["mean_self"]
    except FileNotFoundError:
        return np.nan

def get_mean_self_tau0(X):
    Id, Rep = X
    try:
        FILE = np.load("from_unsorted_control/analysis_tau0/%d_%d.npz" % (Id,Rep))
        return FILE["mean_self"]
    except FileNotFoundError:
        return np.nan


mean_self = Parallel(n_jobs=num_cores)(delayed(get_mean_self)(inputt) for inputt in inputs)
mean_self = np.array(mean_self).reshape(N,N,rep,100)
mean_self_t0 = Parallel(n_jobs=num_cores)(delayed(get_mean_self_tau0)(inputt) for inputt in inputs_tau0)
mean_self_t0 = np.array(mean_self_t0).reshape(N,rep,100)



final_mean_mean_self = mean_self.mean(axis=2)[:,:,-1]

cmap = plt.cm.viridis

vmax = final_mean_mean_self.max()
vmin = 0.5

fig, ax = plt.subplots(figsize=(3,2.5))
extent, aspect = make_extent(tau_range,np.log10(beta_max_range))
ax.imshow(flipim(final_mean_mean_self),extent=extent,aspect=aspect,cmap=cmap,vmin=vmin,vmax=vmax)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmax=vmax, vmin=vmin))
sm._A = []
cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.085, aspect=10, orientation="vertical")
cl.set_label(r"$\phi_{self}$")
ax.set(xlabel=r"$\tau$",ylabel=r"$log_{10} \ \beta_{max}$")
fig.subplots_adjust(top=0.8, bottom=0.2, left=0.25, right=0.8)
fig.savefig("paper_plots/Fig5/mean_self_phase_diag.pdf",dpi=300)


pis = mean_self.mean(axis=2)
pist0 = mean_self_t0.mean(axis=1)

fig, ax = plt.subplots(1,6,figsize=(12,2),sharey=True)
cmap = plt.cm.cividis(np.linspace(0,1,12))
ax = ax.ravel()
for i in range(12):
    for ji, j in enumerate(np.arange(1,12,2)):
        ax[ji].plot(np.linspace(0,499,100).astype(np.int64),pis[i,j],color=cmap[i],linewidth=1)
        ax[ji].plot(np.linspace(0,499,100).astype(np.int64),pist0[j],color="darkred",linewidth=1.3,alpha=0.5)
        ax[ji].set_title("%.1f"%np.log10(beta_max_range[j]))
        ax[ji].set(xlabel="t")
ax[0].set(ylabel=r"$\phi_{self}$")
# sm = plt.cm.ScalarMappable(cmap=plt.cm.cividis, norm=plt.Normalize(vmax=tau_range.max(), vmin=tau_range.min()))
# sm._A = []
# cl = plt.colorbar(sm, ax=ax[-1], pad=0.05, fraction=0.11, aspect=10, orientation="vertical")
# cl.set_label(r"$\tau$")
fig.subplots_adjust(top=0.8, bottom=0.3, left=0.25, right=0.8)
fig.show()
fig.savefig("paper_plots/Fig5/line_plots_phi.pdf",dpi=300)


fig, ax = plt.subplots(figsize=(2.5,2),sharey=True)
cmap = plt.cm.cividis(np.linspace(0,1,12))
ji,j = 5,11
for i in range(12):
    ax.plot(np.linspace(0,499,100).astype(np.int64),pis[i,j],color=cmap[i],linewidth=1)
    ax.plot(np.linspace(0,499,100).astype(np.int64),pist0[j],color="darkred",linewidth=1.3,alpha=0.5)
    ax.set_title("%.1f"%np.log10(beta_max_range[j]))
    ax.set(xlabel="t")
ax.set(ylabel=r"$\phi_{self}$")
sm = plt.cm.ScalarMappable(cmap=plt.cm.cividis, norm=plt.Normalize(vmax=tau_range.max(), vmin=tau_range.min()))
sm._A = []
cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.11, aspect=10, orientation="vertical")
cl.set_label(r"$\tau$")
fig.subplots_adjust(top=0.8, bottom=0.3, left=0.25, right=0.8)
fig.show()
fig.savefig("paper_plots/Fig5/line_plots_phi_self_scalebar_single.pdf",dpi=300)




df = pd.DataFrame({"mean_self":mean_self[:,:,:,-1].ravel(),"beta":BB.ravel(),"tau":TT.ravel(),"rep":RR.ravel(),"logbeta":np.log10(BB.ravel())})
df = pd.concat([df,pd.DataFrame({"mean_self":mean_self_t0[:,:,-1].ravel(),"beta":BBt0.ravel(),"tau":np.zeros_like(BBt0).ravel(),"rep":RRt0.ravel(),"logbeta":np.log10(BBt0.ravel())})])

import seaborn as sb

fig, ax = plt.subplots(figsize=(5,2))
sb.lineplot(data=df.loc[df["tau"]==0],x="logbeta",y="mean_self",color=plt.cm.cividis(0.1),ax=ax,label=r"$\tau$"" =0")
sb.lineplot(data=df.loc[df["tau"]==500],x="logbeta",y="mean_self",color=plt.cm.plasma(0.85),ax=ax,label=r"$\tau$"" =500")
ax.set(xlabel=r"$log_{10} \ \beta_{max}$",ylabel=r"$\phi_{self}$")
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),ncol=2)

fig.subplots_adjust(top=0.8, bottom=0.3, left=0.25, right=0.5)

fig.savefig("paper_plots/Fig5/final_phi_self_lowhigh_tau.pdf",dpi=300)

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
