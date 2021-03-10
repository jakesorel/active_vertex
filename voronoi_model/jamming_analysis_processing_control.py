import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import multiprocessing
from joblib import Parallel, delayed
from voronoi_model.plotting_functions import *
from scipy.optimize import curve_fit
plt.rcParams.update({'pdf.fonttype': 42})


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

t_span = np.arange(0,500,0.025)[15000:]


def get_tot_t1(X):
    Id, Rep = X
    i, run = (Rep%6), int(Rep/6)
    try:
        FILE = np.load("from_unsorted_control/jamming_analysis/%d_%d_%s.npz" % (Id,i,run))
        return FILE["t1_swap_freq"].sum(axis=0)
        # return FILE["av_rate"]

    except FileNotFoundError:
        return np.repeat(np.nan,6)

num_cores = multiprocessing.cpu_count()
tot_t1 = Parallel(n_jobs=num_cores)(delayed(get_tot_t1)(inputt) for inputt in inputs)
tot_t1 = np.array(tot_t1).reshape(N,N,rep,6)


tot_t1_mean = tot_t1[:,:,:,4].mean(axis=2)
tot_t1_mean[np.isinf(tot_t1_mean)] = np.nan

# tot_t1_mean = tot_t1_mean[:8]

fig, ax = plt.subplots(figsize=(3,2.5))
extent, aspect = make_extent(tau_range,np.log10(beta_max_range))
vmin,vmax = np.nanmin(tot_t1_mean),np.nanpercentile(tot_t1_mean,65)
ax.imshow(flipim(tot_t1_mean),extent=extent,aspect=aspect,cmap=plt.cm.coolwarm,vmin=vmin,vmax=vmax)
sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(vmax=vmax, vmin=vmin))
sm._A = []
cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.085, aspect=10, orientation="vertical")
cl.set_label("Number of Type IV T1s")
ax.set(xlabel=r"$v_0$",ylabel=r"$log_{10} \ \beta$")
fig.subplots_adjust(top=0.8, bottom=0.2, left=0.25, right=0.8)
fig.show()

def get_t1s(X):
    Id, Rep = X
    i, run = (Rep%6), int(Rep/6)
    try:
        FILE = np.load("from_unsorted_control/jamming_analysis/%d_%d_%s.npz" % (Id,i,run))
        return FILE["t1_swap_freq"],FILE["changed_t"]
        # return FILE["av_rate"]

    except FileNotFoundError:
        return np.nan

mask = TT == tau_range[0]
inputs2 = np.array([ID_mat[np.where(mask)],RR[np.where(mask)]]).T

num_cores = multiprocessing.cpu_count()
t1s = Parallel(n_jobs=num_cores)(delayed(get_t1s)(inputt) for inputt in inputs2)
t_span = np.arange(0,500,0.025)
fig, ax = plt.subplots(1,6,figsize=(10,2.2),sharey=True)
ax = ax.ravel()
cols = plt.cm.plasma((np.linspace(-2.5,-1,12) + 3)/2)
swap_freq_fin = np.zeros((12,6))
for i in range(12):
    t1_beta = t1s[12*i:12*(i+1)]
    swap_freq = np.zeros((20000,12,6))
    for j in range(6):
        # if type(t1_beta) is not float:
        t1_swap_freq,changed_t = t1_beta[j]
        swap_freq[changed_t,j]=t1_swap_freq
    swap_freq = swap_freq.sum(axis=1)/12
    for j in range(3):
        ax[j].plot(t_span,np.cumsum(swap_freq[:,j]),color=cols[i])
    ax[3].plot(t_span,np.cumsum(swap_freq[:, 3])+np.cumsum(swap_freq[:, 5]), color=cols[i])
    ax[4].plot(t_span,np.cumsum(swap_freq[:, 4]), color=cols[i])
    ax[5].plot(np.cumsum(swap_freq.sum(axis=1)), color=cols[i])
    swap_freq_fin[i] = swap_freq.sum(axis=0)

sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(vmax=-1, vmin=-3))
sm._A = []
cl = plt.colorbar(sm, ax=ax[5], pad=0.05, fraction=0.175, aspect=10, orientation="vertical")
cl.set_label(r"$log_{10} \ \beta_{max}$")
ax[0].set(ylabel="Cumulative \n number of T1s")
for axx in ax:
    axx.set(xlabel="t")
fig.subplots_adjust(top=0.8, bottom=0.3, left=0.25, right=0.8,wspace=0.25)
fig.show()
fig.savefig("paper_plots/Fig5/cumulativeT1_tau500.pdf",dpi=300)




fig, ax = plt.subplots(figsize=(2.6,2.2))
cols = plt.cm.plasma(np.linspace(0,1,12))
for i in range(12):
    t1_beta = t1s[12*i:12*(i+1)]
    swap_freq = np.zeros((20000,12,6))
    for j in range(12):
        # if type(t1_beta) is not float:
        t1_swap_freq,changed_t = t1_beta[j]
        swap_freq[changed_t,j]=t1_swap_freq
    swap_freq = swap_freq.sum(axis=1)/12
    ax.plot(t_span,np.cumsum(swap_freq[:, 4]), color=cols[i])
sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(vmax=np.log10(beta_range.max()), vmin=np.log10(beta_range.min())))
sm._A = []
cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.175, aspect=10, orientation="vertical")
cl.set_label(r"$log_{10} \ \beta$")
ax.set(ylabel="Cumulative\nnumber of T1s",xlabel="t")
fig.subplots_adjust(top=0.8, bottom=0.3, left=0.35, right=0.8,wspace=0.25)
fig.show()
fig.savefig("paper_plots/Fig1/cumulativeT1_v07_type4only.pdf",dpi=300)


fig, ax = plt.subplots(figsize=(2.6,2.2))
cols = plt.cm.plasma(np.linspace(0,1,12))
for i in range(12):
    t1_beta = t1s[12*i:12*(i+1)]
    swap_freq = np.zeros((20000,12,6))
    for j in range(12):
        # if type(t1_beta) is not float:
        t1_swap_freq,changed_t = t1_beta[j]
        swap_freq[changed_t,j]=t1_swap_freq
    swap_freq = swap_freq.sum(axis=1)/12
    ax.plot(t_span,np.cumsum(swap_freq.sum(axis=1)), color=cols[i])
sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(vmax=np.log10(beta_range.max()), vmin=np.log10(beta_range.min())))
sm._A = []
cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.175, aspect=10, orientation="vertical")
cl.set_label(r"$log_{10} \ \beta$")
ax.set(ylabel="Cumulative\nnumber of T1s",xlabel="t")
fig.subplots_adjust(top=0.8, bottom=0.3, left=0.35, right=0.8,wspace=0.25)
fig.show()
fig.savefig("paper_plots/Fig1/cumulativeT1_v07_all.pdf",dpi=300)




fin_swap = np.array([np.repeat(np.nan,6) if type(val[0]) is tuple else val[0].sum(axis=0) for val in t1s])

betaa, Idd = BB[np.where(VV == v0_range[5])],RR[np.where(VV == v0_range[7])]

df = pd.DataFrame(np.column_stack((betaa,Idd,fin_swap)))

fig, ax = plt.subplots()
for i in range(2,8):
    sb.lineplot(x=0,y=i,data = df,ax = ax)
ax.set(xscale="log")
fig.show()

swap_freq_fin[:,3]+=swap_freq_fin[:,5]
swap_freq_fin = swap_freq_fin[:,:5]
plt.plot(swap_freq_fin)
plt.show()

plt.plot(np.log(swap_freq_fin[:,4]))
plt.plot(np.log(swap_freq_fin.sum(axis=1)))
plt.show()


t1s = Parallel(n_jobs=num_cores)(delayed(get_t1s)(inputt) for inputt in inputs)
fin_swap_all = np.array([val[0].sum(axis=0)[4] for val in t1s])


df = pd.DataFrame({"v0": VV.ravel(),"beta":BB.ravel(),"log beta":np.log10(BB.ravel()),"swap":fin_swap_all})

fig, ax = plt.subplots(figsize=(3,3))
cmap = plt.cm.plasma(np.linspace(0,1,12))
for i in range(0,12,2):
    sb.lineplot(data = df.loc[(df["beta"]==beta_range[i])],x = "v0",y="swap",color=cmap[i])
ax.set(xlim=(0.01,0.06),ylim=(0,10))
sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(vmax=np.log10(beta_range.max()), vmin=np.log10(beta_range.min())))
sm._A = []
cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.175, aspect=10, orientation="vertical")
cl.set_label(r"$log_{10} \ \beta$")
ax.set(ylabel="Number of T1s",xlabel=r"$v_0$")
fig.subplots_adjust(top=0.8, bottom=0.3, left=0.35, right=0.8,wspace=0.25)
# ax.set(ylabel="Number het swaps")
fig.show()
fig.savefig("paper_plots/Fig1/number_t1s_by_v0.pdf",dpi=300)

fig, ax = plt.subplots(figsize=(3,3))
cmap = plt.cm.viridis(np.linspace(0,1,6))
for i in range(1,7):
    sb.lineplot(data = df.loc[(df["v0"]==v0_range[i])],x = "log beta",y="swap",color=cmap[i-1])
ax.set(ylabel="Number of T1s",xlabel=r"$log_{10} \ \beta$")
sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmax=v0_range[1], vmin=v0_range[7]))
sm._A = []
cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.175, aspect=10, orientation="vertical")
cl.set_label(r"$v_0$")
fig.subplots_adjust(top=0.8, bottom=0.3, left=0.35, right=0.8,wspace=0.25)
fig.show()
fig.savefig("paper_plots/Fig1/number_t1s_by_beta.pdf",dpi=300)


fig, ax = plt.subplots(figsize=(3,3))
cmap = plt.cm.viridis(np.linspace(0,1,6))
for i in range(1,7):
    sb.lineplot(data = df.loc[(df["v0"]==v0_range[i])],x = "log beta",y="swap",color=cmap[i-1])
ax.set(ylabel="Number of T1s",xlabel=r"$log_{10} \ \beta$")
sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmax=v0_range[1], vmin=v0_range[7]))
sm._A = []
cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.175, aspect=10, orientation="vertical")
cl.set_label(r"$v_0$")
ax.set(ylim=(0,10))
fig.subplots_adjust(top=0.8, bottom=0.3, left=0.35, right=0.8,wspace=0.25)
fig.show()
fig.savefig("paper_plots/Fig1/number_t1s_by_beta_2.pdf",dpi=300)



beta_i = beta_range[6]

plt.scatter(VV[BB==beta_i],np.log10(fin_swap_all[BB==beta_i]))
plt.show()

def get_av_rate(X):
    Id, Rep = X
    i, run = (Rep%6), int(Rep/6)
    try:
        FILE = np.load("from_unsorted/jamming_analysis/%d_%d_%s.npz" % (Id,i,run))
        # return FILE["t1_swap_freq"].sum(axis=0)
        return FILE["av_rate"]

    except FileNotFoundError:
        return np.repeat(np.nan,6)

num_cores = multiprocessing.cpu_count()
av_rate = Parallel(n_jobs=num_cores)(delayed(get_av_rate)(inputt) for inputt in inputs2)
av_rate = np.array(av_rate).reshape(N,rep,6)

av_rate7 = np.nanmean(av_rate,axis=1)
av_rate7[:,3] = (av_rate7[:,3] + av_rate7[:,5])/2
av_rate7[:,:5]
fig, ax = plt.subplots()
for i in range(5):
    plt.plot(av_rate7[:,i],label=types[i])
ax.legend()
# ax.set(yscale="log")
fig.show()


# tot_t1 = np.array(tot_t1).reshape(N,rep,6)




#
#
# def get_L_star(X):
#     Id, Rep = X
#     try:
#         FILE = np.load("from_unsorted/analysis/%d_%d.npz" % (Id,Rep))
#         return FILE["L_star"]
#     except FileNotFoundError:
#         return np.nan
#
#
# L_star = Parallel(n_jobs=num_cores)(delayed(get_L_star)(inputt) for inputt in inputs)
# L_star = np.array(L_star).reshape(N,N,rep,100)
# dL_star = L_star[:,:,:,-1] - L_star[:,:,:,0]
#
# final_mean_L_star = L_star.mean(axis=2)[:,:,-1]
# final_mean_dL_star = dL_star.mean(axis=2)
#
# cmap = plt.cm.plasma
# fig, ax = plt.subplots(figsize=(3,2.5))
# extent, aspect = make_extent(v0_range,np.log10(beta_range))
# ax.imshow(flipim(final_mean_L_star),extent=extent,aspect=aspect,cmap=cmap)
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmax=final_mean_L_star.max(), vmin=final_mean_L_star.min()))
# sm._A = []
# cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.085, aspect=10, orientation="vertical")
# cl.set_label(r"$L^*$")
# ax.set(xlabel=r"$v_0$",ylabel=r"$log_{10} \ \beta$")
# fig.subplots_adjust(top=0.8, bottom=0.2, left=0.25, right=0.8)
# fig.savefig("paper_plots/Fig1/L_star_phase_diag.pdf",dpi=300)
#
#
# cmap = plt.cm.plasma
# fig, ax = plt.subplots(figsize=(3,2.5))
# extent, aspect = make_extent(v0_range,np.log10(beta_range))
# ax.imshow(flipim(final_mean_dL_star),extent=extent,aspect=aspect,cmap=cmap)
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmax=final_mean_dL_star.max(), vmin=final_mean_dL_star.min()))
# sm._A = []
# cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.085, aspect=10, orientation="vertical")
# cl.set_label(r"$\Delta L^*$")
# ax.set(xlabel=r"$v_0$",ylabel=r"$log_{10} \ \beta$")
# fig.subplots_adjust(top=0.8, bottom=0.2, left=0.25, right=0.8)
# fig.savefig("paper_plots/Fig1/dL_star_phase_diag.pdf",dpi=300)
#
#
#
#
# def get_mean_self(X):
#     Id, Rep = X
#     try:
#         FILE = np.load("from_unsorted/analysis/%d_%d.npz" % (Id,Rep))
#         return FILE["mean_self"]
#     except FileNotFoundError:
#         return np.nan
#
#
# mean_self = Parallel(n_jobs=num_cores)(delayed(get_mean_self)(inputt) for inputt in inputs)
# mean_self = np.array(mean_self).reshape(N,N,rep,100)
#
# dmean_self = mean_self[:,:,:,-1] - mean_self[:,:,:,0]
#
# final_mean_dmean_self = dmean_self.mean(axis=2)
# final_mean_mean_self = mean_self.mean(axis=2)[:,:,-1]
#
# cmap = plt.cm.viridis
#
# vmax = 0.85
# vmin = 0.5
#
# fig, ax = plt.subplots(figsize=(3,2.5))
# extent, aspect = make_extent(v0_range,np.log10(beta_range))
# ax.imshow(flipim(final_mean_mean_self),extent=extent,aspect=aspect,cmap=cmap,vmin=vmin,vmax=vmax)
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmax=vmax, vmin=vmin))
# sm._A = []
# cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.085, aspect=10, orientation="vertical")
# cl.set_label(r"$\phi_{self}$")
# ax.set(xlabel=r"$v_0$",ylabel=r"$log_{10} \ \beta$")
# fig.subplots_adjust(top=0.8, bottom=0.2, left=0.25, right=0.8)
# fig.savefig("paper_plots/Fig1/mean_self_phase_diag.pdf",dpi=300)
#
#
# cmap = plt.cm.viridis
#
#
# fig, ax = plt.subplots(figsize=(3,2.5))
# extent, aspect = make_extent(v0_range,np.log10(beta_range))
# ax.imshow(flipim(final_mean_dmean_self),extent=extent,aspect=aspect,cmap=cmap)
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmax=final_mean_dmean_self.max(), vmin=final_mean_dmean_self.min()))
# sm._A = []
# cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.085, aspect=10, orientation="vertical")
# cl.set_label(r"$\Delta \phi_{self}$")
# ax.set(xlabel=r"$v_0$",ylabel=r"$log_{10} \ \beta$")
# fig.subplots_adjust(top=0.8, bottom=0.2, left=0.25, right=0.8)
# fig.savefig("paper_plots/Fig1/dmean_self_phase_diag.pdf",dpi=300)
#
#
# """Time courses"""
#
# tspan_sample = np.linspace(0,500,100)
#
# cmap = plt.cm.plasma
# cols = cmap(np.linspace(0,1,N))
# fig, ax = plt.subplots(figsize=(3,2.5))
# for i in range(N):
#     ax.plot(tspan_sample,mean_self[5,i].mean(axis=0),color=cols[i])
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmax=np.log10(beta_range.max()), vmin=np.log10(beta_range).min()))
# sm._A = []
# cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.085, aspect=10, orientation="vertical")
# cl.set_label(r"$log_{10} \ \beta$")
# ax.set(xlabel=r"$t$",ylabel=r"$\phi_{self}$")
# fig.subplots_adjust(top=0.8, bottom=0.25, left=0.20, right=0.75)
# fig.savefig("paper_plots/Fig1/mean_self_timecourse.pdf",dpi=300)
#
#
# cmap = plt.cm.plasma
# cols = cmap(np.linspace(0,1,N))
# fig, ax = plt.subplots(figsize=(3,2.5))
# for i in range(N):
#     ax.plot(tspan_sample,n_islands[7,i].sum(axis=1).mean(axis=0),color=cols[i])
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmax=np.log10(beta_range.max()), vmin=np.log10(beta_range).min()))
# sm._A = []
# cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.085, aspect=10, orientation="vertical")
# cl.set_label(r"$log_{10} \ \beta$")
# ax.set(xlabel=r"$t$",ylabel=r"$N_{clust}$")
# fig.subplots_adjust(top=0.8, bottom=0.25, left=0.20, right=0.75)
# fig.savefig("paper_plots/Fig1/n_islands_timecourse.pdf",dpi=300)
#
#
# t0 = 10
# t1 = -1
#
# from scipy.stats import linregress
#
# ms_grad = np.zeros((N,N,rep))
# for i in range(N):
#     for j in range(N):
#         for k in range(rep):
#             ms_grad[i,j,k] = linregress(tspan_sample[t0:t1],mean_self[i,j,k,t0:t1])[0]
#
# fig, ax = plt.subplots()
# ax.imshow(flipim(ms_grad.mean(axis=2)))
# fig.show()
#
#
# dL_star = (L_star.T - L_star[:,:,:,0].T).T
#
# cmap = plt.cm.plasma
# cols = cmap(np.linspace(0,1,N))
# fig, ax = plt.subplots(figsize=(3,2.5))
# for i in range(N):
#     ax.plot(tspan_sample,dL_star[5,i,:].mean(axis=0),color=cols[i])
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmax=np.log10(beta_range.max()), vmin=np.log10(beta_range).min()))
# sm._A = []
# cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.085, aspect=10, orientation="vertical")
# cl.set_label(r"$log_{10} \ \beta$")
# ax.set(xlabel=r"$t$",ylabel=r"$\phi_{self}$")
# fig.subplots_adjust(top=0.8, bottom=0.25, left=0.20, right=0.75)
# fig.savefig("paper_plots/Fig1/L_star_timecourse.pdf",dpi=300)
#
#
#
#
#
# def get_n_het_swap_tot(X):
#     Id, Rep = X
#     try:
#         FILE = np.load("from_unsorted/het_swaps/%d_%d.npz" % (Id,Rep))
#         return FILE["n_het_swap_tot"]
#     except FileNotFoundError:
#         return np.nan
#
#
#
#
# def get_n_het_swap_tot(X,t0=2000):
#     Id, Rep = X
#     try:
#         FILE = np.load("from_unsorted/het_swaps/%d_%d.npz" % (Id,Rep))
#         n_het_swap,changed_t = FILE["n_het_swap"], FILE["changed_t"]
#         return n_het_swap[changed_t>t0].sum()
#     except FileNotFoundError:
#         return np.nan
#
#
#
#
# num_cores = multiprocessing.cpu_count()
# n_het_swap_tot = Parallel(n_jobs=num_cores)(delayed(get_n_het_swap_tot)(inputt) for inputt in inputs)
# n_het_swap_tot = np.array(n_het_swap_tot).reshape(N,N,rep)
#
#
# fig, ax = plt.subplots()
# ax.imshow(flipim(n_het_swap_tot.mean(axis=2)))
# fig.show()
#
# mean_swap_rate = n_het_swap_tot.mean(axis=2)/((20000-2000)*(0.025))
#
# vmin = mean_swap_rate.min()
# vmax = 0.005#np.percentile(n_het_swap_tot.mean(axis=2),50)
#
# cmap = plt.cm.Reds
# fig, ax = plt.subplots(figsize=(3,2.5))
# extent, aspect = make_extent(v0_range,np.log10(beta_range))
# ax.imshow(flipim(mean_swap_rate),extent=extent,aspect=aspect,cmap=cmap,vmax = vmax)
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmax=vmax, vmin=vmin))
# sm._A = []
# cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.085, aspect=10, orientation="vertical")
# cl.formatter.set_powerlimits((0, 0))
# cl.set_label(r"$\frac{dN_{swap}}{dt}$")
# ax.set(xlabel=r"$v_0$",ylabel=r"$log_{10} \ \beta$")
# fig.subplots_adjust(top=0.8, bottom=0.2, left=0.25, right=0.8)
# # fig.show()
# fig.savefig("paper_plots/Fig1/n_het_swap.pdf",dpi=300)
#
#
# vals = np.log10(mean_swap_rate).copy()
# vals[np.isinf(vals)] = np.nan
#
# vmin = np.nanmin(vals)
# vmax = np.nanpercentile(vals,90)
# cmap = plt.cm.Reds
# fig, ax = plt.subplots(figsize=(3.15,2.5))
# extent, aspect = make_extent(v0_range,np.log10(beta_range))
# ax.imshow(flipim(np.log10(mean_swap_rate)),extent=extent,aspect=aspect,cmap=cmap,vmin=vmin,vmax=vmax)
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmax=vmax, vmin=vmin))
# sm._A = []
# cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.085, aspect=10, orientation="vertical")
# cl.formatter.set_powerlimits((0, 0))
# cl.set_label(r"$log_{10} \left(\frac{dN_{swap}}{dt}\right)$")
# ax.set(xlabel=r"$v_0$",ylabel=r"$log_{10} \ \beta$")
# fig.subplots_adjust(top=0.8, bottom=0.2, left=0.25, right=0.75)
# # fig.show()
# fig.savefig("paper_plots/Fig1/n_het_swap_log.pdf",dpi=300)
