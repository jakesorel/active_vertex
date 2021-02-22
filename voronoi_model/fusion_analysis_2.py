import numpy as np
# import dask
# from dask.distributed import Client
import matplotlib.pyplot as plt
import multiprocessing
from joblib import Parallel, delayed
from voronoi_model.plotting_functions import *
plt.rcParams.update({'pdf.fonttype': 42})
import seaborn as sb
import pandas as pd

optv0s = np.zeros((12,8))
for optv0_file in os.listdir("optv0s"):
    i = int(optv0_file.split(".txt")[0])
    optv0s[i] = np.loadtxt("optv0s/%s"%optv0_file)

df_optv0 = pd.DataFrame(optv0s)
df_optv0.index = np.logspace(-3, -1, 12)
df_optv0.columns = np.arange(8)
df_optv0 = df_optv0.melt(ignore_index = False)
df_optv0.columns = ["Rep","optv0"]
df_optv0["beta"] = df_optv0.index



fig, ax = plt.subplots()
sb.lineplot(data = df_optv0,x = "beta",y="optv0",ax=ax)
# fig.show()
# ax.plot(beta_range,optv0s.mean(axis=1))
# ci = 90
# ax.fill_between(beta_range,np.percentile(optv0s,100-ci,axis=1),np.percentile(optv0s,ci,axis=1))
# for optv0 in optv0s.T:
#     ax.scatter(np.arange(optv0.size),optv0)
ax.set(xscale="log",yscale="log")
fig.show()

N = 12
beta_range = np.logspace(-3, -1, N)

repn = 8
n_t = 1700


rep_range = np.arange(repn)
BB,RR = np.meshgrid(beta_range,rep_range, indexing="ij")
ID_mat = np.arange(N)
ID_mat = np.stack([ID_mat for i in range(repn)],axis=1)
inputs = np.array([ID_mat.ravel(),RR.ravel()]).T

dir_name = "fusion_using_optv0"

def extract_energies(Id,rep):
    try:
        return np.load("%s/%d_%d.npz"%(dir_name,Id,rep))["energies_all"]
    except:
        return np.ones(n_t)*np.nan



def extract_n_islands(Id,rep):
    try:
        return np.load("%s/%d_%d.npz"%(dir_name,Id,rep))["n_islands"]
    except:
        return np.nan

def extract_swapped(Id, rep):
    try:
        return np.load("%s/%d_%d.npz"%(dir_name,Id,rep))["swapped"]
    except:
        return np.nan



num_cores = multiprocessing.cpu_count()
out = Parallel(n_jobs=num_cores)(delayed(extract_energies)(*inputt) for inputt in inputs)
out = np.array(out).reshape(RR.shape[0],RR.shape[1],-1)



#
# inputs = np.array([ID_mat.ravel(),RR.ravel()]).T
# inputs = inputs.astype(np.int64)
# lazy_results = []
# for inputt in inputs:
#     lazy_result = dask.delayed(extract_n_islands)(*inputt)
#     lazy_results.append(lazy_result)
# out_nislands = dask.compute(*lazy_results)
# out_nislands = np.array(out_nislands).reshape(RR.shape[0],RR.shape[1],RR.shape[2])
#


# num_cores = multiprocessing.cpu_count()
out_swapped = Parallel(n_jobs=num_cores)(delayed(extract_swapped)(*inputt) for inputt in inputs)
out_swapped = np.array(out_swapped).reshape(RR.shape[0],RR.shape[1])


out[~out_swapped] = np.nan
E_t,E_act,E_0 = np.zeros((N,repn)),np.zeros((N,repn)),np.zeros((N,repn))
for i in range(N):
    for j in range(repn):
        E = out[i,j]
        E_t[i,j] = E[-1]
        E_0[i,j]  = E[:np.nanargmax(E)].min()
        E_act[i,j] = np.nanmax(E) - E_0[i,j]
# E_t,E_act = out[:,:,-1],np.nanmax(out,axis=-1)
dE = E_t - E_0


fig, ax = plt.subplots()
cmap = plt.cm.plasma(np.linspace(0,1,12))
id = 1
for i in range(12):
    ax.plot(out[i,id] - E_0[i,id],color=cmap[i])
ax.set(ylim=(-1,1))
fig.show()

E_max = np.nanpercentile(E_act,90)
unstable_mask = E_act>E_max
E_act[unstable_mask] = np.nan
dE[unstable_mask] = np.nan

dir_name = "paper_plots/Fig3"
make_directory(dir_name)

from scipy.stats import linregress
fig, ax = plt.subplots(figsize=(3,3))
ax.scatter(np.log10(beta_range),np.log10(-np.nanmean(dE,axis=1)),color=plt.cm.plasma(0.8),alpha=0.6)
m,c = linregress(np.log10(beta_range),np.log10(-np.nanmean(dE,axis=1)))[:2]
ax.plot(np.log10(beta_range),c+m*np.log10(beta_range),color=plt.cm.plasma(0.8),label=r"$\| \Delta E \|$")
ax.scatter(np.log10(beta_range),np.log10(np.nanmean(E_act,axis=1)),color=plt.cm.plasma(0.2),alpha=0.6)
m,c = linregress(np.log10(beta_range),np.log10(np.nanmean(E_act,axis=1)))[:2]
ax.plot(np.log10(beta_range),c+m*np.log10(beta_range),color=plt.cm.plasma(0.2),label=r"$E_a$")
ax.legend()
ax.set(xlabel=r"$log_{10} \ \beta$",ylabel=r"$log_{10} \ Energy$")
fig.subplots_adjust(top=0.8, bottom=0.2, left=0.2, right=0.8,hspace=0.1)
fig.show()
fig.savefig("%s/dE vs Eact vs beta.pdf"%dir_name,dpi=300)

beta_rangef = np.logspace(np.log10(beta_range[0]),np.log10(beta_range[-1]),200)

fig, ax = plt.subplots(figsize=(3,3))
ax.scatter(np.log10(beta_range),-np.nanmean(dE,axis=1),color=plt.cm.plasma(0.8),alpha=0.6)
m,c = linregress(np.log10(beta_range),np.log10(-np.nanmean(dE,axis=1)))[:2]
ax.plot(np.log10(beta_rangef),10**(c+m*np.log10(beta_rangef)),color=plt.cm.plasma(0.8),label=r"$\| \Delta E \|$")
ax.scatter(np.log10(beta_range),np.nanmean(E_act,axis=1),color=plt.cm.plasma(0.2),alpha=0.6)
m,c = linregress(np.log10(beta_range),np.log10(np.nanmean(E_act,axis=1)))[:2]
ax.plot(np.log10(beta_rangef),10**(c+m*np.log10(beta_rangef)),color=plt.cm.plasma(0.2),label=r"$E_a$")
ax.legend()
ax.set(xlabel=r"$log_{10} \ \beta$",ylabel=r"$ Energy$")
fig.subplots_adjust(top=0.8, bottom=0.2, left=0.2, right=0.8,hspace=0.1)
fig.show()
fig.savefig("%s/dE vs Eact vs beta linscale.pdf"%dir_name,dpi=300)


fig, ax = plt.subplots()
ax.scatter(np.nanmean(E_act,axis=1),-np.nanmean(dE,axis=1))
ax.set(xscale="log",yscale="log")
fig.show()

E_norm = (out.T - E_0.T).T
E_shift = np.zeros((E_norm.shape[0],E_norm.shape[1],E_norm.shape[2]*2))*np.nan
for i in range(N):
    for j in range(repn):
        if not np.isnan(dE[i,j]):
            t0 = E_norm.shape[2] - np.nanargmax(E_norm[i,j])
            E_shift[i,j,t0:t0+E_norm.shape[2]] = E_norm[i,j]

dT = 500
E_shift_crop = E_shift[:,:,E_norm.shape[2] - dT:E_norm.shape[2]+dT]
# E_shift_crop = (E_shift_crop.T - E_shift_crop[:,:,0].T).T
fig, ax = plt.subplots()
ax.scatter(-np.nanmean(E_shift_crop[:,:,-1],axis=1),np.nanmean(np.nanmax(E_shift_crop,axis=-1),axis=1))
# ax.set(xscale="log",yscale="log")
fig.show()
id = 7
fig, ax = plt.subplots()
cols = plt.cm.plasma(np.linspace(0,1,N))
for i in range(N):
    # ax.plot((np.arange(E_norm[0,0].size*2) - E_norm[0,0].size)[::50]*0.025,np.nanmean(E_shift[i],axis=0)[::50],color=cols[i])
    ax.plot(E_shift_crop[i,id],color=cols[i])
# ax.set(ylim=(-0.05,0.05))
# ax.set(xlim=(1200,2200))
fig.show()
    t = np.arange(eps.size) - np.argmax(eps)
    ax.plot(t, out[i,5]-out[i,5,0],color=cols[i])
# ax.set(ylim=(-0.01,0.01))
fig.show()




fig, ax = plt.subplots(2,figsize=(3.5,6),sharex=True)
extent = [p0_range.min(),p0_range.max(),np.log10(beta_range.min()),np.log10(beta_range.max())]
aspect = (extent[1]-extent[0])/(extent[3]-extent[2])
ax[0].imshow(np.log(np.flip(np.nanmean(E_act,axis=-1).T,axis=0)),aspect=aspect,extent=extent,cmap=plt.cm.plasma)
ax[1].imshow(np.flip(np.nanmean(dE,axis=-1).T,axis=0),aspect=aspect,extent=extent,cmap = plt.cm.inferno)
sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(vmax=np.nanmax(E_act), vmin=np.nanmin(E_act)))
sm._A = []
cl = plt.colorbar(sm, ax=ax[0], pad=0.05, fraction=0.085, aspect=10, orientation="vertical")
cl.set_label(r"$E_a$")
sm = plt.cm.ScalarMappable(cmap=plt.cm.inferno, norm=plt.Normalize(vmax=np.nanmax(dE), vmin=np.nanmin(dE)))
sm._A = []
cl = plt.colorbar(sm, ax=ax[1], pad=0.05, fraction=0.085, aspect=10, orientation="vertical")
cl.set_label(r"$\Delta E$")
fig.subplots_adjust(top=0.8, bottom=0.2, left=0.2, right=0.8,hspace=0.1)
for axx in ax:
    axx.set(ylabel=r"$log_{10} \ \beta$")
ax[1].set(xlabel=r"$p_0$")
fig.savefig("analysis_plots/Ea and dE heatmap.pdf",dpi=300)
fig.show()


print(out_swapped.mean()*100,"% swapped (ensure this is 100%)")


def normalise(x):
    return (x - np.nanmin(x))/(np.nanmax(x)-np.nanmin(x))


colsB = plt.cm.viridis(normalise(np.log10(BB[~unstable_mask])))
colsP = plt.cm.inferno(normalise(PP[~unstable_mask]))

s = 4.5

fig, ax = plt.subplots(2,figsize=(3,5),sharex=True)
ax[0].scatter(dE[~unstable_mask],E_act[~unstable_mask],c=colsB,s=s)
sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmax=np.log10(beta_range.max()), vmin=np.log10(beta_range.min())))
sm._A = []
cl = plt.colorbar(sm, ax=ax[0], pad=0.05, fraction=0.085, aspect=10, orientation="vertical")
cl.set_label(r"$log_{10} \ \beta$")

ax[1].scatter(dE[~unstable_mask],E_act[~unstable_mask],c=colsP,s=s)
sm = plt.cm.ScalarMappable(cmap=plt.cm.inferno, norm=plt.Normalize(vmax=p0_range.max(), vmin=p0_range.min()))
sm._A = []
cl = plt.colorbar(sm, ax=ax[1], pad=0.05, fraction=0.085, aspect=10, orientation="vertical")
cl.set_label(r"$p_0$")
for axx in ax:
    axx.set(ylabel=r"$E_{a}$")
ax[1].set(xlabel=r"$\Delta E$")

fig.subplots_adjust(top=0.8, bottom=0.2, left=0.2, right=0.75)
fig.savefig("analysis_plots/scatter vs beta and p0.pdf",dpi=300)


out_zeroed = (out.T - out[:,:,:,399].T).T
out_zeroed[unstable_mask] = np.nan
out_zeroed_mean = np.nanmean(out_zeroed,axis=-2)

n_t =  20000
no_noise_time = 400
T_eval = np.concatenate((np.arange(no_noise_time * 4).astype(np.int64),np.linspace(no_noise_time * 4, n_t - 1, 100).astype(np.int64)))*0.025

fig, ax = plt.subplots(figsize=(5,4.5))
cols = plt.cm.plasma(np.linspace(0,1,10))
for i in range(10):
    ax.plot(T_eval[no_noise_time:]-no_noise_time*0.025,out_zeroed[-3,i,2,no_noise_time:],color=cols[i])
# ax.set(xscale="log",ylim=(-0.1,0.1))
sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(vmax=np.log10(beta_range.max()), vmin=np.log10(beta_range.min())))
sm._A = []
cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.11, aspect=10, orientation="vertical")
cl.set_label(r"$log_{10} \ \beta$")
ax.set(xscale="log",xlabel="Time",ylabel=r"$\epsilon$")
fig.subplots_adjust(top=0.8, bottom=0.2, left=0.2, right=0.8,wspace=1)
fig.savefig("analysis_plots/eps vs time vs beta 2.pdf",dpi=300)


#############################



fig, ax = plt.subplots()
cols = plt.cm.plasma(np.linspace(0,1,N))
for i in range(8):
    ax.plot(out[0,2,i],color=cols[i])
# ax.set(xscale="log")
fig.show()


max_energy_cutoff = np.nanpercentile(np.nanmax(out,axis=-1).ravel(),99) #remove instability e.g. s

E_0 = np.zeros((N,N,repn))
E_t = np.zeros((N,N,repn))
E_act = np.zeros((N,N,repn))
for k in range(repn):
    for i in range(N):
        for j in range(N):
            mask = out_nislands[i,:,j,k]==2
            if mask.sum()!= 0:
                l = np.nonzero(mask)[0][0]
                energy = out[i,l,j,k]
                max_energy = energy.max()
                if max_energy > max_energy_cutoff:
                    E_0[i,j,k] = np.nan
                    E_t[i,j,k] = np.nan
                    E_act[i, j, k] = np.nan

                else:
                    E_0[i,j,k] = energy[0]
                    E_t[i,j,k] = energy[-1]
                    if (max_energy>E_0[i,j,k])*(max_energy>E_t[i,j,k]):
                        E_act[i,j,k] = max_energy - energy[0]
                    else:
                        E_act[i,j,k] = np.nan
            else:
                E_act[i,j,k] = np.nan

dE = E_t - E_0

fig, ax = plt.subplots(figsize=(4,4))
i = 1
ax.scatter(np.log(-dE[:,:,:]),np.log(E_act[:,:,:]),s=2,color="k")
# ax.set(xscale="log",yscale="log")
ax.set(xlim=(-8,0),ylim=(-5,0))
ax.set(xlabel="dE",ylabel="E_act")
fig.show()


fig, ax = plt.subplots(1,2)
extent=[p0_range.min(),p0_range.max(),np.log10(beta_range.min()),np.log10(beta_range.max())]
aspect = (extent[1]-extent[0])/(extent[3]-extent[2])
ax[0].imshow(np.flip(np.nanmean(E_act,axis=-1).T,axis=0),extent=extent,aspect=aspect)
ax[1].imshow(np.flip(np.nanmean(dE,axis=-1).T,axis=0),extent=extent,aspect=aspect)
ax[0].set_title(r"$E_{act}$")
ax[1].set_title(r"$\Delta E$")
for axx in ax:
    axx.set(xlabel="p0",ylabel="log beta")
fig.show()

E_0 = np.zeros((N,N,N,repn))
E_t = np.zeros((N,N,N,repn))
E_act = np.zeros((N,N,N,repn))
for k in range(repn):
    for l in range(N):
        for i in range(N):
            for j in range(N):
                energy = out[i,l,j,k]
                max_energy = energy.max()
                if max_energy > max_energy_cutoff:
                    E_0[i,j,l,k] = np.nan
                    E_t[i,j,l,k] = np.nan
                    E_act[i, j,l, k] = np.nan
                else:
                    if (max_energy>E_0[i,l,j,k])*(max_energy>E_t[i,l,j,k]):
                        E_act[i,l,j,k] = max_energy - energy[0]
                        E_0[i, j, l, k] = energy[0]
                        E_t[i, j, l, k] = energy[-1]
                    else:
                        E_0[i, j, l, k] = np.nan
                        E_t[i, j, l, k] = np.nan
                        E_act[i, j, l, k] = np.nan


E_act_mean = np.zeros((N,N))*np.nan
dE_mean = np.zeros((N,N))*np.nan
for i in range(N):
    for j in range(N):
        eacts = []
        dEs = []
        for k in range(repn):
            eact = np.nanmin(E_act[i, :, j,k])
            if ~np.isnan(eact):
                eacts.append(eact)
                dEs.append(E_t[i,np.nanargmin(E_act[i, :, j,k]),j,k]-E_0[i,np.nanargmin(E_act[i, :, j,k]),j,k])

        E_act_mean[i,j] = np.nanmean(eacts)
        dE_mean[i,j] = np.nanmean(dEs)

fig, ax = plt.subplots()
ax.scatter(dE_mean,E_act_mean)
# ax.set(xscale="log")
fig.show()


dE_mean = E_t_mean - E_0_mean

fig, ax = plt.subplots(1,2)
extent=[p0_range.min(),p0_range.max(),np.log10(beta_range.min()),np.log10(beta_range.max())]
aspect = (extent[1]-extent[0])/(extent[3]-extent[2])
ax[0].imshow(np.flip(E_act_mean.T,axis=0),extent=extent,aspect=aspect)
ax[1].imshow(np.flip((dE_mean).T,axis=0),extent=extent,aspect=aspect)
ax[0].set_title(r"$E_{act}$")
ax[1].set_title(r"$\Delta E$")
for axx in ax:
    axx.set(xlabel="p0",ylabel="log beta")
fig.show()


from scipy.interpolate import bisplrep,bisplev
nfine=200
p0_spacefine, v0_spacefine,beta_spacefine = np.linspace(p0_range.min(),p0_range.max(), nfine), np.linspace(v0_range.min(), v0_range.max(), nfine),np.logspace(np.log10(beta_range.min()),np.log10(beta_range.max()),nfine)
PPs, lBBs = np.meshgrid(p0_range, np.log10(beta_range), indexing="ij")

E_act_mean = np.nanmean(E_act,axis=-1)
E_act_mean_mask = ~np.isnan(E_act_mean)
z = bisplev(p0_spacefine,np.log10(beta_spacefine), bisplrep(PPs[E_act_mean_mask].ravel(),lBBs[E_act_mean_mask].ravel(),E_act_mean[E_act_mean_mask].ravel(),s=1))


fig, ax = plt.subplots()
ax.hist(np.nanmax(out,axis=-1).ravel(),bins=100)
ax.set(yscale="log")
fig.show()
np.where((np.nanmax(out,axis=-1)>8)*(np.nanmax(out,axis=-1)<10))