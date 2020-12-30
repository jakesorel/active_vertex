import numpy as np
import dask
from dask.distributed import Client
import matplotlib.pyplot as plt

n_slurm_tasks = 8
client = Client(threads_per_worker=1, n_workers=n_slurm_tasks, memory_limit="1GB")

N = 10
p0_range = np.linspace(3.5, 4, N)
v0_range = np.linspace(1e-2, 1, N)
beta_range = np.logspace(-3, -1, N)

repn = 8
n_t = 20000


rep_range = np.arange(repn)
PP, BB,RR = np.meshgrid(p0_range, beta_range,rep_range, indexing="ij")
ID_mat = np.arange(N**2).astype(int).reshape(N,N)
ID_mat = np.stack([ID_mat for i in range(repn)],axis=2)

dir_name = "fusion_using_optv0"

def extract_energies(Id,rep):
    try:
        return np.load("%s/%d_%d.npz"%(dir_name,Id,rep))["energies"]
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



inputs = np.array([ID_mat.ravel(),RR.ravel()]).T
inputs = inputs.astype(np.int64)
lazy_results = []
for inputt in inputs:
    lazy_result = dask.delayed(extract_energies)(*inputt)
    lazy_results.append(lazy_result)
out = dask.compute(*lazy_results)
out = np.array(out).reshape(RR.shape[0],RR.shape[1],RR.shape[2],-1)


inputs = np.array([ID_mat.ravel(),RR.ravel()]).T
inputs = inputs.astype(np.int64)
lazy_results = []
for inputt in inputs:
    lazy_result = dask.delayed(extract_n_islands)(*inputt)
    lazy_results.append(lazy_result)
out_nislands = dask.compute(*lazy_results)
out_nislands = np.array(out_nislands).reshape(RR.shape[0],RR.shape[1],RR.shape[2])



inputs = np.array([ID_mat.ravel(),RR.ravel()]).T
inputs = inputs.astype(np.int64)
lazy_results = []
for inputt in inputs:
    lazy_result = dask.delayed(extract_swapped)(*inputt)
    lazy_results.append(lazy_result)
out_swapped = dask.compute(*lazy_results)
out_swapped = np.array(out_swapped).reshape(RR.shape[0],RR.shape[1],RR.shape[2])


out[~out_swapped] = np.nan

E_0,E_t,E_act = out[:,:,:,0],out[:,:,:,-1],np.nanmax(out,axis=-1)
dE = E_t - E_0

fig, ax = plt.subplots(1,2)
ax[0].imshow(np.log(np.flip(np.nanmean(E_act,axis=-1).T,axis=0)))
ax[1].imshow(np.flip(np.nanmean(dE,axis=-1).T,axis=0))
fig.show()


print(out_swapped.mean()*100,"% swapped (ensure this is 100%)")




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