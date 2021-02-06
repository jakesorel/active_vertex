import numpy as np
import matplotlib.pyplot as plt
import dask
from dask.distributed import Client

"""
Assorted code snippets for extracting values from the analysis file for further plotting etc.

Most of this code is copied elsewhere into functional scripts
"""

n_slurm_tasks = 8
client = Client(threads_per_worker=1, n_workers=n_slurm_tasks, memory_limit="1GB")
N = 10
rep = 6
runs = 2
# p0_range = np.linspace(3.5, 4, N)
# v0_range = np.linspace(5e-3, 1e-1, N)
# beta_range = np.linspace(0, 0.3)
v0_range = np.linspace(5e-3, 1e-1, N)
beta_range = np.logspace(-3, -1, N)
rep_range = np.arange(rep)
run_range = np.arange(runs)
VV, BB,RR = np.meshgrid(v0_range, beta_range,rep_range,run_range, indexing="ij")
ID_mat = np.arange(N**2).astype(int).reshape(N,N)
ID_mat = np.stack([np.stack([ID_mat for i in range(rep)],axis=2) for i in range(runs)])


def get_L_star(X):
    Id, Rep = X
    try:
        FILE = np.load("analysis/%d_%d.npz" % (Id,Rep))
        return FILE["L_star"]
    except FileNotFoundError:
        return np.ones(100)*np.nan




inputs = np.array([ID_mat.ravel(),RR.ravel()]).T
inputs = inputs.astype(np.int64)
lazy_results = []
for inputt in inputs:
    lazy_result = dask.delayed(get_L_star)(inputt)
    lazy_results.append(lazy_result)
out = dask.compute(*lazy_results)
out = np.array(out).reshape(RR.shape[0],RR.shape[1],RR.shape[2],RR.shape[3],100)

vmin,vmax = np.percentile(out,5),np.percentile(out,95)
for i_rep in range(rep):
    fig, ax = plt.subplots(5,2)
    ti = -1
    ax = ax.ravel()
    rep_out = out[:,:,:,i_rep,ti]

    for i in range(10):
        ax[i].imshow(np.flip(rep_out[:,:,i].T,axis=1),vmin=vmin,vmax=vmax)
        ax[i].set_title(r"$\beta = 10^{%.3f}$"%beta_range[i],fontsize=4)
    fig.tight_layout()
    fig.savefig("analysis_plots/beta_rep=%d.pdf"%i_rep)


fig, ax = plt.subplots(5,2)
ti = -1
ax = ax.ravel()
rep_out = out[:,:,:,:,ti].mean(axis=-1)
vmin,vmax = np.percentile(rep_out,5),np.percentile(rep_out,100)

for i in range(10):
    ax[i].imshow(np.flip(rep_out[:,:,i].T,axis=1),vmin=vmin,vmax=vmax)
    ax[i].set_title(r"$\beta = 10^{%.3f}$"%beta_range[i],fontsize=4)
fig.tight_layout()
fig.savefig("analysis_plots/beta_mean.pdf")



vmin,vmax = np.percentile(out,5),np.percentile(out,95)
for i_rep in range(rep):
    fig, ax = plt.subplots(5,2)
    ti = -1
    ax = ax.ravel()
    rep_out = out[:,:,:,i_rep,ti]

    for i in range(10):
        ax[i].imshow(np.flip(rep_out[:,i,:].T,axis=1),vmin=vmin,vmax=vmax)
        ax[i].set_title(r"$v_0 = %.3f$"%v0_range[i],fontsize=4)
    fig.tight_layout()
    fig.savefig("analysis_plots/v0_rep=%d.pdf"%i_rep)


fig, ax = plt.subplots(5,2)
ti = -1
ax = ax.ravel()
rep_out = out[:,:,:,:,ti].mean(axis=-1)
vmin,vmax = np.percentile(rep_out,5),np.percentile(rep_out,100)

for i in range(10):
    ax[i].imshow(np.flip(rep_out[:,:,i].T,axis=1),vmin=vmin,vmax=vmax)
    ax[i].set_title(r"$\beta = 10^{%.3f}$"%beta_range[i],fontsize=4)
fig.tight_layout()
fig.savefig("analysis_plots/mean.pdf")





"""
NUM ISLANDS 

"""



def get_n_islands(X):
    Id, Rep = X
    try:
        FILE = np.load("analysis/%d_%d.npz" % (Id,Rep))
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

vmin,vmax = np.percentile(n_islands_tot,5),np.percentile(n_islands_tot,95)
for i_rep in range(rep):
    fig, ax = plt.subplots(5,2)
    ti = -1
    ax = ax.ravel()
    rep_out = n_islands_tot[:,:,:,i_rep,ti]

    for i in range(10):
        ax[i].imshow(np.flip(rep_out[:,:,i].T,axis=1),vmin=vmin,vmax=vmax)
        ax[i].set_title(r"$\beta = 10^{%.3f}$"%beta_range[i],fontsize=4)
    fig.tight_layout()
    fig.savefig("analysis_plots/n_islands_beta_rep=%d.pdf"%i_rep)



fig, ax = plt.subplots(5,2)
ti = -1
ax = ax.ravel()
rep_out = n_islands_tot[:,:,:,:,ti].mean(axis=-1)
vmin,vmax = np.percentile(rep_out,5),np.percentile(rep_out,100)

for i in range(10):
    ax[i].imshow(np.flip(rep_out[:,:,i].T,axis=1),vmin=vmin,vmax=vmax)
    ax[i].set_title(r"$\beta = 10^{%.3f}$"%beta_range[i],fontsize=4)
fig.tight_layout()
fig.savefig("analysis_plots/n_islands_meanbeta.pdf")


fig, ax = plt.subplots(5,2)
ti = -1
ax = ax.ravel()
rep_out = n_islands_tot[:,:,:,:,ti].mean(axis=-1)
vmin,vmax = np.percentile(rep_out,5),np.percentile(rep_out,100)

for i in range(10):
    ax[i].imshow(np.flip(rep_out[:,i,:].T,axis=1),vmin=vmin,vmax=vmax)
    ax[i].set_title(r"$v_0 = %.3f$" % v0_range[i], fontsize=4)
fig.tight_layout()
fig.savefig("analysis_plots/n_islands_meanv0.pdf")



fig, ax = plt.subplots(5,2)
ti = -1
ax = ax.ravel()
rep_out = n_islands_tot[:,:,:,:,ti].mean(axis=-1)
vmin,vmax = np.percentile(rep_out,5),np.percentile(rep_out,100)

for i in range(10):
    ax[i].imshow(np.flip(rep_out[i,:,:].T,axis=1),vmin=vmin,vmax=vmax)
    ax[i].set_title(r"$p_0 = %.3f$" % p0_range[i], fontsize=4)
fig.tight_layout()
fig.savefig("analysis_plots/n_islands_meanp0.pdf")


n_islands_tot_mean = n_islands_tot[:,:,:,:,-1].mean(axis=-1)
from scipy.interpolate import bisplrep,bisplev
nfine = 200
p0_spacefine, v0_spacefine,beta_spacefine = np.linspace(p0_range.min(),p0_range.max(), nfine), np.linspace(v0_range.min(), v0_range.max(), nfine),np.logspace(np.log10(beta_range.min()),np.log10(beta_range.max()),nfine)
# PPf,VVf,BBf = np.meshgrid(p0_spacefine,v0_spacefine,beta_spacefine,indexing="ij")

PPf,VVf = np.meshgrid(p0_spacefine,v0_spacefine,indexing="ij")
PPs,VVs = np.meshgrid(p0_range,v0_range,indexing="ij")

for i in range(10):
    z = bisplev(p0_spacefine,v0_spacefine, bisplrep(PPs.ravel(),VVs.ravel(),n_islands_tot_mean[:,:,i].ravel(),s=1))


VVf,BBf = np.meshgrid(v0_spacefine,beta_spacefine,indexing="ij")
VVs,lBBs = np.meshgrid(v0_range,np.log10(beta_range),indexing="ij")

for i in range(10):
    z = bisplev(v0_spacefine, np.log10(beta_spacefine),bisplrep(VVs.ravel(),lBBs,n_islands_tot_mean[i,:,:].ravel(),s=1))
    plt.imshow(z)
    plt.savefig("analysis_plots/z_p0=%.3f.pdf"%p0_range[i])
    plt.close("all")

from scipy.interpolate import UnivariateSpline

ni_min,ni_max = np.percentile(n_islands_tot_mean,5),np.percentile(n_islands_tot_mean,95)
ni_med = np.median(n_islands_tot_mean)
ni_mid = (ni_max+ni_min)/2
deviation = np.log10((n_islands_tot_mean - ni_mid)**2 + 1 - ((n_islands_tot_mean - ni_mid)**2).min())

weights = (deviation.max() - deviation)/(deviation.max() - deviation.min())
# weights = 1.0*(deviation<1)
# weights = (n_islands_tot_mean-ni_min)**2 * (n_islands_tot_mean-ni_max)**2
VVs,lBBs = np.meshgrid(v0_range,np.log10(beta_range),indexing="ij")



PP, VV, BB = np.meshgrid(p0_range, v0_range, beta_range, indexing="ij")


from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.gca(projection='3d')
for i in range(10):
    spl = UnivariateSpline(VVs.ravel(), lBBs.ravel(),weights[i,:,:].ravel(), k=2)
    plt.plot(v0_spacefine,spl(v0_spacefine),np.repeat(p0_range[i],v0_spacefine.size),color="k")
    # spl2 = UnivariateSpline(VVs.ravel(), lBBs.ravel(), weights[i, :, :].ravel(), k=2)
    # plt.plot(v0_spacefine,spl(v0_spacefine),np.repeat(p0_range[i],v0_spacefine.size),color="k")
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
thresh = 0.85
X,Y,Z = PP[weights>thresh],VV[weights>thresh],np.log10(BB[weights>thresh])
surf = ax.scatter(X, Y, Z)
ax.set(xlabel="p0",ylabel="v0",zlabel="log beta")
fig.show()


import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"
fig = go.Figure(data=[go.Surface(z=Z,x=X,y=Y)])

fig.update_layout(title='Mt Bruno Elevation', autosize=False,
                  width=500, height=500)

fig.show()


import plotly.express as px
df = px.data.iris()
fig = px.scatter_3d(z=Z,x=X,y=Y)
fig.show()



import scipy as sp
import scipy.interpolate
spline = sp.interpolate.Rbf(X,Z,Y,smooth=1, episilon=5)

PP,lBB = np.meshgrid(p0_spacefine,np.log10(beta_spacefine),indexing="ij")
yy = spline(PP,lBB)
fig = go.Figure(data=[go.Surface(z=zz,x=xx[0],y=yy[:,0])])

fig.update_layout(title='Mt Bruno Elevation', autosize=False,
                  width=500, height=500)

fig.show()
file_content = np.column_stack((PP.ravel(),VV.ravel(),BB.ravel(),n_islands_tot_mean.ravel()))
np.savetxt("param_scan.txt",file_content)


from scipy.interpolate import interp2d
VVs,lBBs = np.meshgrid(v0_range,np.log10(beta_range),indexing="ij")

interp = interp2d(VVs,lBBs,weights[:,:,4],kind="linear")
betaa = interp(v0_spacefine,np.log10(beta_spacefine))
plt.imshow(betaa,vmin=np.percentile(betaa,10),vmax=np.percentile(betaa,99))
plt.show()

def ellipsoid(x0,y0,z0,a,b,c):
    return



"""beta = 4th"""
beta = beta_range[1]
locs = np.where(BB==beta)
Ids = ID_mat[locs]
Reps = RR[locs]

L_stars = np.array([get_L_star(Id,Rep) for Id,Rep in zip(Ids,Reps)])
L_stars = L_stars.reshape(8,8,8,100)
L_star_mean = np.nanmean(L_stars,axis=2)

PPs, VVs = np.meshgrid(p0_range, v0_range, indexing="ij")



fig, ax = plt.subplots(figsize=(5,4))
vals = L_star_mean[:,:,-1].ravel()
levels = np.linspace(vals.min(),np.percentile(vals,60),20)
vals[vals>=levels.max()] = levels.max()-1e-3
cnt = ax.tricontourf(PPs.ravel(),VVs.ravel(),vals,levels=levels)
for c in cnt.collections:
    c.set_edgecolor("face")
sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmax=levels.max(), vmin=levels.min()))
sm._A = []
cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.1, aspect=10, orientation="vertical")
cl.set_label(r"$L*$")
fig.subplots_adjust(top=0.8, bottom=0.2, left=0.2, right=0.8)
ax.set(xlabel=r"$p_0$",ylabel=r"$v_0$")
fig.savefig("L_star.pdf")



"""
Analysis of shapes
"""

def exp_model(t_span,f,s,tau):
    return f + (s-f)*np.exp(-t_span/tau)

t_span = np.arange(0,500,500/100)

from scipy.optimize import curve_fit,Bounds

(f,s,tau),__ = curve_fit(exp_model,t_span,L)


def get_taus(Id,Rep):
    try:
        FILE = np.load("analysis/%d_%d.npz" % (Id,Rep))
        L_star = FILE["L_star"]
        (f, s, tau), __ = curve_fit(exp_model, t_span, L_star,[5,12,1000])
        return tau
    except FileNotFoundError:
        return np.nan


def lin_model(t_span,a,b):
    return a + b*t_span


def get_grads(Id,Rep,p):
    try:
        FILE = np.load("analysis/%d_%d.npz" % (Id,Rep))
        L_star = FILE["L_star"]
        (a,b), __ = curve_fit(lin_model, t_span[int(t_span.size * p):], L_star[int(t_span.size * p):],[5,2])
        return b
    except FileNotFoundError:
        return np.nan



beta = beta_range[1]
locs = np.where(BB==beta)
Ids = ID_mat[locs]
Reps = RR[locs]

Taus = np.array([get_grads(Id,Rep,0.0) for Id,Rep in zip(Ids,Reps)])
Taus = Taus.reshape(8,8,8)
# Taus = np.log10(Taus)
plt.hist(Taus.ravel(),bins=100)
plt.show()
Taus = np.nanmean(Taus,axis=2)
plt.imshow(np.flip(Taus.T,axis=0))
plt.show()

plt.cm.plasma()

plt.plot(L_stars[L_stars[:,-1]<7.5][::30].T,color="black")
plt.plot(L_stars[L_stars[:,-1]>15][::30].T,color="red")
plt.show()

L_norm = (L_stars.T - np.mean(L_stars[:,:10],axis=1))/(np.mean(L_stars[:,-10:],axis=1) - np.mean(L_stars[:,:10],axis=1))
# L_norm = L_stars.T-np.mean(L_stars[:,:10],axis=1)
half_t = np.argmin((L_norm-0.5)**2,axis=0)



fig, ax = plt.subplots()
ax.plot(L_norm[:,half_t<20][:,::5],color="red")
ax.plot(L_norm[:,half_t>40][:,::5],color="black")
ax.set(ylim=(0,1))
fig.show()

Half_t = half_t.astype(np.float).reshape(8,8,8)
Half_t[Half_t==0] = np.nan
Half_t = np.nanmin(Half_t,axis=2)
plt.imshow(Half_t)
plt.show()

PPs, VVs,RRs = np.meshgrid(p0_range, v0_range,rep_range, indexing="ij")

plt.plot(L_stars[np.where((PPs==p0_range[0])&(VVs==v0_range[3]))].T,color="black")
plt.plot(L_stars[np.where((PPs==p0_range[-1])&(VVs==v0_range[3]))].T,color="red")
plt.show()
plt.scatter(VVs.ravel(),half_t.ravel())
plt.show()


plt.show()


PPs, VVs = np.meshgrid(p0_range, v0_range, indexing="ij")



fig, ax = plt.subplots(figsize=(5,4))
vals = Taus.ravel().copy()
levels = np.linspace(vals.min(),np.percentile(vals,50),20)
vals[vals>=levels.max()] = levels.max()-1e-3
cnt = ax.tricontourf(PPs.ravel(),VVs.ravel(),vals,levels=levels)
for c in cnt.collections:
    c.set_edgecolor("face")
sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmax=levels.max(), vmin=levels.min()))
sm._A = []
cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.1, aspect=10, orientation="vertical")
cl.set_label(r"$L*$")
fig.subplots_adjust(top=0.8, bottom=0.2, left=0.2, right=0.8)
ax.set(xlabel=r"$p_0$",ylabel=r"$v_0$")
fig.show()
fig.savefig("L_star.pdf")



"""Mean self"""



def get_mean_self(Id,Rep):
    try:
        FILE = np.load("analysis/%d_%d.npz" % (Id,Rep))
        return FILE["mean_self"]
    except FileNotFoundError:
        return np.ones(100)*np.nan



"""beta = 4th mean self"""
beta = beta_range[1]
locs = np.where(BB==beta)
Ids = ID_mat[locs]
Reps = RR[locs]

MS = np.array([get_mean_self(Id,Rep) for Id,Rep in zip(Ids,Reps)])
MS = MS.reshape(8,8,8,100)
MS_mean = np.nanmean(MS,axis=2)

PPs, VVs = np.meshgrid(p0_range, v0_range, indexing="ij")

fig, ax = plt.subplots(figsize=(5,4))
vals = MS_mean[:,:,-1].ravel()
levels = np.linspace(vals.min(),np.percentile(vals,60),20)
vals[vals>=levels.max()] = levels.max()-1e-3
cnt = ax.tricontourf(PPs.ravel(),VVs.ravel(),vals,levels=levels)
for c in cnt.collections:
    c.set_edgecolor("face")
sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmax=levels.max(), vmin=levels.min()))
sm._A = []
cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.1, aspect=10, orientation="vertical")
cl.set_label(r"$\phi_{self}$")
fig.subplots_adjust(top=0.8, bottom=0.2, left=0.2, right=0.8)
ax.set(xlabel=r"$p_0$",ylabel=r"$v_0$")
fig.savefig("mean_self.pdf")





def get_n_island(Id,Rep):
    try:
        FILE = np.load("analysis/%d_%d.npz" % (Id,Rep))
        return FILE["n_islands"]
    except FileNotFoundError:
        return np.ones((2,100))*np.nan



"""beta = 4th n_island"""
beta = beta_range[1]
locs = np.where(BB==beta)
Ids = ID_mat[locs]
Reps = RR[locs]

NI = np.array([get_n_island(Id,Rep) for Id,Rep in zip(Ids,Reps)])
NI = NI.reshape(8,8,8,2,100)
NI = np.sum(NI,axis=3)
NI_mean = np.nanmean(NI,axis=2)


PPs, VVs = np.meshgrid(p0_range, v0_range, indexing="ij")

fig, ax = plt.subplots(figsize=(5,4))
vals = NI_mean[:,:,-1].ravel()
levels = np.linspace(vals.min(),vals.max(),20)
vals[vals>=levels.max()] = levels.max()-1e-3
cnt = ax.tricontourf(PPs.ravel(),VVs.ravel(),vals,levels=levels)
for c in cnt.collections:
    c.set_edgecolor("face")
sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmax=levels.max(), vmin=levels.min()))
sm._A = []
cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.1, aspect=10, orientation="vertical")
cl.set_label(r"$n island$")
fig.subplots_adjust(top=0.8, bottom=0.2, left=0.2, right=0.8)
ax.set(xlabel=r"$p_0$",ylabel=r"$v_0$")
fig.show()
fig.savefig("n_island.pdf")








plt.imshow(L_star_mean[:,:,-1])
plt.show()



L_stars = np.array([get_L_star(Id) for Id in range(N**3)])

fig, ax = plt.subplots(figsize=(5,3))
ax.plot(L_stars[20],label="Jammed")
ax.plot(L_stars[51],label="No D.A.")
ax.plot(L_stars[-1],label="Unjammed")
ax.legend()
ax.set(xlabel="Time",ylabel="Autocorrelation \n Lengthscale")
fig.show()

v0 = v0 = v0_range[0]

mask = VV==v0
Id_sel = np.arange(N**3)[mask.ravel()]

plt.tricontourf(PP.ravel()[Id_sel],BB.ravel()[Id_sel],L_stars[Id_sel,-1])
plt.show()

VVs,BBs = np.meshgrid(v0_range,beta_range,indexing="ij")
vals = np.zeros_like(VVs)
for i, L_star in enumerate(L_stars):
    if PP.take(i) == p0_range[0]:
        i,j = np.where((VVs==VV.take(i))&(BBs==BB.take(i)))
        vals[i,j] = L_star[-1]

plt.imshow(vals[:,::6],aspect="auto")
plt.show()


def get_mean_self(Id):
    FILE = np.load("analysis/%d.npz" % Id)
    return FILE["mean_self"]

mean_self = np.array([get_mean_self(Id) for Id in range(N**3)])

VVs,BBs = np.meshgrid(v0_range,beta_range,indexing="ij")
vals = np.zeros_like(VVs)
for i, mean_S in enumerate(mean_self):
    if PP.take(i) == p0_range[0]:
        i,j = np.where((VVs==VV.take(i))&(BBs==BB.take(i)))
        vals[i,j] = mean_S[-1]

plt.imshow(vals,aspect="auto")
plt.show()

v0 = v0_range[-1]

mask = VV==v0

Id_sel = np.arange(N**3)[mask.ravel()]

levels = np.linspace(vals.min(),vals.max(),6)
plt.tricontourf(VVs.ravel(),BBs.ravel(),vals.ravel(),levels = levels)
plt.show()



def get_n_bound(Id):
    FILE = np.load("analysis/%d.npz" % Id)
    return FILE["n_bound"]

n_bound = np.array([get_n_bound(Id) for Id in range(N**3)])

v0 = v0_range[-1]

mask = VV==v0

Id_sel = np.arange(N**3)[mask.ravel()]

plt.tricontourf(PP.ravel()[Id_sel],BB.ravel()[Id_sel],n_bound[Id_sel,-1])
plt.show()





def get_mean_self(Id):
    FILE = np.load("analysis/%d.npz" % Id)
    return FILE["mean_self"]

mean_self = np.array([get_mean_self(Id) for Id in range(N**3)])


mask = BB==beta_range[-1]

Id_sel = np.arange(N**3)[mask.ravel()]

plt.tricontourf(PP.ravel()[Id_sel],VV.ravel()[Id_sel],mean_self[Id_sel,-1])
plt.show()


for i in range(512):
    # if BB.take(i) == 0:
    if BB.ravel()[i] == 0:
        plt.plot(mean_self[i],color="blue")
    # if VV.take(i) == v0_range[-1]:
plt.show()

MS = mean_self.reshape(8,8,8,mean_self.shape[-1])
for val in MS[:,:,0,-1]:
    plt.plot(val)
plt.show()