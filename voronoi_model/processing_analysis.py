import numpy as np
import matplotlib.pyplot as plt
N = 10
rep = 8
# p0_range = np.linspace(3.5, 4, N)
# v0_range = np.linspace(5e-3, 1e-1, N)
# beta_range = np.linspace(0, 0.3)
p0_range = np.linspace(3.5, 4, N)
v0_range = np.linspace(5e-3, 1e-1, N)
beta_range = np.logspace(-3, -1, N)
rep_range = np.arange(rep)
PP, VV, BB,RR = np.meshgrid(p0_range, v0_range, beta_range,rep_range, indexing="ij")
ID_mat = np.arange(N**3).astype(int).reshape(N,N,N)
ID_mat = np.stack([ID_mat for i in range(rep)],axis=3)

# PP,VV,BB = PP[:,:,:8],VV[:,:,:8],BB[:,:,:8]

def get_L_star(Id,Rep):
    try:
        FILE = np.load("analysis/%d_%d.npz" % (Id,Rep))
        return FILE["L_star"]
    except FileNotFoundError:
        return np.ones(100)*np.nan



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