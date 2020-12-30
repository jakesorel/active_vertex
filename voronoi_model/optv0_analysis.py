import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import bisplrep,bisplev

dir_name = "optv0s"
N = 10
p0_range = np.linspace(3.5, 4, N)
beta_range = np.logspace(-3, -1, N)
repn = 8
optv0s = np.zeros((N,N,repn))*np.nan
k = 0
for i in range(N):
    for j in range(N):
        optv0 = np.loadtxt("%s/%d.txt"%(dir_name,k))
        optv0s[i,j,:optv0.size] = optv0
        k+=1


optv0s_mean = np.nanmean(optv0s,axis=-1)
plt.imshow(np.log10(optv0s_mean))
plt.show()
nfine = 200
p0_spacefine,beta_spacefine = np.linspace(p0_range.min(),p0_range.max(), nfine),np.logspace(np.log10(beta_range.min()),np.log10(beta_range.max()),nfine)
PPs,lBBs = np.meshgrid(p0_range,np.log10(beta_range),indexing="ij")
z = bisplev(p0_spacefine, np.log10(beta_spacefine), bisplrep(PPs.ravel(), lBBs.ravel(),np.log10(optv0s_mean).ravel(), s=0.1))

fig, ax = plt.subplots(figsize=(3.5,3))
extent = [p0_range.min(),p0_range.max(),np.log10(beta_range.min()),np.log10(beta_range.max())]
aspect = (extent[1]-extent[0])/(extent[3]-extent[2])
ax.imshow(np.flip(z.T,axis=0),aspect=aspect,extent=extent)
ax.set(xlabel=r"$p_0$",ylabel=r"$log_{10} \ \beta$")
sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmax=z.max(), vmin=z.min()))
sm._A = []
cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.085, aspect=10, orientation="vertical")
cl.set_label(r"$log_{10} \ v_{crit}$")
fig.subplots_adjust(top=0.8, bottom=0.2, left=0.2, right=0.8)
fig.savefig("analysis_plots/v_crit.pdf",dpi=300)

fig, ax = plt.subplots(figsize=(3.5,3))
extent = [p0_range.min(),p0_range.max(),np.log10(beta_range.min()),np.log10(beta_range.max())]
aspect = (extent[1]-extent[0])/(extent[3]-extent[2])
ax.imshow(np.flip(optv0s_mean.T,axis=0),aspect=aspect,extent=extent)
ax.set(xlabel=r"$p_0$",ylabel=r"$log_{10} \ \beta$")
sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmax=z.max(), vmin=z.min()))
sm._A = []
cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.085, aspect=10, orientation="vertical")
cl.set_label(r"$log_{10} \ v_{crit}$")
fig.subplots_adjust(top=0.8, bottom=0.2, left=0.2, right=0.8)
fig.savefig("analysis_plots/v_crit_raw.pdf",dpi=300)

