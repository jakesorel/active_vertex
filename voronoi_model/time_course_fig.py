from voronoi_model.voronoi_model_periodic import *
import numpy as np
import matplotlib.pyplot as plt

li = 0
dir_name = "lattices"
x = np.loadtxt("%s/x_%d.txt"%(dir_name,li))
c_types = np.loadtxt("%s/c_types_%d.txt"%(dir_name,li)).astype(np.int64)
vor = Tissue()
vor.generate_cells(600)
vor.x = x
vor.x0 = vor.x
vor.n_c = vor.x0.shape[0]
vor.n_C = vor.n_c
vor.L = 9


p0 = 3.8333333333333335
r = 5
vor.v0 = 0.06833333
vor.Dr = 1e-1
beta = 0.05994843

vor.kappa_A = 1
vor.kappa_P = 1/r
vor.A0 = 1
vor.P0 = p0
vor.a = 0.3
vor.k = 1

vor.set_interaction(W = (2*beta*vor.P0/r)*np.array([[0, 1], [1, 0]]),c_types=c_types,pE=0.5)


vor.set_t_span(0.025,500)

vor.simulate()

def change_center(x,x0,y0,L):
    xx = x + np.array((x0,y0))
    return np.mod(xx,L)


vor.plot_scatter = False

nt = 4
t_range = np.linspace(0,vor.t_span.size-1,nt).astype(np.int64)
fig, ax = plt.subplots(1,nt)
for i, t in enumerate(t_range):
    plot_vor(vor,change_center(vor.x_save[t],-1.3,6.4,vor.L),ax[i])
    # vor.plot_vor(change_center(vor.x_save[t],-1.3,6.4,vor.L),ax[i])

    ax[i].axis("off")
# fig.savefig("analysis_plots/time_course p0=0.8333, v0 =0.06833333, beta = 0.007742636826811269.pdf",dpi=300)
fig.savefig("analysis_plots/time_course p0=0.8333, v0 =0.06833333, beta = 0.05994843.pdf",dpi=300)

fig, ax = plt.subplots()
percent = 0.20
x0 = vor.x_save[0]
inside_mask = (x0.min(axis=1)>vor.L*percent) * (x0.max(axis=1)<vor.L*(1-percent))
cids = np.nonzero(inside_mask)[0]
plot_vor(vor,x0,ax)

tri0 = vor.tri_save[0]
reduced_tri_ids = []
for ci in cids:
    tids = np.where(tri0 == ci)[0]
    if tids.size!=0:
        for ti in tids:
            reduced_tri_ids.append(ti)
reduced_tri_ids = np.unique(reduced_tri_ids)
tri = tri0[reduced_tri_ids]
for TRI in tri:
    for j in range(3):
        a, b = TRI[j], TRI[np.mod(j + 1, 3)]
        if (a >= 0) and (b >= 0):
            X = np.stack((x[a], x[b])).T
            ax.plot(X[0], X[1], color="black")

xysize=1.5
x0,y0 = 4.76,4.55
ax.set(xlim=(x0-xysize,x0+xysize),ylim=(y0-xysize,y0+xysize))
ax.axis("off")

fig.savefig("analysis_plots/explaining_model_schematic.pdf",dpi=300)