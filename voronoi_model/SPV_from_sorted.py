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
vor.v0 = 0.1
vor.Dr = 1e-1
beta = 0.00278256

vor.kappa_A = 1
vor.kappa_P = 1/r
vor.A0 = 1
vor.P0 = p0
vor.a = 0.3
vor.k = 1


A_mask = vor.x[:,0]<vor.L/2
c_types = np.zeros(vor.n_c,dtype=np.int64)
c_types[~A_mask] = 1
vor.set_interaction(W = (2*beta*vor.P0/r)*np.array([[0, 1], [1, 0]]),pE=0.5,c_types=c_types)


vor.set_t_span(0.025,500)

vor.simulate()
vor.plot_scatter = False


nt = 4
t_range = np.linspace(0,vor.t_span.size-1,nt).astype(np.int64)
fig, ax = plt.subplots(1,nt)
for i, t in enumerate(t_range):
    vor.plot_vor(vor.x_save[t],ax[i])
    ax[i].axis("off")
fig.savefig("analysis_plots/p0 = 3.833 v0 = 0.1, beta = 0.0027 time course.pdf",dpi=300)




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
vor.v0 = 0.1
vor.Dr = 1e-1
beta = 0.05994842503189409

vor.kappa_A = 1
vor.kappa_P = 1/r
vor.A0 = 1
vor.P0 = p0
vor.a = 0.3
vor.k = 1


A_mask = vor.x[:,0]<vor.L/2
c_types = np.zeros(vor.n_c,dtype=np.int64)
c_types[~A_mask] = 1
vor.set_interaction(W = (2*beta*vor.P0/r)*np.array([[0, 1], [1, 0]]),pE=0.5,c_types=c_types)


vor.set_t_span(0.025,500)

vor.simulate()
vor.plot_scatter = False


nt = 4
t_range = np.linspace(0,vor.t_span.size-1,nt).astype(np.int64)
fig, ax = plt.subplots(1,nt)
for i, t in enumerate(t_range):
    vor.plot_vor(vor.x_save[t],ax[i])
    ax[i].axis("off")
fig.savefig("analysis_plots/p0 = 3.833 v0 = 0.1, beta = 0.05994 time course.pdf",dpi=300)
