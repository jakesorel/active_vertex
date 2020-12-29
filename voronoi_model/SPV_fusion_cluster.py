from voronoi_model.voronoi_model_periodic import *
from voronoi_model.t1_functions import *
import numpy as np
import matplotlib.pyplot as plt
import sys

"""
To do:

Need to equilibrate simulations FOR EACH PARAMETER SET. 

Need to fix the angles. May be a much easier way to choose cells etc. 
"""


def simulate(X):
    p0,v0_chosen,beta,Id,li = X
    dir_name = "fusion_lattices"
    x = np.loadtxt("%s/x_%d.txt"%(dir_name,Id))
    c_types = np.loadtxt("%s/c_types_%d.txt"%(dir_name,Id)).astype(np.int64)
    vor = Tissue()
    vor.generate_cells(600)
    vor.x = x
    vor.x0 = vor.x
    vor.n_c = vor.x0.shape[0]
    vor.n_C = vor.n_c
    vor.L = 9


    p0 = p0
    r = 5
    v0 = 0
    vor.Dr = 1e-1
    beta = beta

    vor.kappa_A = 1
    vor.kappa_P = 1/r
    vor.A0 = 1
    vor.P0 = p0
    vor.a = 0.3
    vor.k = 1


    vor.c_types = c_types
    vor._triangulate_periodic(vor.x)
    vor.triangulate_periodic(vor.x)
    vor.assign_vertices()

    quartets = get_quartets(vor)
    Is = quartets[1]
    thetas = get_thetas(vor, Is)
    vor.Is = Is
    vor.set_t_span(0.025,50)
    vor.n_t = vor.t_span.size

    generate_noise_fixed(vor,Is,thetas)
    c_types[Is[0]] = 0
    vor.set_interaction(W = (2*beta*vor.P0/r)*np.array([[0, 1], [1, 0]]),pE=0.5,c_types=c_types)

    vor.v0 = v0*np.ones(vor.n_c).reshape(-1,1)
    for i in Is:
        vor.v0[i] = v0_chosen

    vor.simulate_haltv0()

    J = np.zeros_like(vor.J)
    kappa_A = np.zeros(vor.n_c)
    kappa_P = np.zeros(vor.n_c)
    for i in Is:
        kappa_A[i], kappa_P[i] = vor.kappa_A, vor.kappa_P

        J[i] = vor.J[i]
        J[:, i] = vor.J[:, i]

    T_eval = np.arange(vor.n_t).astype(np.int64)
    x_save_fine = vor.x_save[T_eval]
    energies = np.empty(T_eval.size)
    for j, x in enumerate(x_save_fine):
        energies[j] = get_energy(vor, x, kappa_A, kappa_P, J, get_l_interface)


    vor.plot_scatter = False


    nt = 3
    t_range = np.linspace(0,vor.t_span.size-1,nt).astype(np.int64)
    fig, ax = plt.subplots(1,nt)
    for i, t in enumerate(t_range):
        vor.plot_vor(vor.x_save[t],ax[i])
        ax[i].quiver(vor.x_save[t,Is,0],vor.x_save[t,Is,1],vor.noise[t,Is,0],vor.noise[t,Is,1])
        ax[i].axis("off")
    fig.savefig("analysis_plots/quivers_%d_%d.pdf"%(Id,li),dpi=300)

    fig, ax = plt.subplots()
    ax.plot(energies)
    fig.savefig("analysis_plots/energies_%d_%d.pdf"%(Id,li),dpi=300)


    # vor.animate(n_frames=30)

    n_islands = np.array(vor.get_num_islands(2)).sum(axis=0)[-1]

    # np.savez_compressed("fusion/%d_%d.npz",n_islands=n_islands,energies=energies)

for i in range(8):
    simulate((3.8,1.5*1e-1,0.02,i,0))



if __name__ == "__main__":
    Id = int(sys.argv[1])
    N = int(sys.argv[2])
    rep = int(sys.argv[3])
    p0_range = np.linspace(3.5,4,N)
    v0_range = np.linspace(5e-3,1e-1,N)
    beta_range = np.logspace(-3,-1,N)

    PP,VV,BB = np.meshgrid(p0_range, v0_range,beta_range,indexing="ij")
    p0,v0,beta = PP.take(Id),VV.take(Id),BB.take(Id)
    for repn in range(rep):
        simulate((p0,v0,beta,Id,repn))