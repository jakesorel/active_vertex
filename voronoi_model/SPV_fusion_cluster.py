from voronoi_model_periodic import *
from t1_functions import *
import numpy as np
import matplotlib.pyplot as plt
import sys


def get_n_quartets(X):
    p0,v0_chosen,beta,Id = X
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


    vor.c_types = c_types
    vor._triangulate_periodic(vor.x)
    vor.triangulate_periodic(vor.x)
    vor.assign_vertices()

    quartets = get_quartets(vor)
    return quartets.shape[0]

def get_lattice_id(p0,beta):
    N = int(sys.argv[2])
    p0_range = np.linspace(3.5,4,N)
    beta_range = np.logspace(-3,-1,N)
    PP,BB = np.meshgrid(p0_range,beta_range,indexing="ij")
    return np.where((np.abs(PP.ravel()-p0)<1e-16)*(np.abs(BB.ravel()-beta)<1e-16))[0][0]

def simulate(X):
    n_quartets = get_n_quartets(X)
    for rep in range(n_quartets):
        p0,v0_chosen,beta,Id = X
        lId = get_lattice_id(p0,beta)
        dir_name = "fusion_lattices"
        x = np.loadtxt("%s/x_%d.txt"%(dir_name,lId))
        c_types = np.loadtxt("%s/c_types_%d.txt"%(dir_name,lId)).astype(np.int64)
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
        Is = quartets[rep]
        thetas = get_thetas(vor, Is)
        vor.Is = Is
        vor.set_t_span(0.01,20)
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


        n_islands = np.array(vor.get_num_islands(2)).sum(axis=0)[-1]

        np.savez_compressed("fusion/%d_%d.npz"%(Id,rep),n_islands=n_islands,energies=energies)


if __name__ == "__main__":
    Id = int(sys.argv[1])
    N = int(sys.argv[2])
    p0_range = np.linspace(3.5,4,N)
    v0_range = np.linspace(1e-2,1,N)
    beta_range = np.logspace(-3,-1,N)

    PP,VV,BB = np.meshgrid(p0_range, v0_range,beta_range,indexing="ij")
    p0,v0,beta = PP.take(Id),VV.take(Id),BB.take(Id)
    simulate((p0,v0,beta,Id))