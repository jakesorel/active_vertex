from voronoi_model_periodic import *
from t1_functions import *
import numpy as np
import matplotlib.pyplot as plt
import sys

"""
THis has been played with substantially since the parameter scan. Refer to Git 
"""

def get_n_quartets(lId):
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
#
# def plot_tcourse(vor):
#     nt = 3
#     t_range = np.linspace(0, vor.t_span.size - 1, nt).astype(np.int64)
#     fig, ax = plt.subplots(1, nt)
#     for i, t in enumerate(t_range):
#         vor.plot_vor(vor.x_save[t], ax[i])
#         ax[i].quiver(vor.x_save[t, vor.Is, 0], vor.x_save[t, vor.Is, 1], vor.noise[t, vor.Is, 0], vor.noise[t, vor.Is, 1])
#         ax[i].axis("off")
#
#     fig.show()

def simulate(X):
    beta, Id = X

    # p0,beta = 4,1e-3
    # sys.argv[2] = 10
    # rep = 0

    lId = Id#get_lattice_id(p0, beta)
    n_quartets = get_n_quartets(lId)
    n_quartets = np.min((8,n_quartets))
    for rep in range(n_quartets):
        dir_name = "fusion_lattices"
        x = np.loadtxt("%s/x_%d.txt"%(dir_name,lId))
        c_types = np.loadtxt("%s/c_types_%d.txt"%(dir_name,lId)).astype(np.int64)
        v0_chosen = np.loadtxt("optv0s/%d.txt"%lId)[rep]
        vor = Tissue()
        vor.generate_cells(600)
        vor.x = x
        vor.x0 = vor.x
        vor.n_c = vor.x0.shape[0]
        vor.n_C = vor.n_c
        vor.L = 9


        p0 = 3.9
        r = 10
        v0 = 0
        vor.Dr = 1e-1
        beta = beta

        vor.kappa_A = 1
        vor.kappa_P = 1/r
        vor.A0 = 1
        vor.P0 = p0
        vor.a = 0.3
        vor.k = 0


        vor.c_types = c_types
        vor._triangulate_periodic(vor.x)
        vor.triangulate_periodic(vor.x)
        vor.assign_vertices()

        quartets = get_quartets(vor)
        Is = quartets[rep]
        thetas = get_thetas(vor, Is)
        vor.Is = Is
        vor.set_t_span(0.025,500)
        vor.n_t = vor.t_span.size
        vor.no_noise_time = 400

        generate_noise_fixed(vor,Is,thetas)
        c_types[Is[0]] = 0
        vor.set_interaction(W = beta*np.array([[0, 1], [1, 0]]),pE=0.5,c_types=c_types)

        vor.v0 = v0*np.ones(vor.n_c).reshape(-1,1)
        for i in Is:
            vor.v0[i] = v0_chosen
        vor.v0_orig = vor.v0.copy()
        vor.haltwait = 1
        vor.simulate_haltv0()

        # plot_tcourse(vor)

        J = np.zeros_like(vor.J)
        kappa_A = np.zeros(vor.n_c)
        kappa_P = np.zeros(vor.n_c)
        for i in Is:
            kappa_A[i]= vor.kappa_A
            kappa_P[i] = vor.kappa_P

            J[i] = vor.J[i]
            J[:, i] = vor.J[:, i]

        T_eval = np.arange(vor.no_noise_time*4).astype(np.int64)
        x_save_sample = vor.x_save[T_eval]
        energies = np.zeros(T_eval.size)
        vor.x = x_save_sample[0]
        for j, x in enumerate(x_save_sample):
            energies[j] = get_energy(vor, x, kappa_A, kappa_P, J, get_l_interface)

        T_eval2 = np.linspace(vor.no_noise_time*4,vor.n_t-1,100).astype(np.int64)
        x_save_sample2 = vor.x_save[T_eval2]
        energies2 = np.zeros(T_eval2.size)
        vor.x = x_save_sample2[0]
        for j, x in enumerate(x_save_sample2):
            energies2[j] = get_energy(vor, x, kappa_A, kappa_P, J, get_l_interface)

        # T_eval_all = np.concatenate((T_eval,T_eval2))
        energies_all = np.concatenate((energies,energies2))


        # plt.close("all")
        # fig, ax = plt.subplots()
        # ax.plot(T_eval_all,energies_all)
        # fig.show()



        n_islands = np.array(vor.get_num_islands(2)).sum(axis=0)[-1]
        swapped = np.sum(vor.v0)==0
        dir_name = "fusion_using_optv0"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        np.savez_compressed("fusion_using_optv0/%d_%d.npz"%(Id,rep),n_islands=n_islands,energies_all=energies_all,swapped=swapped)
        print("done")

if __name__ == "__main__":
    Id = int(sys.argv[1])
    N = int(sys.argv[2])
    # p0_range = np.linspace(3.5,4,N)
    beta_range = np.logspace(-3,-1,N)

    # PP,BB = np.meshgrid(p0_range,beta_range,indexing="ij")
    # p0,beta = PP.take(Id),BB.take(Id)
    beta = beta_range.take(Id)
    simulate((beta,Id))