from voronoi_model_periodic import *
from t1_functions import *
import numpy as np
import matplotlib.pyplot as plt
import sys

"""
THis has been played with substantially since the parameter scan. Refer to Git 
"""

def get_n_quartets(X,C_TYPES):
    vor = Tissue()
    vor.generate_cells(600)
    vor.x = X.copy()
    vor.x0 = vor.x
    vor.n_c = vor.x0.shape[0]
    vor.n_C = vor.n_c
    vor.L = 9

    c_types = C_TYPES.copy()
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
    beta, Id = X

    def generate_lattice(X):
        beta, Id = X
        dir_name = "lattices"
        x = np.loadtxt("%s/x_%d.txt" % (dir_name, Id))
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
        vor.kappa_P = 1 / r
        vor.A0 = 1
        vor.P0 = p0
        vor.a = 0.3
        vor.k = 0

        A_mask = vor.x[:, 0] < vor.L / 2
        c_types = np.zeros(vor.n_c, dtype=np.int64)
        c_types[~A_mask] = 1
        vor.c_types = c_types
        vor._triangulate_periodic(vor.x)
        vor.assign_vertices()

        vor.set_interaction(W=beta * np.array([[0, 1], [1, 0]]), pE=0.5, c_types=c_types)
        vor.set_t_span(0.025, 100)

        vor.v0 = v0

        vor.simulate(equiangulate=False)
        return vor,vor.c_types,vor.x_save[-1]

    vor, X,C_TYPES = generate_lattice(X)
    quartets = get_quartets(vor)
    n_quartets = get_n_quartets(X,C_TYPES)
    def evaluate_forward(v0_chosen,quartet,X,C_TYPES):
        vor = Tissue()
        vor.generate_cells(600)
        vor.x = X.copy
        c_types = C_TYPES.copy()
        vor.x0 = vor.x
        vor.n_c = vor.x0.shape[0]
        vor.n_C = vor.n_c
        vor.L = 9


        r = 10
        v0 = 0
        vor.Dr = 1e-1

        vor.kappa_A = 1
        vor.kappa_P = 1/r
        vor.A0 = 1
        vor.P0 = 3.9
        vor.a = 0.3
        vor.k = 0


        vor.c_types = c_types
        vor._triangulate_periodic(vor.x)
        vor.triangulate_periodic(vor.x)
        vor.assign_vertices()

        Is = quartet
        thetas = get_thetas(vor, Is)
        vor.Is = Is
        vor.set_t_span(0.025,30)
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
        return vor.v0.sum()==0

    def get_v0_opt_forward(rep):
        v0_range = np.arange(0.1,1.1,0.1)
        fused = np.array([evaluate_forward(v0_chosen,rep) for v0_chosen in v0_range])
        first_fused1 = np.where(fused)[0][0]
        v0_range2 = np.arange(v0_range[first_fused1]-0.09,v0_range[first_fused1]+0.01,0.01)
        fused2 = np.array([evaluate_forward(v0_chosen,rep) for v0_chosen in v0_range2])
        first_fused2 = np.where(fused2)[0][0]
        v0opt = v0_range2[first_fused2]
        return v0opt

    v0_opt_forward = np.array([get_v0_opt_forward(rep) for rep in range(n_quartets)])
    dir_name = "optv0s"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    np.savetxt("%s/%d.txt"%(dir_name,Id),v0_opts)

if __name__ == "__main__":
    Id = int(sys.argv[1])
    N = int(sys.argv[2])
    beta_range = np.logspace(-3,-1,N)

    beta = beta_range.take(Id)
    simulate((beta,Id))