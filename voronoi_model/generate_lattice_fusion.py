from voronoi_model_periodic import *
from t1_functions import *
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def simulate(X):
    beta, Id = X
    dir_name = "lattices"
    x = np.loadtxt("%s/x_%d.txt"%(dir_name,0))
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


    A_mask = vor.x[:,0]<vor.L/2
    c_types = np.zeros(vor.n_c,dtype=np.int64)
    c_types[~A_mask] = 1
    vor.c_types = c_types
    vor._triangulate_periodic(vor.x)
    vor.assign_vertices()


    vor.set_interaction(W = (2*beta*vor.P0/r)*np.array([[0, 1], [1, 0]]),pE=0.5,c_types=c_types)
    vor.set_t_span(0.025,100)


    vor.v0 = v0

    vor.simulate(equiangulate=False)

    dir_name = "fusion_lattices"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    np.savetxt("%s/x_%d.txt" % (dir_name, Id), vor.x)
    np.savetxt("%s/c_types_%d.txt" % (dir_name, Id), c_types)

    return vor.x,c_types


if __name__ == "__main__":
    dir_name = "fusion_lattices"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    Id = int(sys.argv[1])
    N = int(sys.argv[2])
    # p0_range = np.linspace(3.5,4,N)
    beta_range = np.logspace(-3,-1,N)
    # PP,BB = np.meshgrid(p0_range,beta_range,indexing="ij")
    beta = beta_range.take(Id)
    x,c_types = simulate((beta,Id))
