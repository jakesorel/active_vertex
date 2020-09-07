from voronoi_model_periodic import *
import numpy as np
import sys

def run_simulation(X):
    p0, v0, beta, Id,rep = X
    vor = Tissue()
    vor.generate_cells(600)
    vor.make_init(14,noise = 0.05)
    p0 = p0
    r = 5
    vor.v0 = v0
    vor.Dr = 0.1
    beta = beta

    vor.kappa_A = 1
    vor.kappa_P = 1/r
    vor.A0 = 1
    vor.P0 = p0
    vor.a = 0.3
    vor.k = 1

    vor.set_interaction(W = (2*beta*vor.P0/r)*np.array([[0, 1], [1, 0]]),pE=0.5)

    vor.set_t_span(0.025,500)

    vor.simulate()

    np.savez_compressed("tri_save/%d_%d.npz"%(Id,rep),vor.tri_save.reshape(vor.n_t,3*vor.n_v))
    np.savez_compressed("x_save/%d_%d.npz"%(Id,rep),vor.x_save.reshape(vor.n_t,2*vor.n_c))
    np.savez_compressed("c_types/%d_%d.npz"%(Id,rep),vor.c_types)


if __name__ == "__main__":
    Id = int(sys.argv[1])
    N = int(sys.argv[2])
    rep = int(sys.argv[3])
    p0_range = np.linspace(3.5,4,N)
    v0_range = np.linspace(5e-3,1e-1,N)
    beta_range = np.linspace(0,0.3,N)

    PP,VV,BB = np.meshgrid(p0_range, v0_range,beta_range,indexing="ij")
    p0,v0,beta = PP.take(Id),VV.take(Id),BB.take(Id)
    for repn in range(rep):
        run_simulation((p0,v0,beta,Id,repn))

