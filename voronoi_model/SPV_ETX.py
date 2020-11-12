from voronoi_model.voronoi_model_periodic import *
import numpy as np
import matplotlib.pyplot as plt


vor = Tissue()
vor.generate_cells(300)
vor.make_init_boundary(20,0.15)
alpha = 0.08
vor.b_tension = 2*alpha
gamma_ET,gamma_TT,gamma_EX,gamma_TX,gamma_XX = 1,0,1.1,1.1,1.3
vor.set_interaction_boundary_ETX(W = alpha*np.array([[0, gamma_ET,gamma_EX], [gamma_ET, gamma_TT,gamma_TX],[gamma_EX,gamma_TX,gamma_XX]]),pE=0.5,pX=0)
vor._triangulate(vor.x0)
vor.triangulate(vor.x0)
vor.assign_vertices()
vor.x0 = vor.check_boundary(vor.x0)

vor.kappa_B = 0#0.05
vor.l_b0 = 0.1
vor.zeta = 0#0.05

# vor.P0 = 3.00
p0 = 4.2
vor.A0 = 0.86
vor.P0 = p0*np.sqrt(vor.A0)
print(vor.P0)

vor.v0 = 5*10**-1
vor.Dr = 40
vor.kappa_A = 0.1
vor.kappa_P = 0.05
vor.a = 0.05
vor.k = 2

vor.cols = "red","blue","green","white"
vor.plot_scatter = False

vor.set_t_span(0.025,50)
vor.simulate_boundary(b_extra=10,print_every=2000)

vor.animate(n_frames=50,an_type="boundary")


def get_theta(ri,rj,rk):
    rij = ri - rj
    rik = ri - rk
    return np.arccos(np.dot(rij,rik)/(np.linalg.norm(rij)*np.linalg.norm(rik)))

