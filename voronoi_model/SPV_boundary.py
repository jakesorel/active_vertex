from voronoi_model.voronoi_model_periodic import *
import numpy as np
import matplotlib.pyplot as plt



vor = Tissue()
vor.generate_cells(300)
vor.make_init_boundary(22,0.15)
alpha = 0.03
vor.set_interaction_boundary(W = alpha*np.array([[0, 1], [1, 0]]),pE=0.55,Wb=[0.3,0.05])
vor._triangulate(vor.x0)
vor.triangulate(vor.x0)
vor.assign_vertices()
vor.x0 = vor.check_boundary(vor.x0)
vor.Gamma_bound = 5e-3
vor.kappa_B = 0#0.05
vor.l_b0 = 0.3
vor.zeta = 0#0.05

# vor.P0 = 3.00
p0 = 3.95
vor.A0 = 0.9
vor.P0 = p0*np.sqrt(vor.A0)
print(vor.P0)

vor.v0 = 3e-1
vor.Dr = 40
vor.kappa_A = 0.1
vor.kappa_P = 0.05
vor.a = 0.2
vor.k = 2

vor.cols = "red","green","white"
vor.plot_scatter = False

vor.set_t_span(0.025,250)
vor.simulate_boundary(b_extra=10,print_every=2000)

vor.plot_forces = False
vor.animate(n_frames=50,an_type="boundary",tri=False)
