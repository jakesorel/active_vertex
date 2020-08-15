from voronoi_model.voronoi_model_periodic import *
import numpy as np
import matplotlib.pyplot as plt


vor = Tissue()
vor.generate_cells(300)
vor.make_init_boundary(20,0.15)
alpha = 0.05
vor.b_tension = 0.16
vor.set_interaction_boundary(W = alpha*np.array([[0, 1], [1, 0]]),pE=0.5)
vor._triangulate(vor.x0)
vor.triangulate(vor.x0)
vor.assign_vertices()
vor.x0 = vor.check_boundary(vor.x0)

vor.kappa_B = 0.3
vor.l_b0 = 0.1
vor.zeta = 0#0.05

# vor.P0 = 3.00
p0 = 4.2 #3.81
vor.A0 = 0.86
vor.P0 = p0*np.sqrt(vor.A0)
print(vor.P0)

vor.v0 = 1e-1
vor.Dr = 1
vor.kappa_A = 0.2
vor.kappa_P = 0.05
vor.a = 0.3
vor.k = 2

vor.cols = "red","blue","white"
vor.plot_scatter = False

vor.set_t_span(0.025,40)
vor.simulate_boundary(b_extra=10,print_every=2000)

vor.animate(n_frames=50,an_type="boundary")
