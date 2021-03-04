from voronoi_model.voronoi_model_periodic import *
from voronoi_model.t1_functions import *
li = 1
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


p0 = 3.9
r = 10
vor.v0 = 0.001
vor.Dr = 1e-1
beta = 1e-3

vor.kappa_A = 1
vor.kappa_P = 1/r
vor.A0 = 1
vor.P0 = p0
vor.a = 0.3
vor.k = 0

A_mask = vor.x[:, 0] < vor.L / 2
c_types = np.zeros(vor.n_c, dtype=np.int64)
c_types[~A_mask] = 1
vor.set_interaction(W = beta*np.array([[0, 1], [1, 0]]),c_types=c_types,pE=0.5)

vor.set_t_span(0.025, 50)
vor.n_t = vor.t_span.size
vor.no_movement_time = 10

vor.initialize_t1(0,t1_type="forward")

self = vor
self.assign_vertices()
tris,c_types,v_neighbours,neighbours,L,vs,mobile_i,t1_type,n_c,CV_matrix = self.tris,self.c_types,self.v_neighbours,self.neighbours,self.L,self.vs,self.mobile_i,self.t1_type,self.n_c,self.CV_matrix
vor.plot_scatter = False

tri_i,tri_k,complete = get_t1_dir(tris,c_types,v_neighbours,neighbours,L,vs,mobile_i,self.n_c,self.CV_matrix,t1_type)

fig, ax = plt.subplots()
self.plot_vor(vor.x,ax)

ax.scatter(vor.x[vor.tris[tri_i],0],vor.x[vor.tris[tri_i],1],color="green")
ax.scatter(vor.x[vor.tris[vor.v_neighbours[tri_i,tri_k]],0],vor.x[vor.tris[vor.v_neighbours[tri_i,tri_k]],1],color="red")
for i in range(self.n_c):
    ax.text(self.x[i,0],self.x[i,1],i)

fig.show()

vor.simulate_t1(equiangulate=True)
#
# t1_time_i = int(np.round(vor.t1_time/vor.dt))
# dt_i = 400
# n_plot = 2
# t_span_sample = np.arange(-dt_i*n_plot+t1_time_i,(2*n_plot+1)*dt_i+t1_time_i,dt_i)
# fig, ax = plt.subplots(1,t_span_sample.size,figsize=(15,4))
# for i, ti in enumerate(t_span_sample):
#     vor.plot_vor(vor.x_save[ti],ax=ax[i])
#     ax[i].axis("off")
# fig.show()
# fig.savefig("plots/time_course_t1.pdf")


vor.animate(n_frames=30)

