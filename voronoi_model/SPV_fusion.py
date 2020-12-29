from voronoi_model.voronoi_model_periodic import *
from voronoi_model.t1_functions import *
import numpy as np
import matplotlib.pyplot as plt



li = 0
dir_name = "fusion_lattices"
x = np.loadtxt("%s/x_%d.txt"%(dir_name,li))
# Is = np.loadtxt("%s/Is_%d.txt"%(dir_name,li))
# thetas = np.loadtxt("%s/thetas_%d.txt"%(dir_name,li))
c_types = np.loadtxt("%s/c_types_%d.txt"%(dir_name,li)).astype(np.int64)
vor = Tissue()
vor.generate_cells(600)
vor.x = x
vor.x0 = vor.x
vor.n_c = vor.x0.shape[0]
vor.n_C = vor.n_c
vor.L = 9


p0 = 4
r = 5
v0 = 0
vor.Dr = 1e-1
beta = 0.001

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

# Is, thetas = find_quartet(vor, Is[0], Is[1], thetas[0])
# Is, thetas = np.array(Is).astype(np.int64),np.array(thetas)


quartets = get_quartets(vor)
Is = quartets[1]
thetas = get_thetas(vor,Is)
vor.Is = Is
vor.set_t_span(0.01,50)
vor.n_t = vor.t_span.size

# vor.no_noise_time = int(vor.n_t/3)

generate_noise_fixed(vor,Is,thetas)
c_types[Is[0]] = 0
vor.set_interaction(W = (2*beta*vor.P0/r)*np.array([[0, 1], [1, 0]]),pE=0.5,c_types=c_types)


vor.v0 = v0*np.ones(vor.n_c).reshape(-1,1)
v0_chosen = 5e-2
for i in Is:
    vor.v0[i] = v0_chosen

vor.simulate_haltv0()
vor.plot_scatter = False


nt = 3
t_range = np.linspace(0,vor.t_span.size-1,nt).astype(np.int64)
fig, ax = plt.subplots(1,nt)
for i, t in enumerate(t_range):
    vor.plot_vor(vor.x_save[t],ax[i])
    ax[i].quiver(vor.x_save[t,Is,0],vor.x_save[t,Is,1],vor.noise[t,Is,0],vor.noise[t,Is,1])
    ax[i].axis("off")

fig.show()

# vor.animate(n_frames=30)

# n_islands = np.array(vor.get_num_islands(vor.n_t)).sum(axis=0)



J = np.zeros_like(vor.J)
kappa_A = np.zeros(vor.n_c)
kappa_P = np.zeros(vor.n_c)
for i in Is:
    kappa_A[i],kappa_P[i] = vor.kappa_A,vor.kappa_P

    J[i] = vor.J[i]
    J[:,i] = vor.J[:,i]
    # for j in Is:
    #     J[i,j] = vor.J[i,j]

# t1_time = np.nonzero(n_islands==2)[0].min()
tau = 200
t_eval_i = np.arange(-tau,tau+1,1).astype(np.int64)


# T_eval = t1_time+t_eval_i
T_eval = np.arange(vor.n_t).astype(np.int64)
x_save_fine = vor.x_save[T_eval]
energies = np.empty(T_eval.size)
for j, x in enumerate(x_save_fine):
    energies[j] = get_energy(vor,x,kappa_A,kappa_P,J,get_l_interface)


plt.close("all")
fig,ax = plt.subplots(figsize=(3.5,3))
ax.plot(T_eval,energies,color="k")
ax.set(ylabel=r"$\epsilon$",xlabel="Time")
fig.tight_layout()
fig.show()
fig.savefig("analysis_plots/eps vs time.pdf",dpi=300)
t1_time = energies.argmax()

tpm = np.array((-1,0,1))
dt = 1000
t1_t_range = dt*tpm + t1_time
fig, ax = plt.subplots(1,t1_t_range.size)
for i, t in enumerate(t1_t_range):
    vor.plot_vor(vor.x_save[t],ax[i])
    print(t)
    ax[i].axis("off")
fig.show()