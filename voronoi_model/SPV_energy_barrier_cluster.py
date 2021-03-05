from voronoi_model_periodic import *
import sys

def make_directory(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

make_directory("sorted_lattices")

def make_lattice(li,beta,Id):
    dir_name = "lattices"
    x = np.loadtxt("%s/x_%d.txt" % (dir_name, li))
    vor = Tissue()
    vor.generate_cells(600)
    vor.x = x
    vor.x0 = vor.x
    vor.n_c = vor.x0.shape[0]
    vor.n_C = vor.n_c
    vor.L = 9

    p0 = 3.9
    r = 10
    vor.v0 = 0
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
    vor.set_interaction(W=beta * np.array([[0, 1], [1, 0]]), c_types=c_types, pE=0.5)

    vor.set_t_span(0.025, 100)
    vor.n_t = vor.t_span.size

    vor.simulate(equiangulate=True)
    np.savetxt("sorted_lattices/x_%d.txt" % (Id), vor.x)
    np.savetxt("sorted_lattices/c_types_%d.txt" % (Id), c_types)



def get_n_t1_cells(beta,v0,Id,cll_i,t1_type="forward"):
    dir_name = "sorted_lattices"
    x = np.loadtxt("%s/x_%d.txt"%(dir_name,Id))
    c_types = np.loadtxt("%s/c_types_%d.txt"%(dir_name,Id)).astype(np.int64)
    vor = Tissue()
    vor.generate_cells(600)
    vor.x = x
    vor.x0 = vor.x
    vor.n_c = vor.x0.shape[0]
    vor.n_C = vor.n_c
    vor.L = 9


    p0 = 3.9
    r = 10
    vor.v0 = v0
    vor.Dr = 1e-1
    beta = beta

    vor.kappa_A = 1
    vor.kappa_P = 1/r
    vor.A0 = 1
    vor.P0 = p0
    vor.a = 0.3
    vor.k = 0

    vor.set_interaction(W = beta*np.array([[0, 1], [1, 0]]),c_types=c_types,pE=0.5)

    vor.set_t_span(0.025, 50)
    vor.n_t = vor.t_span.size
    vor.no_movement_time = 10

    vor.initialize_t1(cll_i,t1_type=t1_type)
    return vor.n_t1_cells

def run_simulation(beta,v0,Id,cll_i,t1_type="forward"):
    dir_name = "sorted_lattices"
    x = np.loadtxt("%s/x_%d.txt"%(dir_name,Id))
    c_types = np.loadtxt("%s/c_types_%d.txt"%(dir_name,Id)).astype(np.int64)
    vor = Tissue()
    vor.generate_cells(600)
    vor.x = x
    vor.x0 = vor.x
    vor.n_c = vor.x0.shape[0]
    vor.n_C = vor.n_c
    vor.L = 9


    p0 = 3.9
    r = 10
    vor.v0 = v0
    vor.Dr = 1e-1
    beta = beta

    vor.kappa_A = 1
    vor.kappa_P = 1/r
    vor.A0 = 1
    vor.P0 = p0
    vor.a = 0.3
    vor.k = 0

    vor.set_interaction(W = beta*np.array([[0, 1], [1, 0]]),c_types=c_types,pE=0.5)

    vor.set_t_span(0.025, 50)
    vor.n_t = vor.t_span.size
    vor.no_movement_time = 10

    vor.initialize_t1(cll_i,t1_type=t1_type)
    vor.simulate_t1(equiangulate=True,print_every=1e5)
    return vor


def save_simulation(vor,li,Id,cll_i,t1_type="forward"):
    np.savez_compressed("energy_barrier/tri_save/%s/%d_%d_%d.npz" % (t1_type,Id, li,cll_i),
                        vor.tri_save.reshape(vor.n_t, 3 * vor.n_v))
    np.savez_compressed("energy_barrier/x_save/%s/%d_%d_%d.npz" % (t1_type,Id, li, cll_i),
                        vor.x_save.reshape(vor.n_t, 2 * vor.n_c))
    np.savez_compressed("energy_barrier/c_types/%s/%d_%d_%d.npz" % (t1_type,Id, li, cll_i), vor.c_types)
    np.savetxt("energy_barrier/t1_time/%s/%d_%d_%d.txt"% (t1_type,Id, li, cll_i),[vor.t1_time])

def get_v0_opt(beta,li,Id,cll_i,t1_type="forward",n_iter = 5):
    fs = False
    v0_chosen = 0.1
    i = 0
    v0_range = np.arange(0.1,5,0.1)
    n_max = v0_range.size-1
    while (i<n_max)*(fs==False):
        sim = run_simulation(beta, v0_range[i], Id, cll_i, t1_type)
        fs = (sim.t1_time !=False)

        # n_islands = np.array(sim.get_num_islands(2)).sum(axis=0)[-1]
        # if (t1_type == "forward")*(n_islands==2)+(t1_type=="reverse")*(n_islands==3):
        #     fs = True
        if fs==False:
            i+=1
    fs = False
    v0_chosen = v0_range[i] - 0.09
    i = 0
    while (i<11)*(fs==False):
        sim = run_simulation(beta, v0_chosen, Id, cll_i, t1_type)
        fs = (sim.t1_time !=False)
        if fs==False:
            i+=1
            v0_chosen += 0.01
            print(v0_chosen)
    fs = False
    v0_chosen = v0_chosen - 0.009
    print(v0_chosen)
    i = 0
    while (i<11)*(fs==False):
        sim = run_simulation(beta, v0_chosen, Id, cll_i, t1_type)
        fs = (sim.t1_time !=False)
        if fs==False:
            i+=1
            v0_chosen += 0.001
            print(v0_chosen)
    # return v0_chosen,sim
    if fs == True:
        save_simulation(sim, li,Id, cll_i, t1_type)
        np.savetxt("energy_barrier/opt_v0/%s/%d_%d_%d.txt"% (t1_type,Id, li, cll_i),[v0_chosen])




if __name__ == "__main__":
    Id = int(sys.argv[1])
    N = int(sys.argv[2])
    rep = int(sys.argv[3])
    beta_range = np.logspace(-3,-1,N)
    rep_range = np.arange(rep)
    BB,RR = np.meshgrid(beta_range,rep_range,indexing="ij")
    beta = BB.take(Id)
    li = RR.take(Id)
    make_lattice(li,beta,Id)
    make_directory("energy_barrier")
    make_directory("energy_barrier/x_save")
    make_directory("energy_barrier/x_save/forward")
    make_directory("energy_barrier/x_save/reverse")

    make_directory("energy_barrier/c_types")
    make_directory("energy_barrier/c_types/forward")
    make_directory("energy_barrier/c_types/reverse")

    make_directory("energy_barrier/tri_save")
    make_directory("energy_barrier/tri_save/forward")
    make_directory("energy_barrier/tri_save/reverse")

    make_directory("energy_barrier/opt_v0")
    make_directory("energy_barrier/opt_v0/forward")
    make_directory("energy_barrier/opt_v0/reverse")

    make_directory("energy_barrier/t1_time")
    make_directory("energy_barrier/t1_time/forward")
    make_directory("energy_barrier/t1_time/reverse")

    n_t1_f = get_n_t1_cells(beta, 0, Id, 0, t1_type="forward")
    n_t1_r = get_n_t1_cells(beta, 0, Id, 0, t1_type="reverse")

    for cll_i in range(n_t1_f):
        get_v0_opt(beta, li,Id, cll_i, t1_type="forward")
    for cll_i in range(n_t1_r):
        get_v0_opt(beta, li, Id, cll_i, t1_type="reverse")

