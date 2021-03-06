from voronoi_model_periodic import *
import sys

def make_directory(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


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


def get_energy(self, x, tris,kappa_A, kappa_P, J, get_l_interface):
    self.x = x
    self._triangulate_periodic(self.x)
    # self.tris = tris
    # self.Cents = x[self.tris]
    # self.vs = self.get_vertex_periodic()
    self.assign_vertices()
    A = self.get_A_periodic(self.neighbours, self.vs)
    P = self.get_P_periodic(self.neighbours, self.vs)
    l_int = get_l_interface(self.n_v, self.n_c, self.neighbours, self.vs, self.CV_matrix, self.L)
    energy = kappa_A * (A - self.A0) ** 2 + kappa_P * (P - self.P0) ** 2 + np.sum(l_int * J,axis=0)
    return energy


def get_energy_sim(beta,Id,li,cll_i,t1_type="forward"):
    dir_name = "energy_barrier"

    make_directory("energy_barrier/energies/%s"%t1_type)
    make_directory("energy_barrier/energies_mobile_i/%s"%t1_type)
    make_directory("energy_barrier/energies_tot/%s"%t1_type)

    make_directory("energy_barrier/mobile_i/%s"%t1_type)
    tri_save = np.load("energy_barrier/tri_save/%s/%d_%d_%d.npz" % (t1_type,Id, li, cll_i))["arr_0"]
    tri_save = tri_save.reshape(tri_save.shape[0], -1, 3)
    x_save = np.load("energy_barrier/x_save/%s/%d_%d_%d.npz" % (t1_type,Id, li, cll_i))["arr_0"]
    x_save = x_save.reshape(x_save.shape[0], -1, 2)
    c_types = np.load("energy_barrier/c_types/%s/%d_%d_%d.npz" % (t1_type,Id, li, cll_i))["arr_0"]

    vor = Tissue()
    vor.generate_cells(600)
    vor.x = x_save[0]
    vor.x0 = vor.x
    vor.n_c = vor.x0.shape[0]
    vor.n_C = vor.n_c
    vor.L = 9
    vor._triangulate_periodic(vor.x)



    p0 = 3.9
    r = 10
    vor.v0 = 0
    vor.Dr = 1e-1
    beta = beta

    vor.kappa_A = 1
    vor.kappa_P = 1/r
    vor.A0 = 1
    vor.P0 = p0
    vor.a = 0.3
    vor.k = 0

    vor.set_interaction(W = beta*np.array([[0, 1], [1, 0]]),c_types=c_types,pE=0.5)

    vor.set_t_span(0.025, 100)
    vor.n_t = vor.t_span.size

    vor.c_types = c_types
    vor.tri_save = tri_save
    vor.x_save = x_save
    vor.initialize_t1(cll_i,t1_type=t1_type)
    energies = np.zeros((vor.n_t,vor.n_c))
    for i, (x,tris) in enumerate(zip(x_save,tri_save)):
        energies[i] = get_energy(vor, x, tris,vor.kappa_A, vor.kappa_P, vor.J, get_l_interface)
    np.savez_compressed("energy_barrier/energies/%s/%d_%d_%d.npz" % (t1_type,Id, li,cll_i),
                        energies)
    np.savez_compressed("energy_barrier/energies_tot/%s/%d_%d_%d.npz" % (t1_type,Id, li,cll_i),energies.sum(axis=1))
    np.savez_compressed("energy_barrier/energies_mobile_i/%s/%d_%d_%d.npz" % (t1_type, Id, li, cll_i),
                        energies[:,vor.mobile_i])
    np.savetxt("energy_barrier/mobile_i/%s/%d_%d_%d.txt"% (t1_type,Id, li,cll_i),[vor.mobile_i])

if __name__ == "__main__":
    Id = int(sys.argv[1])
    N = int(sys.argv[2])
    rep = int(sys.argv[3])
    beta_range = np.logspace(-3,-1,N)
    rep_range = np.arange(rep)
    BB,RR = np.meshgrid(beta_range,rep_range,indexing="ij")
    beta = BB.take(Id)
    li = RR.take(Id)

    n_t1_f = get_n_t1_cells(beta, 0, Id, 0, t1_type="forward")
    n_t1_r = get_n_t1_cells(beta, 0, Id, 0, t1_type="reverse")


    for cll_i in range(n_t1_r):
        try:
            get_energy_sim(beta, Id, li,cll_i, t1_type="reverse")
            print(Id,cll_i,"reverse")
        except:
            a = 1
    for cll_i in range(n_t1_f):
        try:
            get_energy_sim(beta, Id, li,cll_i, t1_type="forward")
            print(Id,cll_i,"forward")
        except:
            a = 1







