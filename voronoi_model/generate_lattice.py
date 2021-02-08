import numpy as np
from voronoi_model.voronoi_model_periodic import *
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
# import dask
# from dask.distributed import Client
import seaborn as sb

"""
To ensure we are measuring bone-fide sorting (i.e. genuine topological changes in cell-neighbourhoods), we want to 
initialize lattices such that the number of "islands" is maximized. 

Further we want to 
"""


def generate_lattice():
    vor = Tissue()
    vor.generate_cells(600)
    vor.make_init_balanced(9, noise=0.0005)
    p0 = 3.81
    r = 10
    vor.v0 = 5e-3
    vor.Dr = 1e-1
    beta = 0.1

    vor.kappa_A = 1
    vor.kappa_P = 1 / r
    vor.A0 = 1
    vor.P0 = p0
    vor.a = 0.3
    vor.k = 1

    vor.set_interaction(W=(2 * beta * vor.P0 / r) * np.array([[0, 1], [1, 0]]), pE=0, randomize=True)

    vor.set_t_span(0.025, 150)

    vor.simulate()
    return vor.x


def true_randomize(x, pE=0.5, n_trial=1000):
    vor = Tissue()
    vor.generate_cells(600)
    vor.x = x
    vor.x0 = vor.x
    vor.n_c = vor.x0.shape[0]
    vor.n_C = vor.n_c
    vor.L = 9
    vor.set_interaction(W=np.array([[0, 1], [1, 0]]), pE=pE, randomize=True)

    vor._triangulate_periodic(x)
    c_type_mat = np.zeros((n_trial, vor.n_c),dtype=np.int64)
    n_islands = np.zeros(n_trial)
    for i in range(n_trial):
        np.random.shuffle(vor.c_types)
        c_type_mat[i] = vor.c_types
        Adj = np.zeros((vor.n_c, vor.n_c), dtype=np.float32)
        Adj[vor.tris, np.roll(vor.tris, -1, axis=1)] = 1
        AdjA = Adj[vor.c_types == 0][:, vor.c_types == 0]
        AdjB = Adj[vor.c_types == 1][:, vor.c_types == 1]
        A_islands, B_islands = connected_components(csgraph=csr_matrix(AdjA), directed=False)[0], \
                               connected_components(csgraph=csr_matrix(AdjB), directed=False)[0]
        n_islands[i] = A_islands + B_islands
    return c_type_mat[np.where(n_islands == n_islands.max())[0][0]]



def null_model(x, pE=0.5, n_trial=1000):
    vor = Tissue()
    vor.generate_cells(600)
    vor.x = x
    vor.x0 = vor.x
    vor.n_c = vor.x0.shape[0]
    vor.n_C = vor.n_c
    vor.L = 9
    vor.set_interaction(W=np.array([[0, 1], [1, 0]]), pE=pE, randomize=True)

    vor._triangulate_periodic(x)
    c_type_mat = np.zeros((n_trial, vor.n_c),dtype=np.int64)
    n_islands = np.zeros(n_trial)
    for i in range(n_trial):
        np.random.shuffle(vor.c_types)
        c_type_mat[i] = vor.c_types
        Adj = np.zeros((vor.n_c, vor.n_c), dtype=np.float32)
        Adj[vor.tris, np.roll(vor.tris, -1, axis=1)] = 1
        AdjA = Adj[vor.c_types == 0][:, vor.c_types == 0]
        AdjB = Adj[vor.c_types == 1][:, vor.c_types == 1]
        A_islands, B_islands = connected_components(csgraph=csr_matrix(AdjA), directed=False)[0], \
                               connected_components(csgraph=csr_matrix(AdjB), directed=False)[0]
        n_islands[i] = A_islands + B_islands
    return n_islands




def find_topologically_sorted(x, pE=0.5, n_trial=1000):
    vor = Tissue()
    vor.generate_cells(600)
    vor.x = x
    vor.x0 = vor.x
    vor.n_c = vor.x0.shape[0]
    vor.n_C = vor.n_c
    vor.L = 9
    vor.set_interaction(W=np.array([[0, 1], [1, 0]]), pE=pE, randomize=True)

    vor._triangulate_periodic(x)
    c_type_mat = np.zeros((n_trial, vor.n_c),dtype=np.int64)
    n_islands = np.zeros(n_trial)
    for i in range(n_trial):
        np.random.shuffle(vor.c_types)
        c_type_mat[i] = vor.c_types
        Adj = np.zeros((vor.n_c, vor.n_c), dtype=np.float32)
        Adj[vor.tris, np.roll(vor.tris, -1, axis=1)] = 1
        AdjA = Adj[vor.c_types == 0][:, vor.c_types == 0]
        AdjB = Adj[vor.c_types == 1][:, vor.c_types == 1]
        A_islands, B_islands = connected_components(csgraph=csr_matrix(AdjA), directed=False)[0], \
                               connected_components(csgraph=csr_matrix(AdjB), directed=False)[0]
        n_islands[i] = A_islands + B_islands
    return c_type_mat[np.where(n_islands == n_islands.min())[0][0]]


def generate_sorted_lattice():
    vor = Tissue()
    vor.generate_cells(600)
    vor.make_init_balanced(15, noise=0.0005)
    p0 = 3.81
    r = 10
    vor.v0 = 5e-3
    vor.Dr = 1e-1
    beta = 0.1

    vor.kappa_A = 1
    vor.kappa_P = 1 / r
    vor.A0 = 1
    vor.P0 = p0
    vor.a = 0.3
    vor.k = 1

    A_mask = vor.x[:, 0] < vor.L / 2
    c_types = np.zeros(vor.n_c, dtype=np.int64)
    c_types[~A_mask] = 1
    vor.set_interaction(W=(2 * beta * vor.P0 / r) * np.array([[0, 1], [1, 0]]), pE=0.5, randomize=True)

    vor.set_t_span(0.025, 150)

    vor.simulate()
    return vor.x


def make_random_lattice(pE=0.5, n_trial=1000):
    x = generate_lattice()
    c_types = true_randomize(x, pE, n_trial)
    return x, c_types



def make_sorted_lattice(pE=0.5, n_trial=1000):
    x = generate_lattice()
    c_types = find_topologically_sorted(x, pE, n_trial)
    return x, c_types




if __name__ == "__main__":
    n_rep = 25
    dir_name = "lattices"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


    # n_slurm_tasks = 8#int(os.environ["SLURM_NTASKS"])
    # client = Client(threads_per_worker=1, n_workers=n_slurm_tasks,memory_limit="2GB")
    # lazy_results = []
    out = []
    for i in range(n_rep):
        # lazy_result = dask.delayed(make_random_lattice)()
        # lazy_results.append(lazy_result)
        out.append(make_random_lattice())
    # out = dask.compute(*lazy_results)

    for i, outt in enumerate(out):
        x,c_types = outt
        np.savetxt("%s/x_%d.txt"%(dir_name,i),x)
        np.savetxt("%s/c_types_%d.txt"%(dir_name,i),c_types)
    #
    #
    #
    # sorted_dir_name = "sorted_lattices"
    # if not os.path.exists(sorted_dir_name):
    #     os.makedirs(sorted_dir_name)
    #
    # lazy_results = []
    # for i in range(n_rep):
    #     lazy_result = dask.delayed(make_sorted_lattice)()
    #     lazy_results.append(lazy_result)
    # out = dask.compute(*lazy_results)
    #
    # for i, outt in enumerate(out):
    #     x,c_types = outt
    #     np.savetxt("%s/x_%d.txt"%(sorted_dir_name,i),x)
    #     np.savetxt("%s/c_types_%d.txt"%(sorted_dir_name,i),c_types)
    #
    #

    ###null_model
    x = generate_lattice()
    n_islands = null_model(x,n_trial=10000)
    fig, ax = plt.subplots(figsize=(3.5,2))
    sb.histplot(n_islands,bins=np.arange(np.amax(n_islands)),ax=ax,discrete=True,stat="density")
    ax.set(xlabel=r"$N_{clust}$",ylabel=r"$P(X = N_{clust})$")
    fig.subplots_adjust(top=0.8, bottom=0.25, left=0.2, right=0.8)
    fig.savefig("paper_plots/Fig1/prob_dist_fn_N_clust.pdf",dpi=300)


