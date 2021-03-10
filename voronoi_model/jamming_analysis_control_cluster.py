import numpy as np
import matplotlib.pyplot as plt
from voronoi_model_periodic import *
from scipy.stats import linregress
# plt.rcParams.update({'pdf.fonttype': 42})
import sys

def make_directory(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def initialize_vor(Id, i, run):
    vor = Tissue()
    vor.generate_cells(600)
    vor.make_init_balanced(9, noise=0.0005)
    p0 = 3.9
    r = 10
    vor.v0 = 5e-3
    vor.Dr = 1e-1
    beta = 0.1

    vor.kappa_A = 1
    vor.kappa_P = 1 / r
    vor.A0 = 1
    vor.P0 = p0
    vor.a = 0.3
    vor.k = 0

    vor.set_interaction(W=(2 * beta * vor.P0 / r) * np.array([[0, 1], [1, 0]]), pE=0.5)

    vor.set_t_span(0.025, 500)



    tri_save = np.load("from_unsorted_control/tri_save/%d_%d_%d.npz" % (Id, i, run))["arr_0"]
    tri_save = tri_save.reshape(tri_save.shape[0], -1, 3)
    x_save = np.load("from_unsorted_control/x_save/%d_%d_%d.npz" % (Id, i, run))["arr_0"]
    x_save = x_save.reshape(x_save.shape[0], -1, 2)
    c_types = np.load("from_unsorted_control/c_types/%d_%d_%d.npz" % (Id, i, run))["arr_0"]

    vor.n_t = tri_save.shape[0]
    vor.n_c = x_save.shape[1]
    vor.n_v = tri_save.shape[1]

    vor.c_types = c_types
    vor.x_save = x_save
    vor.tri_save = tri_save
    return vor



def get_x2(T,L,x_save):
    x_s = x_save[T:]
    disp = np.mod(x_s - x_s[0] + L/2,L)-L/2
    x2 = disp[:,:,0]**2 + disp[:,:,1]**2
    return x2

def get_disp(T,L,x_save):
    x_s = x_save[T:]
    disp = np.mod(x_s - x_s[0] + L/2,L)-L/2
    # x2 = disp[:,:,0]**2 + disp[:,:,1]**2
    return disp +x_s[0]

def get_D(vor,T = 15000):
    x2 = get_x2(T, vor.L, vor.x_save)
    x2_mean = x2.mean(axis=1)
    D = linregress(vor.t_span[T:],x2_mean).slope/(2*2)
    return D,x2_mean

@jit(nopython=True)
def any_axis_1(mask):
    return mask[:,0]+ mask[:,1]+ mask[:,2]

@jit(nopython=True)
def any_axis_1_4(mask):
    return mask[:,0]+ mask[:,1]+ mask[:,2] + mask[:,3]

@jit(nopython=True)
def all_axis_1_4(mask):
    return mask[:,0]* mask[:,1]* mask[:,2] * mask[:,3]

@jit(nopython=True)
def all_axis_1(mask):
    return mask[:,0]* mask[:,1]* mask[:,2]


@jit(nopython=True)
def pair_t1s(tri_t):
    arange = np.arange(tri_t.shape[0],dtype=np.float64)
    pairs = np.ones(tri_t.shape[0],dtype=np.int64)*-1
    quartets = np.ones((tri_t.shape[0],4),dtype=np.int64)*-1
    for i, tri in enumerate(tri_t):
        if (i==pairs).sum() == 0:
            for j in range(3):
                a,b,c = np.roll(tri,j)
                mask = any_axis_1(tri_t == a)*any_axis_1(tri_t == b)*(~any_axis_1(tri_t == c))
                if mask.sum()==1:
                    pairs[i] = int(np.dot(mask*1.0,arange))
                    tri_pair = tri_t[mask][0]
                    other_cell = tri_pair[(tri_pair!=a)*(tri_pair!=b)][0]
                    quartet = np.append(tri,other_cell)
                    quartets[i] = quartet
    return quartets[any_axis_1_4(quartets != -1)]

@jit(nopython=True)
def filter_pairs(quartets_t,quartets_tp1):
    mask = np.zeros(quartets_t.shape[0],dtype=np.bool_)
    for i, quartet in enumerate(quartets_t):
        predicted_quartet = np.array((quartet[0],quartet[3],quartet[2],quartet[1]))
        mask[i] = all_axis_1_4(predicted_quartet == quartets_tp1).any() + all_axis_1_4(np.flip(predicted_quartet) == quartets_tp1).any()
    return quartets_t[mask]

@jit(nopython=True)
def flip_axis_1(mat):
    for i, val in enumerate(mat):
        mat[i] = np.flip(val)
    return mat

@jit(nopython=True)
def check_cache(cache,cache_T,changed_t,t,ti,quartets_t):
    cache_times = changed_t[ti - cache_T:ti]
    cache_times_check = (t - cache_times < cache_T).sum()
    for i in range(cache_times_check):
        cache_i = cache[(ti - i) % cache_T]
        for j, quartet in enumerate(quartets_t):
            for cache_ii in cache_i:
                if (quartet == cache_ii).all():
                    quartets_t[j] = -1
    quartets_t = quartets_t[any_axis_1_4(quartets_t!=-1)]
    return quartets_t

@jit(nopython=True)
def get_T1_swap_frequencies(tri_save, c_types,cache_T = 20):
    """
    caches quartets for cache_T to prevent counting of flipping

    :param tri_save:
    :param c_types:
    :param cache_T:
    :return:
    """


    unchanged = np.zeros(tri_save.shape[0]-1,dtype=np.bool_)
    for i, (tri_tp1, tri_t) in enumerate(zip(tri_save[1:], tri_save[:-1])):
        unchanged[i] = (tri_tp1 ==tri_t).all()
    changed_t = np.nonzero(~unchanged)[0]

    types = np.array(((0, 0, 0, 0),
                      (0, 0, 0, 1),
                      (0, 0, 1, 0),
                      (0, 0, 1, 1),
                      (1, 0, 0, 1),
                      (0, 1, 0, 1)))
    inv_types = 1 - types

    recip_types = flip_axis_1(types)
    recip_inv_types = 1 - recip_types
    type_n = np.arange(types.shape[0])

    swap_type = np.zeros((changed_t.size, types.shape[0]))
    cache = [np.ones((1,4),dtype=np.int64)*-1 for i in range(cache_T)]
    for ti, t in enumerate(changed_t):
        same_rows = all_axis_1(tri_save[t] == tri_save[t + 1])
        tri_t = tri_save[t][~same_rows]
        tri_tp1 = tri_save[t + 1][ ~same_rows]
        quartets_t = pair_t1s(tri_t)
        quartets_tp1 = pair_t1s(tri_tp1)
        quartets_t = filter_pairs(quartets_t, quartets_tp1)


        if quartets_t.shape[0] != 0:
            quar = check_cache(cache,cache_T,changed_t,t,ti,quartets_t)
            cache[ti % cache_T] = quartets_t
            quartets_t = quar
            if quartets_t.size != 0:
                conf_type = c_types[quartets_t.ravel()].reshape(-1, 4)
                for conf_t in conf_type:
                    mask = all_axis_1_4(conf_t == types) + all_axis_1_4(conf_t == inv_types) + all_axis_1_4(
                        conf_t == recip_types) + all_axis_1_4(conf_t == recip_inv_types)
                    if mask.sum() != 0:
                        swap_type[ti][int(np.dot(mask*1.0, type_n*1.0))] += 1
            else:
                cache[ti % cache_T] = np.array((-1,-1,-1,-1)).reshape(1,4)
    return swap_type,changed_t




@jit(nopython=True)
def get_neighbours(tri):
    """
    Given a triangulation, find the neighbouring triangles of each triangle.

    By convention, the column i in the output -- neigh -- corresponds to the triangle that is opposite the cell i in that triangle.

    Can supply neigh, meaning the algorithm only fills in gaps (-1 entries)

    :param tri: Triangulation (n_v x 3) np.int32 array
    :param neigh: neighbourhood matrix to update {Optional}
    :return: (n_v x 3) np.int32 array, storing the three neighbouring triangles. Values correspond to the row numbers of tri
    """
    n_v = tri.shape[0]
    neigh = np.ones_like(tri, dtype=np.int32) * -1
    Range = np.arange(n_v)
    tri_compare = np.concatenate((tri.T, tri.T)).T.reshape((-1, 3, 2))
    for j in Range:  # range(n_v):
        tri_sample_flip = np.flip(tri[j])
        tri_i = np.concatenate((tri_sample_flip, tri_sample_flip)).reshape(3, 2)
        for k in range(3):
            if neigh[j, k] == -1:
                neighb, l = np.nonzero((tri_compare[:, :, 0] == tri_i[k, 0]) * (tri_compare[:, :, 1] == tri_i[k, 1]))
                if neighb.size != 0:
                    neighb, l = neighb[0], l[0]
                    neigh[j, k] = neighb
                    neigh[neighb, np.mod(2 - l, 3)] = j
    return neigh


@jit(nopython=True, cache=True)
def get_k2(tri, v_neighbours):
    """
    To determine whether a given neighbouring pair of triangles needs to be re-triangulated, one considers the sum of
    the pair angles of the triangles associated with the cell centroids that are **not** themselves associated with the
    adjoining edge. I.e. these are the **opposite** angles.

    Given one cell centroid/angle in a given triangulation, k2 defines the column index of the cell centroid/angle in the **opposite** triangle

    :param tri: Triangulation (n_v x 3) np.int32 array
    :param v_neighbours: Neighbourhood matrix (n_v x 3) np.int32 array
    :return:
    """
    three = np.array([0, 1, 2])
    nv = tri.shape[0]
    k2s = np.empty((nv, 3), dtype=np.int32)
    for i in range(nv):
        for k in range(3):
            neighbour = v_neighbours[i, k]
            k2 = ((v_neighbours[neighbour] == i) * three).sum()
            k2s[i, k] = k2
    return k2s

@jit(nopython=True)
def get_quart_type_from_conf_type(conf_type,types,inv_types,recip_types,recip_inv_types,type_n):
    quart_type = np.zeros(6,dtype=np.int64)
    for conf_t in conf_type:
        mask = all_axis_1_4(conf_t == types) + all_axis_1_4(conf_t == inv_types) + all_axis_1_4(
            conf_t == recip_types) + all_axis_1_4(conf_t == recip_inv_types)
        if mask.sum() != 0:
            quart_type[int(np.dot(mask * 1.0, type_n * 1.0))] += 1
    return quart_type

def get_quart_types(tri_save, c_types,changed_t):

    types = np.array(((0, 0, 0, 0),
                      (0, 0, 0, 1),
                      (0, 0, 1, 0),
                      (0, 0, 1, 1),
                      (1, 0, 0, 1),
                      (0, 1, 0, 1)))
    inv_types = 1 - types

    recip_types = flip_axis_1(types)
    recip_inv_types = 1 - recip_types
    type_n = np.arange(types.shape[0])
    quart_types = np.zeros((changed_t.size,6),dtype=np.int64)
    for ti, t in enumerate(changed_t):
        tri = tri_save[ti]
        neigh_t = get_neighbours(tri)
        k2_t = get_k2(tri, neigh_t)
        conf = c_types[tri_save[ti]]
        c_types_opp = c_types[tri[neigh_t, k2_t]]
        conf_type = np.row_stack([np.column_stack((conf,opp)) for opp in c_types_opp.T])
        quart_types[ti] = get_quart_type_from_conf_type(conf_type, types, inv_types, recip_types, recip_inv_types, type_n)
    return quart_types

if __name__ == "__main__":
    Id = int(sys.argv[1])
    N = int(sys.argv[2])
    rep = int(sys.argv[3])

    make_directory("from_unsorted_control/jamming_analysis")
    for i in range(rep):
        for run in range(2):
            vor = initialize_vor(Id,i,run)

            D,x2_mean = get_D(vor, T=15000)

            t1_swap_freq,changed_t = get_T1_swap_frequencies(vor.tri_save,vor.c_types,cache_T=20)

            quart_types = get_quart_types(vor.tri_save, vor.c_types,changed_t)

            av_rate = (t1_swap_freq/quart_types).sum(axis=0) / (vor.t_span[-1] + vor.dt)

            np.savez_compressed("from_unsorted_control/jamming_analysis/%d_%d_%d.npz" % (Id, i, run),changed_t=changed_t,t1_swap_freq=t1_swap_freq,quart_types=quart_types,av_rate=av_rate,D=D,x2_mean=x2_mean)
