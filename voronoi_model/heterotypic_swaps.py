import numpy as np
from numba import jit
import sys


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
    neigh = np.ones_like(tri,dtype=np.int32)*-1
    Range = np.arange(n_v)
    tri_compare = np.concatenate((tri.T, tri.T)).T.reshape((-1, 3, 2))
    for j in Range:#range(n_v):
        tri_sample_flip = np.flip(tri[j])
        tri_i = np.concatenate((tri_sample_flip,tri_sample_flip)).reshape(3,2)
        for k in range(3):
            if neigh[j,k]==-1:
                neighb,l = np.nonzero((tri_compare[:,:,0]==tri_i[k,0])*(tri_compare[:,:,1]==tri_i[k,1]))
                if neighb.size !=0:
                    neighb,l = neighb[0],l[0]
                    neigh[j,k] = neighb
                    neigh[neighb,np.mod(2-l,3)] = j
    return neigh


@jit(nopython=True,cache=True)
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


def get_n_het_swap(tri_save,c_types):
    ##1. filter no change
    unchanged = ((tri_save[1:] - tri_save[:-1]) == 0).all(axis=(1, 2))
    changed_t = np.nonzero(~unchanged)[0]

    # 2. select for the triangles that have changed for a given time t
    n_het_swap = np.zeros_like(changed_t)
    for ti, t in enumerate(changed_t):
        same_rows = (tri_save[t] == tri_save[t + 1]).all(axis=1)
        tri_t = tri_save[t, ~same_rows]

        neigh_t = get_neighbours(tri_t)
        single_swap = (neigh_t == -1).sum(axis=1)==2
        if not single_swap.all():
            tri_t = tri_t[single_swap]
            neigh_t = get_neighbours(tri_t)
        k2_t = get_k2(tri_t,neigh_t)
        tri_i, i = np.nonzero(neigh_t != -1)
        c_types_t = c_types[tri_t]
        c_types_ti = c_types_t[tri_i, i]
        c_types_tip1 = c_types_t[tri_i, np.mod(i + 1, 3)]
        c_types_tim1 = c_types_t[tri_i, np.mod(i - 1, 3)]
        c_types_opp = c_types[tri_t[neigh_t[tri_i,i],k2_t[tri_i,i]]]
        het_swap = (c_types_ti != c_types_tim1) * (c_types_ti != c_types_tip1)*(c_types_ti == c_types_opp)
        n_het_swap[ti] = het_swap.sum()/2
    n_het_swap_tot = n_het_swap.sum()
    return changed_t,n_het_swap,n_het_swap_tot

if __name__ == "__main__":

    def do_analysis(Id):
        for i in range(int(sys.argv[3])):
            for run in range(2):
                try:
                    tri_save = np.load("from_unsorted/tri_save/%d_%d_%d.npz" % (Id,i,run))["arr_0"]
                    tri_save = tri_save.reshape(tri_save.shape[0], -1, 3)
                    c_types = np.load("from_unsorted/c_types/%d_%d_%d.npz" % (Id,i,run))["arr_0"]
                    changed_t,n_het_swap,n_het_swap_tot = get_n_het_swap(tri_save,c_types)
                    np.savez_compressed("from_unsorted/het_swaps/%d_%d.npz" % (Id,i + run*int(sys.argv[3])),changed_t=changed_t, n_het_swap=n_het_swap,n_het_swap_tot = n_het_swap_tot)
                except FileNotFoundError:
                    print("False")
    do_analysis(int(sys.argv[1]))




