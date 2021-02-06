from voronoi_model_periodic import *
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import dask
from dask.distributed import Client
import sys



@jit(nopython=True,cache=True)
def type_im_fast(XX,YY,res,x,L,c_types):
    d = (np.mod(np.outer(XX, np.ones_like(x[:, 0])) - np.outer(np.ones_like(XX), x[:, 0]) + L / 2, L) - L / 2) ** 2 + (np.mod(np.outer(YY, np.ones_like(x[:, 1])) - np.outer(np.ones_like(XX), x[:, 1]) + L / 2, L)- L / 2) ** 2
    im = np.empty(XX.shape, dtype=np.int32)
    for i, D in enumerate(d):
        im[i] = np.argmin(D)
    im = im.reshape((res,res))
    tim = c_types[im.ravel()].reshape((res,res))
    return tim

@jit(nopython=True,cache=True)
def get_radial_profile_type_norm(X,Y,res,x,L,c_types,Dround):
    """
    Radial profile, normalized by the numbers of each cell type (or rather the number of occupied pixels)

    This counteracts artefact in autocorrelation where self-self > 0 as x --> infinity due to unequal cell numbers.
    :param X:
    :param Y:
    :param res:
    :param x:
    :param L:
    :param c_types:
    :param Dround:
    :return:
    """
    tim = type_im_fast(X.ravel(), Y.ravel(), res, x, L, c_types)
    type_mask = (tim==0).ravel()
    tim = 2 * tim - 1
    val = np.outer(tim.ravel()[type_mask], np.ones_like(tim.ravel())) * np.outer(np.ones(type_mask.sum()), tim.ravel())
    tbin = np.bincount(Dround[type_mask].ravel(), val.ravel())
    nr = np.bincount(Dround[type_mask].ravel())
    radialprofileA = tbin / nr
    val = np.outer(tim.ravel()[~type_mask], np.ones_like(tim.ravel())) * np.outer(np.ones((~type_mask).sum()), tim.ravel())
    tbin = np.bincount(Dround[~type_mask].ravel(), val.ravel())
    nr = np.bincount(Dround[~type_mask].ravel())
    radialprofileB = tbin / nr
    return (radialprofileA + radialprofileB)/2

def get_radial_profiles_type(x_save,skip,mult,L,c_types,res):
    x_range = (np.arange(res)+0.5)/res*L
    X,Y = np.meshgrid(x_range,x_range,indexing="ij")
    dX = np.outer(X.ravel(), np.ones_like(X.ravel())) - np.outer(np.ones_like(X.ravel()), X.ravel())
    dY = np.outer(Y.ravel(), np.ones_like(Y.ravel())) - np.outer(np.ones_like(Y.ravel()), Y.ravel())
    dX, dY = np.mod(dX + L / 2, L) - L / 2, np.mod(dY + L / 2, L) - L / 2
    D = np.sqrt(dX ** 2 + dY ** 2)
    Dround = (D * mult).astype(int)
    Dmax = np.amax(Dround) + 1
    ds = np.arange(Dmax)/mult

    radialprofiles = np.zeros((x_save[::skip].shape[0],ds.size))
    for i, x in enumerate(x_save[::skip]):
        radialprofiles[i]=  get_radial_profile_type_norm(X, Y, res, x, L, c_types, Dround)
    return radialprofiles,ds


def corr_fn(r,a,b):
    return np.exp(-a*r)*np.cos(r*np.pi*2/b)

def get_L_star(x_save,skip,mult,L,c_types,res):
    rads,ds = get_radial_profiles_type(x_save,skip,mult,L,c_types,res)
    L_stars = np.zeros(rads.shape[0])
    a,b = L,L
    for i, rad in enumerate(rads):
        mask = ~np.isnan(rad)
        a,b = curve_fit(corr_fn, ds[mask],rad[mask],(a,b),bounds=(np.array([0,0]),np.array([np.inf,np.sqrt(2)*L])))[0]
        L_stars[i] = b
    return rads,L_stars



if __name__ == "__main__":

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
    vor.P0 = 3.9
    vor.a = 0.3
    vor.k = 0

    vor.set_interaction(W=(2 * beta * vor.P0 / r) * np.array([[0, 1], [1, 0]]), pE=0.5)

    vor.set_t_span(0.05,2000)
    # vor.n_t = vor.t_span.size

    def do_analysis(Id):
        for i in range(int(sys.argv[3])):
            try:
                tri_save = np.load("from_unsorted/tri_save/%d_%d.npz" % (Id,i))["arr_0"]
                tri_save = tri_save.reshape(tri_save.shape[0], -1, 3)
                x_save = np.load("from_unsorted/x_save/%d_%d.npz" % (Id,i))["arr_0"]
                x_save = x_save.reshape(x_save.shape[0], -1, 2)
                c_types = np.load("from_unsorted/c_types/%d_%d.npz" % (Id,i))["arr_0"]

                vor.n_t = tri_save.shape[0]
                vor.n_c = x_save.shape[1]
                vor.n_v = tri_save.shape[1]

                vor.c_types = c_types
                vor.x_save = x_save
                vor.tri_save = tri_save

                vor.get_self_self_interface(100)
                nA, nB = vor.get_num_islands(100)
                n_islands = np.array([nA, nB])
                n_bound = vor.get_num_boundaries(100)
                mean_self = vor.self_self_interface.mean(axis=1)

                skip = int(vor.n_t / 100)
                rads, L_star = get_L_star(vor.x_save, skip, 10, vor.L, vor.c_types, res=40)

                np.savez_compressed("from_unsorted/analysis/%d_%d.npz" % (Id,i), n_islands=n_islands, n_bound=n_bound, L_star=L_star,
                                    mean_self=mean_self)
            except FileNotFoundError:
                print("False")


    do_analysis(int(sys.argv[1]))