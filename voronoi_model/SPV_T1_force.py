from voronoi_model.voronoi_model_periodic import *
import numpy as np
import matplotlib.pyplot as plt
import collections


"""
Introduction
------------


-- Simulate a monolayer with the same cell type
-- Induce a second cell type
-- Re-initialize the noise in the resultant quartet such that it is approximately C-E movement (--> ^ v <--) + noise 
(basically to prevent massive over-sampling to get the right movements)
-- Assay the energy 

"""

vor = Tissue()
vor.generate_cells(600)
vor.make_init(7,noise = 0.05)
p0 = 3.9
r = 5
vor.v0 = 1e-1
vor.Dr = 1e-1
beta = 0.01#0.1

vor.kappa_A = 1
vor.kappa_P = 1/r
vor.A0 = 1
vor.P0 = p0
vor.a = 0.3
vor.k = 1


vor.set_interaction(W = (2*beta*vor.P0/r)*np.array([[0, 1], [1, 0]]),pE=0.5,randomize=True)



vor.cell_movement_mask = np.zeros_like(vor.c_types,dtype=np.bool)

vor.set_t_span(0.025,200)
vor.no_noise_time = 0
vor._triangulate_periodic(vor.x)

def get_quartet():
    tri_i = int(np.random.random()*vor.tris.shape[0])
    Tri = vor.tris[tri_i]
    other_tri_mask = np.zeros(vor.tris.shape[0],dtype=np.bool)
    for i in range(3):
        mask = np.sum(vor.tris - np.roll(np.flip(Tri),i)==0,axis=1)==2
        other_tri_mask += mask
    other_tri_i = np.nonzero(other_tri_mask)[0][2] #arbitrarily pick the 3rd
    other_tri = vor.tris[other_tri_i]
    countdict = collections.Counter(np.dstack((other_tri,Tri)).ravel())

    for i, val in enumerate(countdict):
        if countdict[val] ==1:
            id = np.nonzero(Tri==val)[0]
            if id.size!=0:
                Tri = np.roll(Tri,-id)
            id = np.nonzero(other_tri==val)[0]
            if id.size!=0:
                other_tri = np.roll(other_tri,-id)
    return Tri, other_tri

def get_CE_dirs(self):
    """

    plt.scatter(vor.x[:,0],vor.x[:,1])
    plt.scatter(vor.x[tri,0],vor.x[tri,1],color="red")
    plt.scatter(vor.x[other_tri,0],vor.x[other_tri,1],color="red")
    plt.quiver(vor.x[tri[0],0],vor.x[tri[0],1],convergence[0],convergence[1])
    plt.quiver(vor.x[other_tri[0],0],vor.x[other_tri[0],1],-convergence[0],-convergence[1])
    plt.quiver(vor.x[tri[2],0],vor.x[tri[2],1],-extension[0],-extension[1])
    plt.quiver(vor.x[tri[1],0],vor.x[tri[1],1],extension[0],extension[1])
    plt.show()

    :param self:
    :return:
    """
    tri,other_tri = get_quartet()
    convergence = self.x[other_tri[0]] - self.x[tri[0]]
    convergence /= np.linalg.norm(convergence)
    convergence_angle = np.arccos(convergence[0])
    extension = self.x[tri[1]]-self.x[tri[2]]
    extension /= np.linalg.norm(extension)
    extension_angle = np.arccos(extension[0])
    return tri,other_tri,convergence,extension



vor.set_interaction(W = (2*beta*vor.P0/r)*np.array([[0, 1], [1, 0]]),pE=0.5,randomize=True,c_types=vor.c_types)


vor.simulate()

vor.cell_movement_mask[list(t1cells)] = True
vor.c_types[vor.cell_movement_mask] = vor.c_types[vor.cell_movement_mask] + 2

vor.cols = "red","blue","orange","green"
vor.plot_scatter = False
vor.animate(n_frames=30)


def get_unique_rows(A,B):
       """https://stackoverflow.com/questions/8317022/get-intersecting-rows-across-two-2d-numpy-arrays"""
       nrows, ncols = A.shape
       dtype={'names':['f{}'.format(i) for i in range(ncols)],
              'formats':ncols * [A.dtype]}
       C = np.setdiff1d(A.view(dtype), B.view(dtype))
       C = C.view(A.dtype).reshape(-1, ncols)
       return C



def get_T1s(A,B):
    C = get_unique_rows(A,B)
    #
    # #bodge -- get rid of cases where there are multiple swaps in an iteration
    # cdict = collections.Counter(C.ravel())
    # for val in cdict:
    #     if cdict[val]!=2:
    #         C = C[np.where((C!=val).all(axis=1))[0]]

    #1. Group into pairs
    Grouped = np.zeros((int(C.shape[0]/2),2,3)).astype(int)
    j = 0

    for i in range(3):
       z = np.absolute(C[:,np.newaxis] - np.roll(np.flip(C,axis=1),i,axis=1))==0
       find_shared = np.array(np.nonzero(np.nansum(z,axis=2)==2)).T
       find_shared = find_shared[find_shared[:,0]<find_shared[:,1]]
       n = find_shared.shape[0]
       Grouped[j:j+n,0],Grouped[j:j+n,1] = C[find_shared[:,0]],C[find_shared[:,1]]
       j += n

    #2. Find the corresponding pairs in B

    BGrouped = np.zeros_like(Grouped)
    for k in range(Grouped.shape[0]):
        group = Grouped[k]
        countdict = collections.Counter(group.ravel())
        countmat = np.zeros((4,2)).astype(int)
        for i, val in enumerate(countdict):
               countmat[i] = val,countdict[val]
        edge_cells = countmat[countmat[:,1]==1,0]
        for i in range(2):
               for edge_cell in edge_cells:
                     id = np.nonzero(group[i] == edge_cell)[0]
                     if id.size is not 0:
                            group[i] = np.roll(group[i],-id)
        bgroup = np.array([[group[0,0],group[0,1],group[1,0]],
                           [group[1,0],group[1,1],group[0,0]]])
        BGrouped[k] = bgroup

    return Grouped, BGrouped


dt = 1
t_span = np.arange(100,vor.tri_save.shape[0]-dt,dt).astype(int)
t1_counts = np.zeros_like(t_span)
t1_cells = []
for i, t in enumerate(t_span):
       try:
           Grouped,BGrouped=get_T1s(vor.tri_save[t], vor.tri_save[t+dt])
           t1_counts[i] = Grouped.shape[0]
           t1_cells.append(Grouped)
       except ValueError:
           t1_counts[i] = 0

pmt = 20
t_eval = np.arange(-pmt,pmt+1).astype(int)

fig, ax = plt.subplots()

for i, T1_cell in enumerate(t1_cells):
    if T1_cell.size is not 0:
        for t1_cell in T1_cell:
            t1_cell = t1_cell.reshape((2,3))
            opposites = vor.c_types[t1_cell[0,0]]==vor.c_types[t1_cell[1,0]]
            adjacents = vor.c_types[t1_cell[0,1]]==vor.c_types[t1_cell[0,2]]
            different = vor.c_types[t1_cell[0,0]]!=vor.c_types[t1_cell[0,1]]
            if opposites&adjacents&different:
                print("yes")
                kappa_A = np.zeros(vor.n_c)
                kappa_P = np.zeros(vor.n_c)
                kappa_A[np.unique(t1_cell)] = 1
                kappa_P[np.unique(t1_cell)] = 1 / r
                # kappa_A = np.ones(vor.n_c)
                # kappa_P = np.ones(vor.n_c)/r
                J = vor.J.copy()
                J[:,list(set(list(np.arange(vor.n_c))).difference(set(np.unique(t1_cell))))] = 0
                T_eval = t_eval + t_span[i]
                energies = np.zeros_like(T_eval,dtype=np.float64)
                for j, t in enumerate(T_eval):
                    vor.tris = vor.tri_save[t]
                    vor.x = vor.x_save[t]
                    vor.assign_vertices()
                    vor.Cents = vor.x[vor.tris]
                    vor.vs = vor.get_vertex_periodic()
                    n_neigh = get_neighbours(vor.tris)
                    vor.v_neighbours = n_neigh
                    vor.neighbours = vor.vs[vor.v_neighbours]
                    A = vor.get_A_periodic(vor.neighbours,vor.vs)
                    P = vor.get_P_periodic(vor.neighbours,vor.vs)
                    l_int = get_l_interface(vor.n_v, vor.n_c, vor.neighbours, vor.vs, vor.CV_matrix, vor.L)
                    energy = np.sum(kappa_A*(A-vor.A0)**2) + np.sum(kappa_P*(P-vor.P0)**2) + np.sum(l_int*J)
                    energies[j] = energy
                ax.plot(t_eval,energies)
ax.set(ylabel="Energy",xlabel="Time")
fig.show()


"""
Here what we want to do is identify T1 transitions and track the energy before and after them 


"""


#
#
# prop = 0.5
# p0eff = p0*(1-prop*vor.J.max()/(vor.P0*vor.kappa_P))
#
#
#
# shape_index = vor.P/np.sqrt(vor.A)
# fig, ax = plt.subplots()
# ax.hist(shape_index)
# fig.show()
# print(shape_index.mean())
#
# vor.get_self_self_interface(100)
# nA,nB = vor.get_num_islands(100)
#
# fig, ax = plt.subplots(1,2)
# ax[0].plot(vor.self_self_interface.mean(axis=1))
# ax[1].plot(nA)
# ax[1].plot(nB)
# fig.show()
#
#
# #
# # fig, ax = plt.subplots()
# # vor.plot_vor(vor.x_save[93],ax)
# # ax.axis("off")
# # fig.show()
#
#
# #
# def image(x,L,res=25):
#     X,Y = np.meshgrid(np.linspace(0,L,res),np.linspace(0,L,res),indexing="ij")
#     XX,YY = X.ravel(),Y.ravel()
#     d = (np.mod(np.outer(XX, np.ones_like(x[:, 0])) - np.outer(np.ones_like(XX), x[:, 0]) + L / 2, L) - L / 2) ** 2 + (np.mod(np.outer(YY, np.ones_like(x[:, 1])) - np.outer(np.ones_like(XX), x[:, 1]) + L / 2, L)- L / 2) ** 2
#     im = np.empty(XX.shape,dtype=np.int32)
#     for i, D in enumerate(d):
#         im[i] = np.argmin(D)
#     im = im.reshape(X.shape)
#     return im
#
# def type_im(x,L,c_types,res=25):
#     im = image(x,L,res=res)
#     tim = c_types[im]
#     return tim
#
# @jit(nopython=True,cache=True)
# def type_im_fast(XX,YY,res,x,L,c_types):
#     d = (np.mod(np.outer(XX, np.ones_like(x[:, 0])) - np.outer(np.ones_like(XX), x[:, 0]) + L / 2, L) - L / 2) ** 2 + (np.mod(np.outer(YY, np.ones_like(x[:, 1])) - np.outer(np.ones_like(XX), x[:, 1]) + L / 2, L)- L / 2) ** 2
#     im = np.empty(XX.shape, dtype=np.int32)
#     for i, D in enumerate(d):
#         im[i] = np.argmin(D)
#     im = im.reshape((res,res))
#     tim = c_types[im.ravel()].reshape((res,res))
#     return tim
#
# @jit(nopython=True,cache=True)
# def get_radial_profile(X,Y,res,x,L,c_types,Dround):
#     tim = type_im_fast(X.ravel(), Y.ravel(), res, x, L, c_types)
#     tim = 2 * tim - 1
#     val = np.outer(tim.ravel(), np.ones_like(tim.ravel())) * np.outer(np.ones_like(tim.ravel()), tim.ravel())
#     tbin = np.bincount(Dround.ravel(), val.ravel())
#     nr = np.bincount(Dround.ravel())
#     radialprofile = tbin / nr
#     return radialprofile
#
# @jit(nopython=True,cache=True)
# def get_radial_profile_type(X,Y,res,x,L,c_types,Dround,ctype=0):
#     """
#     Radial profile but for a specific cell type
#     :param X:
#     :param Y:
#     :param res:
#     :param x:
#     :param L:
#     :param c_types:
#     :param Dround:
#     :param ctype:
#     :return:
#     """
#     tim = type_im_fast(X.ravel(), Y.ravel(), res, x, L, c_types)
#     type_mask = (tim==ctype).ravel()
#     tim = 2 * tim - 1
#     val = np.outer(tim.ravel()[type_mask], np.ones_like(tim.ravel())) * np.outer(np.ones(type_mask.sum()), tim.ravel())
#     tbin = np.bincount(Dround[type_mask].ravel(), val.ravel())
#     nr = np.bincount(Dround[type_mask].ravel())
#     radialprofile = tbin / nr
#     return radialprofile
#
# @jit(nopython=True,cache=True)
# def get_radial_profile_type_norm(X,Y,res,x,L,c_types,Dround):
#     """
#     Radial profile, normalized by the numbers of each cell type (or rather the number of occupied pixels)
#
#     This counteracts artefact in autocorrelation where self-self > 0 as x --> infinity due to unequal cell numbers.
#     :param X:
#     :param Y:
#     :param res:
#     :param x:
#     :param L:
#     :param c_types:
#     :param Dround:
#     :return:
#     """
#     tim = type_im_fast(X.ravel(), Y.ravel(), res, x, L, c_types)
#     type_mask = (tim==0).ravel()
#     tim = 2 * tim - 1
#     val = np.outer(tim.ravel()[type_mask], np.ones_like(tim.ravel())) * np.outer(np.ones(type_mask.sum()), tim.ravel())
#     tbin = np.bincount(Dround[type_mask].ravel(), val.ravel())
#     nr = np.bincount(Dround[type_mask].ravel())
#     radialprofileA = tbin / nr
#     val = np.outer(tim.ravel()[~type_mask], np.ones_like(tim.ravel())) * np.outer(np.ones((~type_mask).sum()), tim.ravel())
#     tbin = np.bincount(Dround[~type_mask].ravel(), val.ravel())
#     nr = np.bincount(Dround[~type_mask].ravel())
#     radialprofileB = tbin / nr
#     return (radialprofileA + radialprofileB)/2
#
# def get_radial_profiles(x_save,skip,mult,L,c_types,res):
#     x_range = (np.arange(res)+0.5)/res*L
#     X,Y = np.meshgrid(x_range,x_range,indexing="ij")
#     dX = np.outer(X.ravel(), np.ones_like(X.ravel())) - np.outer(np.ones_like(X.ravel()), X.ravel())
#     dY = np.outer(Y.ravel(), np.ones_like(Y.ravel())) - np.outer(np.ones_like(Y.ravel()), Y.ravel())
#     dX, dY = np.mod(dX + L / 2, L) - L / 2, np.mod(dY + L / 2, L) - L / 2
#     D = np.sqrt(dX ** 2 + dY ** 2)
#     Dround = (D * mult).astype(int)
#     ds = np.unique(Dround.ravel()) / mult
#
#     radialprofiles = np.zeros((x_save[::skip].shape[0],np.amax(Dround)+1))
#     for i, x in enumerate(x_save[::skip]):
#         radialprofiles[i]=  get_radial_profile(X, Y, res, x, L, c_types, Dround)
#     return radialprofiles,ds
#
# rad,ds = get_radial_profiles(vor.x_save,300,5,vor.L,vor.c_types,res=70)
# plt.imshow(rad.T,extent=[-1,1,-1,1],vmax=0.1,vmin=-0.1)
# plt.show()
# def get_radial_profiles_type(x_save,skip,mult,L,c_types,res):
#     x_range = (np.arange(res)+0.5)/res*L
#     X,Y = np.meshgrid(x_range,x_range,indexing="ij")
#     dX = np.outer(X.ravel(), np.ones_like(X.ravel())) - np.outer(np.ones_like(X.ravel()), X.ravel())
#     dY = np.outer(Y.ravel(), np.ones_like(Y.ravel())) - np.outer(np.ones_like(Y.ravel()), Y.ravel())
#     dX, dY = np.mod(dX + L / 2, L) - L / 2, np.mod(dY + L / 2, L) - L / 2
#     D = np.sqrt(dX ** 2 + dY ** 2)
#     Dround = (D * mult).astype(int)
#     Dmax = np.amax(Dround) + 1
#     ds = np.arange(Dmax)/mult
#
#     radialprofiles = np.zeros((x_save[::skip].shape[0],ds.size))
#     for i, x in enumerate(x_save[::skip]):
#         radialprofiles[i]=  get_radial_profile_type_norm(X, Y, res, x, L, c_types, Dround)
#     return radialprofiles,ds
#
# rad,ds = get_radial_profiles_type(vor.x_save,100,2,vor.L,vor.c_types,res=40)
# plt.imshow(rad.T,extent=[-1,1,-1,1],vmax=0.1,vmin=-0.1)
# plt.show()
#
# """
# Generates a contour plot for the radial distribution function
# """
#
# t_sel = vor.t_span[::100]
# TT, DD = np.meshgrid(t_sel,ds)
#
# levels = np.linspace(-0.05,0.05,100)
# rad_mod =rad.copy()
# rad_mod[rad<=levels.min()] = levels.min() +1e-17
# rad_mod[rad>=levels.max()] = levels.max() - 1e-17
#
# plt.tricontourf(TT.ravel(),DD.ravel(),(rad_mod.T).ravel(),levels=levels,cmap=plt.cm.plasma)
# plt.show()
#
#
#
# from scipy.optimize import curve_fit
#
# def corr_fn(r,a,b):
#     return np.exp(-a*r)*np.cos(r*np.pi*2/b)
#
# def get_L_star(x_save,skip,mult,L,c_types,res):
#     rads,ds = get_radial_profiles_type(x_save,skip,mult,L,c_types,res)
#     L_stars = np.zeros(rads.shape[0])
#     a,b = L,L
#     for i, rad in enumerate(rads):
#         mask = ~np.isnan(rad)
#         a,b = curve_fit(corr_fn, ds[mask],rad[mask],(a,b),bounds=(np.array([0,0]),np.array([np.inf,np.sqrt(2)*L])))[0]
#         L_stars[i] = b
#     return rads,L_stars
#
#
#
# for res in [20,30,40,50]:
#     rads,L_star = get_L_star(vor.x_save,100,20,vor.L,vor.c_types,res=res)
#     plt.plot(L_star)
# plt.show()
#
#
# rads,L_star = get_L_star(vor.x_save,50,10,vor.L,vor.c_types,res=30)
#
# fig, ax = plt.subplots(figsize=(4,4))
# ax.plot(L_star)
# ax.set(xlabel="Correlation lengthscale",ylabel="Time")
# fig.show()
#
# #
# # vor.animate(n_frames=50)
# #
# #
# # vor.get_self_self()
# # fig, ax = plt.subplots()
# # ax.plot(vor.self_self)
# # ax.set(xlabel="Time",ylabel="Fraction of self-self interactions")
# # fig.savefig("self_self.pdf")
# #
# #
# # ratio = 0.5
# # P0_eff = alpha*ratio/vor.kappa_P + vor.P0
# # p0_eff = P0_eff/np.sqrt(vor.A0)
# # print(p0_eff)
# #
# # """
# # Stat to measure q for each cell.
# # And compare with neighbourhood and thus p0_eff
# # And MSD (short time-scale)
# # """