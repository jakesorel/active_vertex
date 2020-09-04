from voronoi_model.voronoi_model_periodic import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import curve_fit


vor = Tissue()
vor.generate_cells(400)
vor.make_init(9)
alpha = 0.02
vor.set_interaction(W = alpha*np.array([[0, 1], [1, 0]]),pE=0.4)

# vor.P0 = 3.00
p0 = 3.9 #3.81
vor.A0 = 0.86
vor.P0 = p0*np.sqrt(vor.A0)
print(vor.P0)

vor.v0 = 1e-2
vor.Dr = 0.005
vor.kappa_A = 0.2
vor.kappa_P = 0.1
vor.a = 0.3
vor.k = 0.5



vor.set_t_span(0.025,200)
vor.simulate()
vor.plot_scatter = False

fig, ax = plt.subplots()
vor.plot_vor(vor.x,ax)
ax.axis("off")
fig.show()

vor.animate(n_frames=50)




"""Generate an image from the voronoi"""

def image(x,L,res=25):
    X,Y = np.meshgrid(np.linspace(0,L,res),np.linspace(0,L,res),indexing="ij")
    XX,YY = X.ravel(),Y.ravel()
    d = (np.outer(XX,np.ones_like(x[:,0])) - np.outer(np.ones_like(XX),x[:,0]))**2 + (np.outer(YY,np.ones_like(x[:,1])) - np.outer(np.ones_like(XX),x[:,1]))**2
    im = np.empty(XX.shape,dtype=np.int32)
    for i, D in enumerate(d):
        im[i] = np.argmin(D)
    im = im.reshape(X.shape)
    return im

def type_im(x,L,c_types,res=25):
    im = image(x,L,res=res)
    tim = c_types[im]
    return tim

"""Calculate spatial autocorrelation"""

@jit(nopython=True,cache=True)
def get_radials(res,L):
    """
    Defines a set of {dx,dy} vectors to evaluate autocorrelation from each cell centroid.

    If res is the resolution (int), radius is sampled {L/res,L} in a number **res** blocks

    Then for each radius r, sample angles {0,2pi} in res*sin(r/L) blocks (i.e. approx. evenly sample in euclidian space)

    :param res:
    :param L:
    :return:
    """
    r = np.linspace(L/res,L,res)
    dx,dy = [],[]
    th_lengths = []
    for R in r:
        theta = np.linspace(0,2*np.pi,int(res*np.sin(R/L)))
        for th in theta:
            dx.append(R*np.sin(th))
            dy.append(R*np.cos(th))
        th_lengths.append(int(res*np.sin(R/L)))
    return np.array(dx),np.array(dy),np.array(th_lengths),r

@jit(nopython=True,cache=True)
def spatial_autocorr(x,L,c_types,res=25):
    """
    Calculates the spatial autocorrelation.

    Very slow and needs to be optimized.

    :param x:
    :param L:
    :param c_types:
    :param res:
    :return:
    """
    dx,dy,th_lengths,r = get_radials(res,L)
    r = r[th_lengths!=0]
    th_lengths = th_lengths[th_lengths!=0]
    dxy = np.empty((dx.size,2))
    dxy[:,0] = dx
    dxy[:,1] = dy
    C_types = c_types*2 - 1
    gr = np.zeros_like(r)
    for j, xx in enumerate(x):
        radX = xx + dxy
        d = (np.mod(np.outer(radX[:,0], np.ones_like(x[:, 0])) - np.outer(np.ones_like(radX[:,0]), x[:, 0])+L/2,L)-L/2) ** 2 + \
                (np.mod(np.outer(radX[:,1], np.ones_like(x[:, 1])) - np.outer(np.ones_like(radX[:,1]), x[:, 1]) + L/2,L)-L/2) ** 2
        id = np.empty(radX.shape[0], dtype=np.int32)
        for i, D in enumerate(d):
            id[i] = C_types[np.argmin(D)]
        i = 0
        for k, th_length in enumerate(th_lengths):
            id_sample = id[i:th_length+i]
            id_cell = C_types[j]
            gr[k] += (id_sample*id_cell).mean()
            i += th_length
    gr = gr/C_types.size
    return gr,r


def corr_fn(r,a,b):
    return np.exp(-a*r)*np.cos(r*np.pi*2/b)

def get_L_star(x,L,c_types,res=25):
    gr, r = spatial_autocorr(x, L, c_types, res)
    L_star = curve_fit(corr_fn, r, gr)[0][1]
    return L_star

grs = np.array([spatial_autocorr(x, vor.L, vor.c_types, 10)[0] for x in vor.x_save[::100]])

fig, ax = plt.subplots()
ax.imshow(np.flip(grs.T,axis=0),extent=[-1,1,-1,1],cmap=plt.cm.plasma,vmax=0.2,vmin=-0.2)
ax.set(xlabel="Time",ylabel="$r$")
fig.savefig("g(r,t).pdf")
#
L_stars = np.array([get_L_star(x,vor.L,vor.c_types,res=100) for x in vor.x_save[0:500:20]])

fig, ax = plt.subplots(figsize=(4,4))
ax.plot(vor.t_span[0:500:20],L_stars,color="black")
ax.set(xlabel="Time",ylabel=r"$L_*$"" (Correlation lengthscale)")
fig.savefig("L_star.pdf")


nA,nB = vor.get_num_islands(300)
plt.plot(nA+nB)
plt.show()

"""
Topology
--------

Idea: if completely sorted, then can distinguish two topologies. 
(1) A single heterotypic boundary
(2) Two heterotypic boundaries: where each cell type aggregate connects to itself


Approach: find heterotypic vertices. Then sample connectivity

"""



def get_num_boundaries(self, nT=100):
    """
    Get the number of islands, defined as contiguous regions of a given cell type.

    Considers only cases where there are two cell types.

    A_islands is the number of islands of cell_type == 0, and B_islands for cell_type == 1



    :param nT: Number of time-steps to consider (evenly spaced)
    :return: A_islands, B_islands
    """
    t_sel = np.linspace(0, self.n_t - 1, nT).astype(int)
    num_boundaries = np.zeros(nT, dtype=np.int32)
    for i, t in enumerate(t_sel):
        tri = self.tri_save[t]
        tri_types = self.c_types[tri]

        #Find heterotypic edges by comparing each cell with its CW neighbour in the triangulation
        het_neighbours = tri_types != np.roll(tri_types,1,axis=1)
        v_neighbours = get_neighbours(tri)

        #For each triangle (i.e. vertex), find the corresponding neighbouring vertex that makes up the above heterotypic edges
        het_v_neighbour_mask = np.roll(het_neighbours,1,axis=1)
        het_v_neighbours = v_neighbours[het_v_neighbour_mask]

        #Get the other vertex that makes up the edge (i.e. that of the triangle in question)
        het_v = np.repeat(np.arange(self.n_v),3).reshape((self.n_v,3))[het_v_neighbour_mask]

        #Build an adjacency matrix of vertex pairs (i.e. edges) that are members of a boundary
        Adj = np.zeros((self.n_v, self.n_v))
        Adj[het_v,het_v_neighbours] = 1

        #Reduce the adjacency matrix to consider only vertices that are involved in a boundary
        v_mask = (Adj.T@np.ones(self.n_v))!=0

        #Count the number of boundaries
        num_boundaries[i] =connected_components(csgraph=csr_matrix(Adj[v_mask][:,v_mask]), directed=False)[0]

    return num_boundaries

n_b = get_num_boundaries(vor,300)

"""
Stress calculations
"""

def corr_fn2(r,a,b,c):
    return c*np.exp(-a*r)*np.cos(r*np.pi*2/b)

from scipy.interpolate import Rbf


def get_div(x_save,t,N,L):
    x = vor.x_save[t-1]
    F = (np.mod(x_save[t] - x_save[t-1] + L/2,L)-L/2)/vor.dt

    Fx = Rbf(x[:,0],x[:,1],F[:,0])
    Fy = Rbf(x[:,0],x[:,1],F[:,1])

    X,Y = np.mgrid[0:L:L/N,0:L:L/N] + L/(2*N)

    FX = Fx(X,Y)
    FY = Fy(X,Y)


    div = FX + FY
    return div
div = get_div(vor.x_save,30,100,vor.L)
plt.imshow(div)
plt.show()

@jit(nopython=True,cache=True)
def stress_corr(X,Y,div,L):
    XX, YY = X.ravel(), Y.ravel()
    XXX, YYY = np.outer(np.ones_like(XX), XX), np.outer(np.ones_like(YY), YY)
    dX, dY = np.mod(XXX - XXX.T + L / 2,L) - L / 2, np.mod(YYY - YYY.T + L / 2, L) - L / 2
    D = np.sqrt(dX ** 2 + dY ** 2)

    DIV = np.outer(np.ones_like(div), div)
    dDIV = DIV * DIV.T

    nbin = 20
    dbin = np.linspace(0, D.max(), nbin)
    corr = np.zeros(nbin - 1)
    for i in range(nbin - 1):
        mask = ((D <= dbin[1 + i]) * (D > dbin[i]))*1.0
        if mask.sum()!=0:
            corr[i] = np.sum(dDIV*mask)/mask.sum()

    #
    # fig, ax = plt.subplots(1,2)
    # ax[0].imshow(div)
    # ax[1].plot(corr)
    # fig.show()

    bin_mids = (dbin[1:] + dbin[:-1])/2
    corr,bin_mids = corr[~np.isnan(corr)],bin_mids[~np.isnan(corr)]
    return corr,bin_mids

def fit_corr_stress(corr,bin_mids):
    a,b,c = curve_fit(corr_fn2,bin_mids,corr,bounds=((0,0,0),(np.inf,vor.L,np.inf)))[0]
    return a,b,c

def wrapper(t):
    L = vor.L
    N = 25
    X,Y = np.mgrid[0:L:L/N,0:L:L/N] + L/(2*N)
    divergence = get_div(vor.x_save,t,N,L)
    corr,bin_mids = stress_corr(X,Y,divergence,L)
    a,b,c = fit_corr_stress(corr,bin_mids)
    return b

n_steps = 100
div_corr_L = np.zeros(n_steps)
for i, t in enumerate(np.linspace(0,vor.x_save.shape[0]-1,100).astype(int)):
    div_corr_L[i] = wrapper(t)


Corrs, Bin_Mids = [],[]

for i, t in enumerate(np.linspace(0,vor.x_save.shape[0]-1,100).astype(int)):
    L = vor.L
    N = 25
    X,Y = np.mgrid[0:L:L/N,0:L:L/N] + L/(2*N)
    divergence = get_div(vor.x_save,t,N,L)
    corr,bin_mids = stress_corr(X,Y,divergence,L)
    Corrs.append(corr)
    Bin_Mids.append(bin_mids)






from scipy.fft import fft2,fftshift,ifft2





fdiv = fft2(div)
fdiv*np.conjugate(fdiv)

ans = ifft2(fdiv*np.conjugate(fdiv))


# vor.animate(n_frames=50)
#
#
# vor.get_self_self()
# fig, ax = plt.subplots()
# ax.plot(vor.self_self)
# ax.set(xlabel="Time",ylabel="Fraction of self-self interactions")
# fig.savefig("self_self.pdf")
#
#
# ratio = 0.5
# P0_eff = alpha*ratio/vor.kappa_P + vor.P0
# p0_eff = P0_eff/np.sqrt(vor.A0)
# print(p0_eff)
#
# """
# Stat to measure q for each cell.
# And compare with neighbourhood and thus p0_eff
# And MSD (short time-scale)
# """