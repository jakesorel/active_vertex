from voronoi_model.voronoi_model_periodic import *
import numpy as np
import matplotlib.pyplot as plt


vor = Tissue()
vor.generate_cells(400)
vor.make_init(20,noise = 0)
alpha = 0.04
vor.set_interaction(W = alpha*np.array([[0, 1], [1, 0]]),pE=0)
# vor.P0 = 3.00
p0 = 3.9 #3.81
vor.A0 = vor.L**2/vor.n_c
vor.P0 = p0*np.sqrt(vor.A0)


print(vor.P0)

vor.v0 = 5e-2
print("v0",vor.v0)
vor.Dr = 0.1
vor.kappa_A = 0.2
vor.kappa_P = 0.1
vor.a = 0.3
vor.k = 1

prop = 0.5
p0eff = p0*(1-prop*vor.J.max()/(vor.P0*vor.kappa_P))


vor.set_t_span(0.025,250)

vor.simulate()
vor.plot_scatter = False

vor.animate(n_frames=50)

vor.profile_function(vor.simulate)


fig, ax = plt.subplots()
vor.plot_vor(vor.x_save[93],ax)
ax.axis("off")
fig.show()


#
# def image(x,L,res=25):
#     X,Y = np.meshgrid(np.linspace(0,L,res),np.linspace(0,L,res),indexing="ij")
#     XX,YY = X.ravel(),Y.ravel()
#     d = (np.outer(XX,np.ones_like(x[:,0])) - np.outer(np.ones_like(XX),x[:,0]))**2 + (np.outer(YY,np.ones_like(x[:,1])) - np.outer(np.ones_like(XX),x[:,1]))**2
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

from scipy.spatial.distance import pdist, squareform

@jit(nopython=True,cache=True)
def get_radials(res,L):
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

from scipy.optimize import curve_fit

def corr_fn(r,a,b):
    return np.exp(-a*r)*np.cos(r*np.pi*2/b)

def get_L_star(x,L,c_types,res=25):
    gr, r = spatial_autocorr(x, L, c_types, res)
    L_star = curve_fit(corr_fn, r, gr)[0][1]
    return L_star

grs = np.array([spatial_autocorr(x, vor.L, vor.c_types, 60)[0] for x in vor.x_save[::100]])

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

#
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