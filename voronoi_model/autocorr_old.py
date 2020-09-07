


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

grs = np.array([spatial_autocorr(x, vor.L, vor.c_types, 100)[0] for x in vor.x_save[::500]])

fig, ax = plt.subplots()
ax.imshow(np.flip(grs.T,axis=0),extent=[-1,1,-1,1],cmap=plt.cm.plasma,vmax=0.2,vmin=-0.2)
ax.set(xlabel="Time",ylabel="$r$")
fig.savefig("g(r,t).pdf")
#
L_stars = np.array([get_L_star(x,vor.L,vor.c_types,res=100) for x in vor.x_save[0::500]])

fig, ax = plt.subplots(figsize=(4,4))
ax.plot(L_stars,color="black")
ax.set(xlabel="Time",ylabel=r"$L_*$"" (Correlation lengthscale)")
fig.savefig("L_star.pdf")
