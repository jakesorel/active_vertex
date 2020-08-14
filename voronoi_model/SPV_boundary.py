from voronoi_model.voronoi_model_periodic import *
import numpy as np
import matplotlib.pyplot as plt


vor = Tissue()
vor.generate_cells(300)
vor.make_init_boundary(20,0.15)
alpha = 0.05
vor.b_tension = 0.16
vor.set_interaction_boundary(W = alpha*np.array([[0, 1], [1, 0]]),pE=0.5)
vor._triangulate(vor.x0)
vor.assign_vertices()
self = vor


self.kappa_B = 0.3
self.l_b0 = 0.1
self.zeta = 0.01

# vor.P0 = 3.00
p0 = 4.2 #3.81
vor.A0 = 0.86
vor.P0 = p0*np.sqrt(vor.A0)
print(vor.P0)

vor.v0 = 1e-1
vor.Dr = 1
vor.kappa_A = 0.2
vor.kappa_P = 0.05
vor.a = 0.3
vor.k = 2

#
# ix,iy,jx,jy,kx,ky = 1,1,0,0,0,2
#
# dtheta_dr2(ix,iy,jx,jy,kx,ky)
#

#
# self.x0 = self.check_boundary(self.x0)
# self.x = self.x0
# #
n_C = self.n_C
# fig, ax = plt.subplots()
# ax.scatter(self.x0[:n_C,0],self.x0[:n_C,1],color="red")
# ax.scatter(self.x0[n_C:,0],self.x0[n_C:,1],color="blue")
#
# fig.show()
#
# b_cells = self.x0[n_C:]
# r = np.linalg.norm(b_cells - self.L/2,axis=1)
# self.x0[n_C:] = r.mean()*((b_cells-self.L/2).T/r).T + self.L/2
#
# th_true = np.arctan2(((b_cells-self.L/2).T/r)[1],((b_cells-self.L/2).T/r)[0])
# th_true = np.mod(th_true,np.pi*2)
# theta = np.linspace(0,np.pi*2,b_cells.shape[0]+1)[:-1]
# theta = theta[np.argsort(th_true)]
#
# self.x0[n_C:] = r.mean()*np.dstack((np.cos(theta),np.sin(theta)))+ self.L/2
#
# x = self.x0
# self.assign_vertices()
# self.Cents = np.array([(self.CV_matrix.T * x[:, 0]).sum(axis=-1), (self.CV_matrix.T * x[:, 1]).sum(axis=-1)]).T
# self.Cents = self.x0[self.tris]
#
# self.get_A_periodic(self.neighbours,self.vs)
# Cents = self.Cents
# n_C = self.n_C
# n_c = self.n_c
# tris = self.tris
# CV_matrix = self.CV_matrix
#
# Cents,a,k, CV_matrix,n_c,L,n_C,tris,zeta = self.Cents,self.a,self.k, self.CV_matrix,self.n_c,self.L,self.n_C,self.tris,1
#
#
F_bend = get_F_bend(self.n_c,self.CV_matrix,self.n_C,self.x0,1)
fig, ax = plt.subplots()
ax.scatter(self.x0[:n_C,0],self.x0[:n_C,1],color="red")
ax.scatter(self.x0[n_C:,0],self.x0[n_C:,1],color="blue")
ax.quiver(self.x0[:,0],self.x0[:,1],F_bend[:,0],F_bend[:,1],color="black")
print((F_bend**2).mean())
ax.set(aspect=1,xlim=(5,15),ylim=(5,15))
fig.show()
#
vor.cols = "red","blue","white"
vor.plot_scatter = False
#
vor.set_t_span(0.025,400)
vor.simulate_boundary(b_extra=10,print_every=2000)

#
# x = vor.x0
# tri = vor.tris
# V = vor.vs
# fig, ax = plt.subplots()
# ax.scatter(x[self.n_C:,0],x[self.n_C:,1],color="red")
# ax.scatter(x[:self.n_C,0],x[:self.n_C,1],color="blue")
#
# for i in range(self.n_c):
#     ax.text(x[i,0],x[i,1],i,fontsize=10)
# for TRI in tri:
#     for i in range(3):
#         a,b = TRI[i],TRI[np.mod(i+1,3)]
#         if (a>=0)and(b>=0):
#             X = np.stack((x[a],x[b])).T
#             ax.plot(X[0],X[1],color="black")
# V = self.vs
# ax.scatter(V[:,0],V[:,1])
# # ax.set(xlim=(20,30),ylim=(20,30))
# fig.show()
#
# for i in range(self.n_C,self.n_c-1):
#     print(np.linalg.norm((x[i]-x[i+1])))
#
# #
# plt.close("all")
# plt.plot(vor.x_save[:,self.n_C,0],vor.x_save[:,self.n_C,1])
# print(vor.x_save[:,self.n_C,0],vor.x_save[:,self.n_C,1])
# plt.show()


vor.animate_boundary(n_frames=50)
#
# fig, ax = plt.subplots()
# vor.plot_vor(vor.x,ax)
# ax.axis("off")
# fig.savefig("voronoi.pdf")
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
#
# from scipy.spatial.distance import pdist, squareform
#
# @jit(nopython=True,cache=True)
# def get_radials(res,L):
#     r = np.linspace(L/res,L,res)
#     dx,dy = [],[]
#     th_lengths = []
#     for R in r:
#         theta = np.linspace(0,2*np.pi,int(res*np.sin(R/L)))
#         for th in theta:
#             dx.append(R*np.sin(th))
#             dy.append(R*np.cos(th))
#         th_lengths.append(int(res*np.sin(R/L)))
#     return np.array(dx),np.array(dy),np.array(th_lengths),r
#
# @jit(nopython=True,cache=True)
# def spatial_autocorr(x,L,c_types,res=25):
#     dx,dy,th_lengths,r = get_radials(res,L)
#     r = r[th_lengths!=0]
#     th_lengths = th_lengths[th_lengths!=0]
#     dxy = np.empty((dx.size,2))
#     dxy[:,0] = dx
#     dxy[:,1] = dy
#     C_types = c_types*2 - 1
#     gr = np.zeros_like(r)
#     for j, xx in enumerate(x):
#         radX = xx + dxy
#         d = (np.mod(np.outer(radX[:,0], np.ones_like(x[:, 0])) - np.outer(np.ones_like(radX[:,0]), x[:, 0])+L/2,L)-L/2) ** 2 + \
#                 (np.mod(np.outer(radX[:,1], np.ones_like(x[:, 1])) - np.outer(np.ones_like(radX[:,1]), x[:, 1]) + L/2,L)-L/2) ** 2
#         id = np.empty(radX.shape[0], dtype=np.int32)
#         for i, D in enumerate(d):
#             id[i] = C_types[np.argmin(D)]
#         i = 0
#         for k, th_length in enumerate(th_lengths):
#             id_sample = id[i:th_length+i]
#             id_cell = C_types[j]
#             gr[k] += (id_sample*id_cell).mean()
#             i += th_length
#     gr = gr/C_types.size
#     return gr,r
#
# from scipy.optimize import curve_fit
#
# def corr_fn(r,a,b):
#     return np.exp(-a*r)*np.cos(r*np.pi*2/b)
#
# def get_L_star(x,L,c_types,res=25):
#     gr, r = spatial_autocorr(x, L, c_types, res)
#     L_star = curve_fit(corr_fn, r, gr)[0][1]
#     return L_star
#
# grs = np.array([spatial_autocorr(x, vor.L, vor.c_types, 60)[0] for x in vor.x_save[::10]])
#
# fig, ax = plt.subplots()
# ax.imshow(np.flip(grs.T,axis=0),extent=[-1,1,-1,1],cmap=plt.cm.plasma,vmax=0.2,vmin=-0.2)
# ax.set(xlabel="Time",ylabel="$r$")
# fig.savefig("g(r,t).pdf")
#
# L_stars = np.array([get_L_star(x,vor.L,vor.c_types,res=100) for x in vor.x_save[0:500:20]])
#
# fig, ax = plt.subplots(figsize=(4,4))
# ax.plot(vor.t_span[0:500:20],L_stars,color="black")
# ax.set(xlabel="Time",ylabel=r"$L_*$"" (Correlation lengthscale)")
# fig.savefig("L_star.pdf")
#
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
rji = np.array([[ -0.1652488 ,  -0.77930148],
       [  0.80199677,   0.84170409],
       [ -1.20215099,  -0.11986675],
       [ -0.78467641,  -0.03029031],
       [  0.65816083,   0.57671823],
       [  0.8256558 ,   0.34880981],
       [ -0.86231052,   1.24426919],
       [ -0.69645959,  -0.05121206],
       [ -1.04956277,  -0.24048396],
       [  1.2154425 ,   0.35939363],
       [  0.76317095,   0.14888858],
       [  0.4980634 ,  -0.21843338],
       [ -0.52599564,   0.17570595],
       [ -0.81439751,  -0.86485798],
       [  1.07210876,  -0.29802834],
       [  0.74477983,  -0.26131581],
       [ -0.69274979,   1.3330738 ],
       [ -0.46915666,  -0.91234694],
       [ -0.07356453,   1.35589867],
       [  0.04944494,  -0.91708183],
       [  0.17616799,   0.73628519],
       [  0.47968692,  -0.64555787],
       [  0.30039691,   0.77224648],
       [ -0.16923986,  -0.78184834],
       [-14.17339562, -10.32441017],
       [  0.48834119,  -1.16055929],
       [ -0.56790373,  -0.61180928]])
rki = np.array([[  0.16923986,   0.78184834],
       [ -0.30039691,  -0.77224648],
       [  0.69645959,   0.05121206],
       [  1.04956277,   0.24048396],
       [ -0.80199677,  -0.84170409],
       [ -0.4980634 ,   0.21843338],
       [  1.20215099,   0.11986675],
       [  0.78467641,   0.03029031],
       [  0.81439751,   0.86485798],
       [ -0.65816083,  -0.57671823],
       [ -1.07210876,   0.29802834],
       [ -0.74477983,   0.26131581],
       [  0.86231052,  -1.24426919],
       [  0.46915666,   0.91234694],
       [ -1.2154425 ,  -0.35939363],
       [ -0.76317095,  -0.14888858],
       [  0.52599564,  -0.17570595],
       [ -0.04944494,   0.91708183],
       [  0.69274979,  -1.3330738 ],
       [  0.56790373,   0.61180928],
       [  0.07356453,  -1.35589867],
       [  0.1652488 ,   0.77930148],
       [ -0.17616799,  -0.73628519],
       [ -0.48834119,   1.16055929],
       [-14.17339562, -10.32441017],
       [ -0.8256558 ,  -0.34880981],
       [ -0.47968692,   0.64555787]])