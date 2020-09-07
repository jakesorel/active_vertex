import numpy as np
import matplotlib.pyplot as plt
N = 8

p0_range = np.linspace(3.5, 4, N)
v0_range = np.linspace(5e-3, 1e-1, N)
beta_range = np.linspace(0, 0.3)
PP, VV, BB = np.meshgrid(p0_range, v0_range, beta_range, indexing="ij")

# PP,VV,BB = PP[:,:,:8],VV[:,:,:8],BB[:,:,:8]

def get_L_star(Id):
    FILE = np.load("analysis/%d.npz" % Id)
    return FILE["L_star"]

L_stars = np.array([get_L_star(Id) for Id in range(N**3)])

v0 = v0 = v0_range[0]

mask = VV==v0
Id_sel = np.arange(N**3)[mask.ravel()]

plt.tricontourf(PP.ravel()[Id_sel],BB.ravel()[Id_sel],L_stars[Id_sel,-1])
plt.show()

VVs,BBs = np.meshgrid(v0_range,beta_range,indexing="ij")
vals = np.zeros_like(VVs)
for i, L_star in enumerate(L_stars):
    if PP.take(i) == p0_range[0]:
        i,j = np.where((VVs==VV.take(i))&(BBs==BB.take(i)))
        vals[i,j] = L_star[-1]

plt.imshow(vals[:,::6],aspect="auto")
plt.show()


def get_mean_self(Id):
    FILE = np.load("analysis/%d.npz" % Id)
    return FILE["mean_self"]

mean_self = np.array([get_mean_self(Id) for Id in range(N**3)])

VVs,BBs = np.meshgrid(v0_range,beta_range,indexing="ij")
vals = np.zeros_like(VVs)
for i, mean_S in enumerate(mean_self):
    if PP.take(i) == p0_range[0]:
        i,j = np.where((VVs==VV.take(i))&(BBs==BB.take(i)))
        vals[i,j] = mean_S[-1]

plt.imshow(vals,aspect="auto")
plt.show()

v0 = v0_range[-1]

mask = VV==v0

Id_sel = np.arange(N**3)[mask.ravel()]

levels = np.linspace(vals.min(),vals.max(),6)
plt.tricontourf(VVs.ravel(),BBs.ravel(),vals.ravel(),levels = levels)
plt.show()



def get_n_bound(Id):
    FILE = np.load("analysis/%d.npz" % Id)
    return FILE["n_bound"]

n_bound = np.array([get_n_bound(Id) for Id in range(N**3)])

v0 = v0_range[-1]

mask = VV==v0

Id_sel = np.arange(N**3)[mask.ravel()]

plt.tricontourf(PP.ravel()[Id_sel],BB.ravel()[Id_sel],n_bound[Id_sel,-1])
plt.show()





def get_mean_self(Id):
    FILE = np.load("analysis/%d.npz" % Id)
    return FILE["mean_self"]

mean_self = np.array([get_mean_self(Id) for Id in range(N**3)])


mask = BB==beta_range[-1]

Id_sel = np.arange(N**3)[mask.ravel()]

plt.tricontourf(PP.ravel()[Id_sel],VV.ravel()[Id_sel],mean_self[Id_sel,-1])
plt.show()


for i in range(512):
    # if BB.take(i) == 0:
    if BB.ravel()[i] == 0:
        plt.plot(mean_self[i],color="blue")
    # if VV.take(i) == v0_range[-1]:
plt.show()

MS = mean_self.reshape(8,8,8,mean_self.shape[-1])
for val in MS[:,:,0,-1]:
    plt.plot(val)
plt.show()