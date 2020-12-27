import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"
from scipy.interpolate import bisplrep,bisplev
import dask
from dask.distributed import Client
from scipy.interpolate import bisplrep,bisplev

"""
For figure 1. A 3 variable phase diagram for beta, v0 and p0. 

The script takes the n_islands output and calculates the phase boundary (1/2 way between max and min)

Then by spline interpolation, estimates the 3D shell of this boundary. 

Finally plots this output

"""


###1. Load the data

n_slurm_tasks = 8
client = Client(threads_per_worker=1, n_workers=n_slurm_tasks, memory_limit="1GB")
N = 10
rep = 8
p0_range = np.linspace(3.5, 4, N)
v0_range = np.linspace(5e-3, 1e-1, N)
beta_range = np.logspace(-3, -1, N)
rep_range = np.arange(rep)
PP, VV, BB,RR = np.meshgrid(p0_range, v0_range, beta_range,rep_range, indexing="ij")
ID_mat = np.arange(N**3).astype(int).reshape(N,N,N)
ID_mat = np.stack([ID_mat for i in range(rep)],axis=3)


def get_n_islands(X):
    Id, Rep = X
    try:
        FILE = np.load("analysis/%d_%d.npz" % (Id,Rep))
        return FILE["n_islands"]
    except FileNotFoundError:
        return np.ones(100)*np.nan


inputs = np.array([ID_mat.ravel(),RR.ravel()]).T
inputs = inputs.astype(np.int64)
lazy_results = []
for inputt in inputs:
    lazy_result = dask.delayed(get_n_islands)(inputt)
    lazy_results.append(lazy_result)
out_nislands = dask.compute(*lazy_results)
out_nislands = np.array(out_nislands).reshape(RR.shape[0],RR.shape[1],RR.shape[2],RR.shape[3],2,100)
n_islands_tot = out_nislands.sum(axis=-2)

##2. Calculate the mean number of islands

n_islands_tot_mean = n_islands_tot[:,:,:,:,-1].mean(axis=-1)


##3. Find the phase boundary

ni_min,ni_max = np.percentile(n_islands_tot_mean,5),np.percentile(n_islands_tot_mean,95)
ni_mid = (ni_max+ni_min)/2
deviation = np.log10((n_islands_tot_mean - ni_mid)**2 + 1 - ((n_islands_tot_mean - ni_mid)**2).min())
weights = (deviation.max() - deviation)/(deviation.max() - deviation.min())
PP, VV, BB = np.meshgrid(p0_range, v0_range, beta_range, indexing="ij")
thresh=0.85
X,Y,Z = PP[weights>thresh],VV[weights>thresh],np.log10(BB[weights>thresh])


##4. Clean the points, finding the mean of degenerate p0,beta pairs

def clean_points(X,Z,Y):
    XX,ZZ = np.meshgrid(p0_range,np.log10(beta_range),indexing="ij")
    X_new,Y_new,Z_new = [],[],[]
    for x,z in zip(XX.ravel(),ZZ.ravel()):
        ids = np.where((X==x)&(Z==z))[0]
        if ids.size!=0:
            X_new.append(x)
            Z_new.append(z)
            print(ids.size)
            Y_new.append(Y[ids].mean())
    return np.array(X_new),np.array(Z_new),np.array(Y_new)

X_new,Z_new,Y_new = clean_points(X,Z,Y)
points = np.column_stack((X_new,Z_new,Y_new))


##5. Interpolate using a spline

nfine = 2000
p0_spacefine, v0_spacefine,beta_spacefine = np.linspace(p0_range.min(),p0_range.max(), nfine), np.linspace(v0_range.min(), v0_range.max(), nfine),np.logspace(np.log10(beta_range.min()),np.log10(beta_range.max()),nfine)


PP,lBB = np.meshgrid(p0_spacefine,np.log10(beta_spacefine),indexing="ij")
z = bisplev(p0_spacefine,np.log10(beta_spacefine), bisplrep(X_new.ravel(),Z_new.ravel(),Y_new.ravel(),s=0.1))



##6. Remove extrapolated regions from the interpolated dataset

z[z>v0_range.max()*0.9] = np.nan

##7. Set the contour preferences

n_cont = 10
d_beta = (np.log10(beta_range).max()-np.log10(beta_range).min())/n_cont
d_p0 = (p0_range.max()-p0_range.min())/n_cont
d_v0 = (v0_range.max()-v0_range.min())/n_cont


##8. Set the lighting and camera preferences

lighting_effects = dict(ambient=0.4, diffuse=0.5, roughness = 0.9, specular=0.6, fresnel=0.2)

camera = dict(
    up=dict(x=1, y=0, z=0),
    eye = dict(x=2, y=2, z=2))


##9. Make the figure

fig = go.Figure(data=[go.Surface(z=z,x=p0_spacefine,y=np.log10(beta_spacefine),
                                 cmax=0.5,cmin=0.4999,lighting=lighting_effects,
                                 contours={
                                     "y": {"show": True, "color": "white","size":d_beta,"start":np.log10(beta_spacefine).min(),"end":np.log10(beta_spacefine).max()},
                                    "z": {"show": True, "color": "white", "size": d_v0,"start":v0_range.min(),"end":v0_range.max()}}

                                    )])


fig.update_scenes(zaxis_autorange="reversed",yaxis_autorange="reversed")
fig.update_layout(autosize=True,
                  width=500, height=500,scene_camera=camera,scene=dict(xaxis_title="p₀",yaxis_title="log₁₀ β",zaxis_title="v₀",xaxis= {"nticks": 5},yaxis= {"nticks": 5},zaxis= {"nticks": 5}))


##8. View the figure

fig.show()

##9. Save the figure

fig.write_image("3 var phase_diagram.pdf",width=1000,height=1000)
fig.write_image("3 var phase_diagram 2.pdf")


