import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sb

forward_files = os.listdir("energy_barrier/opt_v0/forward")
reverse_files = os.listdir("energy_barrier/opt_v0/forward")

def make_df(t1_type):
    files = os.listdir("energy_barrier/opt_v0/%s"%t1_type)
    dics = []
    for file in files:
        Id,__,cid = file.split("_")
        cid = cid.split(".txt")[0]
        val = float(np.loadtxt("energy_barrier/opt_v0/%s/%s"%(t1_type,file)))
        dics.append({"Id":Id,"cid":cid,"val":val})
    return pd.DataFrame(dics)

beta_range = np.logspace(-3, -1, 12)
rep_range = np.arange(12)
BB, RR = np.meshgrid(beta_range, rep_range, indexing="ij")
beta_dict = dict(zip(np.arange(BB.size),BB.ravel()))
lattice_dict = dict(zip(np.arange(BB.size),RR.ravel()))


def update_df(df,beta_dict,lattice_dict):
    beta = []
    lattice = []
    for Id in df["Id"].values:
        Id = int(Id)
        beta.append(beta_dict[Id])
        lattice.append(lattice_dict[Id])
    df["beta"] = beta
    df["lattice"] = lattice
    return df




fig, ax = plt.subplots(1,2,sharey=True)
for i, t1_type in enumerate(["forward","reverse"]):
    df = make_df(t1_type)
    df = update_df(df,beta_dict,lattice_dict)
    mean_beta = [df.loc[df["beta"] == beta]["val"].mean() for beta in beta_range]
    ax[i].plot(beta_range,mean_beta)
    # sb.lineplot(data = df, x = "beta",y = "val",hue="lattice",ax = ax[i])
    ax[i].set(xscale="log")
fig.show()