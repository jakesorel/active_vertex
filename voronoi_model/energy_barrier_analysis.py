import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sb
import multiprocessing
from joblib import delayed,Parallel

forward_files = os.listdir("energy_barrier/opt_v0/forward")
reverse_files = os.listdir("energy_barrier/opt_v0/reverse")

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



def remove_2nd(df,beta_range):
    dfs = []
    for i in np.arange(0,12,2):
        dfs.append(df.loc[df["beta"]==beta_range[i]])
    return pd.concat(dfs)


fig, ax = plt.subplots(1,2,sharey=True)
for i, t1_type in enumerate(["forward","reverse"]):
    df = make_df(t1_type)
    df = update_df(df,beta_dict,lattice_dict)
    df = remove_2nd(df,beta_range)
    df["val"] = np.log10(df["val"])
    mean_beta = [np.median(df.loc[df["beta"] == beta]["val"].values) for beta in beta_range[::2]]
    LQ_beta = [np.percentile(df.loc[df["beta"] == beta]["val"].values,25) for beta in beta_range[::2]]
    UQ_beta = [np.percentile(df.loc[df["beta"] == beta]["val"].values,75) for beta in beta_range[::2]]
    ax[i].plot(beta_range[::2],mean_beta)
    ax[i].fill_between(beta_range[::2],LQ_beta,UQ_beta,alpha=0.4)

    # ax[i].scatter(df["beta"],df["val"])
    # sb.boxplot(data = df, x = "beta",y = "val",ax = ax[i])
    ax[i].set(xscale="log")
    # ax[i].set(yscale="log")
fig.show()



####energy landscape
_, RRR,CCI = np.meshgrid(beta_range, rep_range,rep_range, indexing="ij")
n_t = 2000

t1_type = "forward"
dir_name = "energy_barrier/energies_mobile_i/%s"%t1_type
def extract_energies(file):
    try:
        return np.load("%s/%s"%(dir_name,file))["arr_0"]
    except:
        return np.ones(n_t)*np.nan


inputs = os.listdir(dir_name)
num_cores = multiprocessing.cpu_count()
out = Parallel(n_jobs=num_cores)(delayed(extract_energies)(inputt) for inputt in inputs)
Id = [int(input.split("_")[0]) for input in inputs]
li = [int(input.split("_")[1]) for input in inputs]
cll_i = [int(input.split("_")[2].split(".npz")[0]) for input in inputs]
t1_time = [int(float(np.loadtxt("energy_barrier_old/t1_time/%s/%d_%d_%d.txt"%(t1_type,Id[i],li[i],cll_i[i])))/0.025) for i in range(len(inputs))]
E_a = [outt[t1_time[i]] - outt[0] for i, outt in enumerate(out)]

df = pd.DataFrame({"Id":Id,"cll_i":cll_i,"E_a":E_a})
df = update_df(df,beta_dict,lattice_dict)
out_i = np.array(out)[np.where(np.array(cll_i)==0)[0]]
fig, ax = plt.subplots()
df = df.loc[df["cll_i"] == 0]
sb.lineplot(data = df,x = "beta",y = "E_a",ax=ax)
ax.set(xscale="log")
fig.show()

"""
Another alternative strategy -- constant and high v0, measure energy barrier
"""