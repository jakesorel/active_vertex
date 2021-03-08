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
rep_range = np.arange(20)
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


fig, ax = plt.subplots(figsize=(4,4))
for i, t1_type in enumerate(["forward","reverse"]):
    df = make_df(t1_type)
    df = update_df(df,beta_dict,lattice_dict)
    # df = remove_2nd(df,beta_range)
    df["val"] = np.log10(df["val"])
    mean_beta = [np.median(df.loc[df["beta"] == beta]["val"].values) for beta in beta_range]
    LQ_beta = [np.percentile(df.loc[df["beta"] == beta]["val"].values,25) for beta in beta_range]
    UQ_beta = [np.percentile(df.loc[df["beta"] == beta]["val"].values,75) for beta in beta_range]
    ax.plot(np.log10(beta_range),mean_beta,label=t1_type)
    ax.fill_between(np.log10(beta_range),LQ_beta,UQ_beta,alpha=0.4)

    # ax[i].scatter(df["beta"],df["val"])
    # sb.boxplot(data = df, x = "beta",y = "val",ax = ax[i])
    # ax.set(xscale="log")
    # ax[i].set(yscale="log")
ax.set(xlabel=r"$log_{10} \beta$",ylabel=r"$log_{10} \ v_{crit}$")
ax.legend(loc=2)
fig.tight_layout()
fig.show()



####energy landscape
_, RRR,CCI = np.meshgrid(beta_range, rep_range,rep_range, indexing="ij")
n_t = 800
n_tlong = 2000


def extract_energies(file):
    try:
        return np.load("%s/%s"%(dir_name,file))["arr_0"]
    except:
        return np.ones(n_t)*np.nan

t1_type = "reverse"
dir_name = "energy_barrier/energies_tot/%s"%t1_type
inputs = os.listdir(dir_name)
num_cores = multiprocessing.cpu_count()
out = Parallel(n_jobs=num_cores)(delayed(extract_energies)(inputt) for inputt in inputs)
Id = [int(input.split("_")[0]) for input in inputs]
li = [int(input.split("_")[1]) for input in inputs]
cll_i = [int(input.split("_")[2].split(".npz")[0]) for input in inputs]
t1_time = [int(float(np.loadtxt("energy_barrier/t1_time/%s/%d_%d_%d.txt"%(t1_type,Id[i],li[i],cll_i[i])))/0.025) for i in range(len(inputs))]
# E_a = [outt[t1_time[i]] - outt[0] for i, outt in enumerate(out)]
E_a = [outt.max() - outt[0] for i, outt in enumerate(out)]

df = pd.DataFrame({"Id":Id,"cll_i":cll_i,"E_a":E_a})
df = update_df(df,beta_dict,lattice_dict)
out_i = np.array(out)[np.where(np.array(cll_i)==0)[0]]
fig, ax = plt.subplots()
df = df.loc[df["cll_i"] == 0]
sb.lineplot(data = df,x = "beta",y = "E_a",ax=ax)
ax.set(xscale="log")
fig.show()


def plot_E_a(ax,t1_type="forward",col="red",label="Sorting",scale="linear"):
    # t1_type = "reverse"
    dir_name = "energy_barrier/energies_tot/%s" % t1_type

    def extract_energies(file):
        try:
            return np.load("%s/%s" % (dir_name, file))["arr_0"]
        except:
            return np.ones(n_t) * np.nan

    inputs = os.listdir(dir_name)
    num_cores = multiprocessing.cpu_count()
    out = Parallel(n_jobs=num_cores)(delayed(extract_energies)(inputt) for inputt in inputs)
    Id = [int(input.split("_")[0]) for input in inputs]
    li = [int(input.split("_")[1]) for input in inputs]
    cll_i = [int(input.split("_")[2].split(".npz")[0]) for input in inputs]
    t1_time = [
        int(float(np.loadtxt("energy_barrier/t1_time/%s/%d_%d_%d.txt" % (t1_type, Id[i], li[i], cll_i[i]))) / 0.025) for
        i in range(len(inputs))]


    df2 = pd.DataFrame({"Id":Id,"cll_i":cll_i,"E":out})
    df2 = update_df(df2,beta_dict,lattice_dict)
    df2["dE"] = [vals[:-1] - vals[1:] for vals in df2["E"]]
    df2["max dE"] = [np.max(np.abs(val)) for val in df2["dE"]]
    df2["mask"] = [val<0.05 for val in df2["max dE"]]
    df2["t1_time"] = t1_time
    df2["Ea"] = [val[t1_time] - val[400] if mask else np.nan for mask, val, t1_time in zip(df2["mask"],df2["E"],df2["t1_time"])]
    df2["log beta"] = np.log10(df2["beta"])
    df2["log Ea"] = np.log10(df2["Ea"])
    if scale == "linear":
        sb.lineplot(data=df2, x="beta", y="Ea", ax=ax, color=col, label=label)
        ax.set(xlabel=r"$\beta$",ylabel=r"$E_a$")
    if scale == "log":
        sb.lineplot(data = df2,x="log beta",y="log Ea",ax=ax,color=col,label=label)
        ax.set(xlabel=r"$log_{10} \ \beta$",ylabel=r"$log_{10} \ E_a$")

def make_real_t1_fig(scale="linear"):
    fig, ax = plt.subplots(figsize=(3.5,2.5))
    plot_E_a(ax,"forward",plt.cm.inferno(0.2),"Sorting",scale)
    plot_E_a(ax,"reverse",plt.cm.inferno(0.8),"Unsorting",scale)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig.subplots_adjust(top=0.8, bottom=0.2, left=0.25, right=0.6)
    fig.savefig("paper_plots/Fig3/real_t1s_%s_3.pdf"%scale,dpi=300)

make_real_t1_fig("linear")
make_real_t1_fig("log")


def plot_energy(t1_type="forward"):
    dir_name = "energy_barrier/energies_tot/%s" % t1_type
    inputs = os.listdir(dir_name)
    num_cores = multiprocessing.cpu_count()

    def extract_energies(file):
        try:
            return np.load("%s/%s" % (dir_name, file))["arr_0"][:n_tlong]
        except:
            return np.ones(n_tlong) * np.nan

    out = Parallel(n_jobs=num_cores)(delayed(extract_energies)(inputt) for inputt in inputs)
    Id = [int(input.split("_")[0]) for input in inputs]
    li = [int(input.split("_")[1]) for input in inputs]
    cll_i = [int(input.split("_")[2].split(".npz")[0]) for input in inputs]
    t1_time = [
        int(float(np.loadtxt("energy_barrier/t1_time/%s/%d_%d_%d.txt" % (t1_type, Id[i], li[i], cll_i[i]))) / 0.025) for
        i in range(len(inputs))]


    df2 = pd.DataFrame({"Id":Id,"cll_i":cll_i,"E":out})
    df2 = update_df(df2,beta_dict,lattice_dict)
    df2["dE"] = [vals[:-1] - vals[1:] for vals in df2["E"]]
    df2["max dE"] = [np.max(np.abs(val)) for val in df2["dE"]]
    df2["mask"] = [val<0.05 for val in df2["max dE"]]
    df2["t1_time"] = t1_time

    fig, ax = plt.subplots(figsize=(3.3,2.5))
    cols = plt.cm.plasma(np.linspace(0,1,12))
    for j, beta in enumerate(beta_range):
        # beta = beta_range[0]
        n_sample = (df2["beta"] == beta).sum()
        E_mat = np.ones((n_sample,2*n_tlong))*np.nan
        df_sample = df2.loc[df2["beta"] == beta]
        for i, (E,t1_time,mask) in enumerate(zip(df_sample["E"],df_sample["t1_time"],df_sample["mask"])):
            if mask:
                E_mat[i,n_tlong-t1_time+400:2*n_tlong-t1_time] = E[400:] - E[400]

        ax.plot(np.arange(-n_tlong*0.025,n_tlong*0.025,0.025)[:n_t+400],np.nanmean(E_mat,axis=0)[:n_tlong+400],color=cols[j])
    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma,
                               norm=plt.Normalize(vmax=np.log10(beta_range.max()), vmin=np.log10(beta_range.min())))
    sm._A = []
    cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.085, aspect=10, orientation="vertical")
    cl.set_label(r"$log_{10} \ \beta$")
    fig.subplots_adjust(top=0.8, bottom=0.2, left=0.25, right=0.8)
    ax.set_title(t1_type)
    # ax.plot((2000,2000),(0.054,0.059),color="k")
    ax.set(xlabel="t",ylabel=r"$\Delta \epsilon$")
    fig.savefig("paper_plots/Fig3/real_energy_profiles_%s_2.pdf"%t1_type,dpi=300)

plot_energy("forward")
plot_energy("reverse")



def E_a_2(t1_type="forward"):
    dir_name = "energy_barrier/energies_tot/%s" % t1_type

    def extract_energies(file):
        try:
            return np.load("%s/%s" % (dir_name, file))["arr_0"][:n_tlong]
        except:
            return np.ones(n_tlong) * np.nan

    inputs = os.listdir(dir_name)
    num_cores = multiprocessing.cpu_count()
    out = Parallel(n_jobs=num_cores)(delayed(extract_energies)(inputt) for inputt in inputs)
    Id = [int(input.split("_")[0]) for input in inputs]
    li = [int(input.split("_")[1]) for input in inputs]
    cll_i = [int(input.split("_")[2].split(".npz")[0]) for input in inputs]
    t1_time = [
        int(float(np.loadtxt("energy_barrier/t1_time/%s/%d_%d_%d.txt" % (t1_type, Id[i], li[i], cll_i[i]))) / 0.025) for
        i in range(len(inputs))]


    df2 = pd.DataFrame({"Id":Id,"cll_i":cll_i,"E":out})
    df2 = update_df(df2,beta_dict,lattice_dict)
    df2["dE"] = [vals[:-1] - vals[1:] for vals in df2["E"]]
    df2["max dE"] = [np.max(np.abs(val)) for val in df2["dE"]]
    df2["mask"] = [val<0.1 for val in df2["max dE"]]
    df2["t1_time"] = t1_time
    dicts = []
    for j, beta in enumerate(beta_range):
        # beta = beta_range[0]
        n_sample = (df2["beta"] == beta).sum()
        E_mat = np.ones((n_sample,2*n_tlong))*np.nan
        df_sample = df2.loc[df2["beta"] == beta]
        for i, (E,t1_time,mask) in enumerate(zip(df_sample["E"],df_sample["t1_time"],df_sample["mask"])):
            if mask:
                E_mat[i,n_tlong-t1_time+400:2*n_tlong-t1_time] = E[400:] - E[400]
        E_a = E_mat[:,n_tlong]
        E_a = E_a[~np.isnan(E_a)]
        dicts.append(pd.DataFrame({"E_a":E_a,"beta":np.repeat(beta,E_a.size)}))
    df3 = pd.concat(dicts)
    return df3

Ear = E_a_2("reverse")
Eaf = E_a_2("forward")
fig, ax = plt.subplots()
sb.lineplot(data=Eaf,x="beta",y="E_a",ax=ax)
sb.lineplot(data=Ear,x="beta",y="E_a",ax=ax)
fig.show()




plt.plot(Eaf)
plt.plot(Ear)
plt.show()

