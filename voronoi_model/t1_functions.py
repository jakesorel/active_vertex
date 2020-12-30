# from voronoi_model.voronoi_model_periodic import *
# from voronoi_model.voronoi_model_periodic import get_l_interface
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from numba import jit


def get_quartets(self):
    tc_type = self.c_types[self.tris]
    two_type1_tris = tc_type.sum(axis=1) == 2
    Ls = np.nonzero(two_type1_tris)[0]
    Is,Js = np.nonzero(tc_type[two_type1_tris]==1)
    type1_b_cells = self.tris[Ls[Is],Js]
    type0_cell = tc_type[two_type1_tris] == 0
    Is, Js = np.nonzero(type0_cell)
    # near_boundary_cells = np.ones_like(Is,dtype=np.int64)*-1
    quartets = np.ones((Is.size,4),dtype=np.int64)*-1
    k = 0
    for i,j in zip(Is,Js):
        m = Ls[i]
        opposite_tri = self.v_neighbours[m,j]
        opposite_cell = np.roll(self.tris[opposite_tri],-self.k2s[m,j])[0]
        if (self.c_types[opposite_cell]==1)*(opposite_cell not in type1_b_cells):
            # near_boundary_cells[k] = opposite_cell
            ordered_tri = np.roll(self.tris[m],-j)
            quartets[k,0] = opposite_cell
            quartets[k,1:] = ordered_tri
            k+=1
    # near_boundary_cells = near_boundary_cells[:k-1]
    quartets = quartets[:k-1]

    return quartets

def get_thetas(self,quartet):
    disp = np.mod(self.x[quartet[1]] - self.x[quartet[0]]+self.L/2,self.L)-self.L/2
    theta = np.arctan2(disp[1],disp[0])
    thetas = [theta,theta+np.pi,theta+np.pi/2,theta-np.pi/2]
    return thetas
#
# def get_near_boundary_cells(self):
#     ctype_score = self.c_types[self.tris].sum(axis=1)
#     tri_mask = (ctype_score == 1)+(ctype_score==2)
#     B_cells = np.arange(self.n_c,dtype=np.int64)[self.c_types.astype(np.bool)]
#     b_cells = np.unique(self.tris[tri_mask][self.c_types[self.tris[tri_mask]].astype(np.bool)])
#     nb_cells = np.unique(self.tris[self.v_neighbours[tri_mask]])
#     near_cells = list(set(list(B_cells)).intersection(set(list(nb_cells)).difference(set(list(b_cells)))))
#     return np.array(near_cells)
#
# def choose_near_boundary_cell(self):
#     near_cells = get_near_boundary_cells(self)
#     near_cell = near_cells[np.random.randint(near_cells.size)]
#     xnc = self.x[near_cell]
#     A_cells = np.arange(self.n_c,dtype=np.int64)[~self.c_types.astype(np.bool)]
#     xA = self.x[A_cells]
#     disp = np.mod(xA - xnc +self.L/2,self.L)-self.L/2
#     dist = np.linalg.norm(disp,axis=1)
#     dir = disp[dist.argmin()]/dist.min()
#     theta = np.arctan2(dir[1],dir[0])
#     neigh_cell = dist.argmin()
#     return near_cell,theta,neigh_cell
#
# def find_quartet(self,near_cell,neigh_cell,theta):
#     other_two = list(set(list(self.tris[(self.tris==near_cell).any(axis=1)].ravel())).intersection(list(self.tris[(self.tris == neigh_cell).any(axis=1)].ravel())))
#     tri_i = np.nonzero((self.tris==near_cell).any(axis=1)*(self.tris==other_two[0]).any(axis=1) * (self.tris==other_two[1]).any(axis=1))[0][0]
#     Tri = self.tris[tri_i]
#     near_i = np.where(Tri == near_cell)[0][0]
#     if self.tris[tri_i,np.mod(near_i+1,3)] == other_two[1]:
#         other_two = other_two[1],other_two[0]
#     thetas = [theta,theta+np.pi,theta-np.pi/2,theta+np.pi/2]
#     Is = [near_cell,neigh_cell,other_two[0],other_two[1]]
#     return Is,thetas
#
# def make_T1(self):
#     Is = 0
#     while Is == 0:
#         try:
#             near_cell, theta, neigh_cell = choose_near_boundary_cell(self)
#             Is, thetas = find_quartet(self, near_cell, neigh_cell, theta)
#         except IndexError:
#             pass
#     return np.array(Is), np.array(thetas)


def generate_noise_fixed(self,Is,thetas):
    theta_noise = np.cumsum(np.random.normal(0, np.sqrt(2 * self.Dr * self.dt), (self.n_t, self.n_c)),
                            axis=0) + np.random.uniform(0, np.pi * 2, self.n_c)
    # for i,theta in zip(np.array(Is)[[0,2,3]],np.array(thetas)[[0,2,3]]):
    for i, theta in zip(Is,thetas):
        theta_noise[:,i] = theta
    self.noise = np.dstack((np.cos(theta_noise), np.sin(theta_noise)))
    if self.cell_movement_mask is not None:
        self.noise[:, ~self.cell_movement_mask] = self.noise[:, ~self.cell_movement_mask] * 0
    if self.no_noise_time is not None:
        self.noise[:self.no_noise_time] = 0 * self.noise[:self.no_noise_time]


def interpolate_x(self,x_save,t1_time,t_eval_i,t_eval):
    x_save_sample = x_save[t1_time+t_eval_i].copy()
    dx_save_sample = np.mod(x_save_sample - x_save_sample[0]+self.L/2,self.L)-self.L/2
    x_save_fine = np.zeros((t_eval.size,x_save.shape[1],2))
    for j, x in enumerate(dx_save_sample.transpose(1,0,2)):
        for i in range(2):
            fn =interp1d(t_eval_i,x[:,i])
            x_save_fine[:,j,i] = x_save_sample[0,j,i]+fn(t_eval)
    return np.mod(x_save_fine,self.L)

def get_energy(self,x,kappa_A,kappa_P,J,get_l_interface,):
    self.x = x
    self._triangulate_periodic(x)
    self.assign_vertices()
    A = self.get_A_periodic(self.neighbours,self.vs)
    P = self.get_P_periodic(self.neighbours,self.vs)
    l_int = get_l_interface(self.n_v, self.n_c, self.neighbours, self.vs, self.CV_matrix, self.L)
    energy = np.sum(kappa_A*(A-self.A0)**2) + np.sum(kappa_P*(P-self.P0)**2) + np.sum(l_int*J)
    return energy

@jit(nopython=True)
def anyaxis1(boolmat):
    return boolmat[:,0]+boolmat[:,1]+boolmat[:,2]

@jit(nopython=True)
def reset_v0(Is,tris):
    return (anyaxis1(Is[0]==tris)*anyaxis1(Is[1]==tris)).sum()!=0
