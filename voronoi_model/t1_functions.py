# from voronoi_model.voronoi_model_periodic import *
# from voronoi_model.voronoi_model_periodic import get_l_interface
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from numba import jit
#

@jit(nopython=True)
def disp_from_min_dist(x,X,L):
    disp = np.mod(X - x + L/2,L)-L/2
    dist = np.sqrt(disp[:,0]**2 + disp[:,1]**2)
    mindisti = np.argmin(dist)
    mindist = dist[mindisti]
    return disp[mindisti]/mindist, mindist


@jit(nopython=True)
def get_mobile_dir(tris,c_types,v_neighbours,neighbours,L,vs,mobile_i,t1_type="forward"):
    """
    For a single A cell moving in (forward) or out (reverse) of the main A aggregate

    Mobile_i must be of type "0" (A)

    :param tris:
    :param mobile_i:
    :param t1_type:
    :return:
    """
    if t1_type == "forward":
        tot = 2
    if t1_type == "reverse":
        tot = 3
    m_tri_mask = (tris[:,0] == mobile_i)+(tris[:,1] == mobile_i)+(tris[:,2] == mobile_i)
    m_tris_i = np.arange(tris.shape[0])[m_tri_mask]
    direc_mat,dist_mat = np.ones((m_tris_i.size,2),dtype=np.float64)*np.nan,np.ones(m_tris_i.size,dtype=np.float64)*np.nan
    n_a_neighbours = np.sum(1 - c_types[tris[m_tris_i].ravel()].reshape(-1,3)) - m_tris_i.size ##not a real meausre, but a proxy for number of "A" type neighbours
    if (n_a_neighbours==0)*(t1_type == "forward") + (n_a_neighbours>0)*(t1_type=="reverse"):
        for j, i in enumerate(m_tris_i):
            tri = tris[i]
            v = vs[i]
            if c_types[tri].sum() ==2: #ensure the two neighbouring cells are of type "1" (B)
                neigh_v = neighbours[i]
                neighbour_tris = tris[v_neighbours[i]]
                neigh_mask = c_types[neighbour_tris.ravel()].reshape(-1,3)
                neigh_mask = (neigh_mask[:,0]+neigh_mask[:,1]+neigh_mask[:,2]) == tot
                non_self_mask = neighbour_tris != mobile_i
                non_self_mask = non_self_mask[:,0]*non_self_mask[:,1]*non_self_mask[:,2]
                neigh_v = neigh_v[neigh_mask*non_self_mask]
                # neigh_v_size = neigh_v.size
                if neigh_v.size == 2:
                    disp = np.mod(neigh_v.ravel() - v + L/2,L) - L/2
                    dist = np.sqrt(disp[0]**2 + disp[1]**2)
                    direc = disp/dist
                    direc_mat[j], dist_mat[j] = direc, dist
                if neigh_v.size > 2:
                    direc,dist = disp_from_min_dist(v,neigh_v,L)
                    direc_mat[j],dist_mat[j] = direc,dist
        if np.isnan(dist_mat).all():
            direc = np.array((0.0,0.0))
        else:
            real_mask = ~np.isnan(dist_mat)
            dist_mat = dist_mat[real_mask]
            direc_mat = direc_mat[real_mask]
            direc = direc_mat[np.argmin(dist_mat)]

    else:
        direc = np.array((0.0,0.0))
    return direc
#
# @jit(nopython=True)
# def disp_from_min_dist(x,X,L):
#     nX = X.shape[0]
#     nx = x.shape[0]
#     dx,dy = np.outer(X[:,0],np.ones(nx)) - np.outer(np.ones(nX),x[:,0]),np.outer(X[:,1],np.ones(nx)) - np.outer(np.ones(nX),x[:,1])
#     disp = np.column_stack((dx.ravel(),dy.ravel()))
#     disp = np.mod(disp + L/2,L)-L/2
#     dist = np.sqrt(disp[:,0]**2 + disp[:,1]**2)
#     mindisti = np.argmin(dist)
#     return disp[mindisti]/dist[mindisti]
#
# def get_dir_forward(self,i):
#     tc_type = self.c_types[self.tris]
#     i_tris = (self.tris==i).any(axis=1)
#     two_type1_tris = (tc_type.sum(axis=1) == 2)*(~i_tris)
#     neigh_vs = self.vs[two_type1_tris]
#     cell_vs = self.vs[i_tris]
#     return disp_from_min_dist(cell_vs,neigh_vs,self.L)


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



def get_quartets_reverse(self):
    ##Find typeB boundary cells (which will be flipped to type 0)
    tc_type = self.c_types[self.tris]
    two_type1_tris = tc_type.sum(axis=1) == 2
    Ls = np.nonzero(two_type1_tris)[0]
    Is,Js = np.nonzero(tc_type[two_type1_tris]==1)
    type1_b_cells = self.tris[Ls[Is],Js]
    quartets = []
    ##2. For each, find tris where all are typeB. Then find those that have an opposite that is also type B
    for type1_cell in type1_b_cells:
        three_type1_tri_mask = ((tc_type.T*(self.tris == type1_cell).any(axis=1).T).sum(axis=0) == 3)
        three_type1_tris = self.tris[three_type1_tri_mask]
        js = np.nonzero(three_type1_tris == type1_cell)[1]
        for tri_i,tri,j in zip(np.nonzero(three_type1_tri_mask)[0],three_type1_tris,js):
            tri = np.roll(tri,-j)
            opposite_tri = self.v_neighbours[tri_i,j]
            if tri[0] not in self.tris[opposite_tri]:
                opposite_cell = np.roll(self.tris[opposite_tri], -self.k2s[tri_i, j])[0]
                if self.c_types[opposite_cell]==1:
                    quartet = np.array((tri[0],opposite_cell,tri[1],tri[2]))
                    # quartet = np.concatenate([[opposite_cell],tri])
                    quartets.append(quartet)
    return np.array(quartets)


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

def get_energy(self,x,kappa_A,kappa_P,J,get_l_interface):
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
