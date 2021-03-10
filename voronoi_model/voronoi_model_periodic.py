import triangle as tr
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import time
from scipy.spatial import Voronoi, voronoi_plot_2d,Delaunay
import os
from matplotlib import animation
from line_profiler import LineProfiler
import math
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib import cm
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

class Cell:
    def __init__(self):
        self.v_id = []
        self.vs = []
        self.Del_n = []
        self.x = []

    def sort_vertices(self):
        angles = np.arctan2(self.vs[:, 1] - self.x[1], self.vs[:, 0] - self.x[0])
        self.vs = np.array([self.vs])[np.argsort(angles)]
        self.v_id = np.array([self.v_id])[np.argsort(angles)]
        self.Del_n = np.array([self.Del_n])[np.argsort(angles)]

    def get_P(self):
        self.P = np.sum(np.sqrt(((self.vs - np.roll(self.vs,1,axis=0))**2).sum()))

    def get_A(self):
        v_roll = np.roll(self.vs,1,axis=0)
        self.A = 0.5*np.sum(np.absolute(self.vs[:,0]*v_roll[:,1] - self.vs[:,1]*v_roll[:,0]))

class Tissue:
        def __init__(self):
            self.n_c = []
            self.n_v = []
            self.cells = []
            self.x0 = []
            self.vs = []
            self.tris = []

            self.A0 = 1
            self.P0 = 3.4641
            self.kappa_A = 1
            self.kappa_P = 1

            self.J = []
            self.c_types = []

            self.k2s = []

            self.grid_x,self.grid_y = np.mgrid[-1:2,-1:2]
            self.grid_x[0,0],self.grid_x[1,1] = self.grid_x[1,1],self.grid_x[0,0]
            self.grid_y[0,0],self.grid_y[1,1] = self.grid_y[1,1],self.grid_y[0,0]
            self.grid_xy = np.array([self.grid_x.ravel(),self.grid_y.ravel()]).T

            self.x_save = []
            self.tri_save = []

            self.cols = (1,0,0,0.5),(0,0,1,0.5)
            self.plot_scatter = True
            self.cell_movement_mask = None
            self.no_noise_time = None
            self.b_extra = 2
            self.noise = []
            self.t1_time = False


            self.plot_forces = False

        def generate_cells(self,n_c):
            """
            Generate the cells.

            Saves **Cell** objects, which will be utilised later in the simulation.
            :param n_c: number of cells (np.int32)
            :return: self.cells, a list of **Cell** objects
            """
            self.n_c = n_c
            self.cells = [Cell() for i in range(n_c)]
            return self.cells

        def hexagonal_lattice(self,rows=3, cols=3, noise=0.0005, A = None):
            """
            Assemble a hexagonal lattice

            :param rows: Number of rows in lattice
            :param cols: Number of columns in lattice
            :param noise: Noise added to cell locs (Gaussian SD)
            :return: points (nc x 2) cell coordinates.
            """
            points = []
            for row in range(rows * 2):
               for col in range(cols):
                   x = (col + (0.5 * (row % 2))) * np.sqrt(3)
                   y = row * 0.5
                   x += np.random.normal(0,noise)
                   y += np.random.normal(0,noise)
                   points.append((x, y))
            points = np.asarray(points)
            if A is not None:
                points = points * np.sqrt(2*np.sqrt(3)/3)
            return points


        def make_init(self,L,noise=0.005):
            """
            Make initial condition. Currently, this is a hexagonal lattice + noise

            Makes reference to the self.hexagonal_lattice function, then crops down to the reference frame

            Stores:
                self.n_c = number of cells
                self.x0 = (nc x 2) matrix denoting cell coordinates
                self.x = clone of self.x0

            :param L: Domain size/length (np.float32)
            :param noise: Gaussian noise added to {x,y} coordinates (np.float32)
            """
            self.L = L
            self.x0 = self.hexagonal_lattice(int(np.ceil(self.L/0.5)),int(np.ceil(self.L/np.sqrt(3))),noise=noise)
            # self.x0 = self.hexagonal_lattice(self.n_c,self.n_c,noise=noise)
            # self.x0 = self.x0[self.x0.max(axis=1) < L*0.95]
            self.x0 += 1e-3
            self.x0 = self.x0[self.x0.max(axis=1) < L*0.97]
            self.x = self.x0
            self.n_c = self.x0.shape[0]
            self.n_C = self.n_c



        def make_init_balanced(self,L,noise=0.005):
            """
            Make initial condition. Currently, this is a hexagonal lattice + noise

            Makes reference to the self.hexagonal_lattice function, then crops down to the reference frame

            Stores:
                self.n_c = number of cells
                self.x0 = (nc x 2) matrix denoting cell coordinates
                self.x = clone of self.x0

            :param L: Domain size/length (np.float32)
            :param noise: Gaussian noise added to {x,y} coordinates (np.float32)
            """
            self.L = L
            self.x0 = self.hexagonal_lattice(int(self.L),int(np.ceil(self.L)),noise=noise,A = self.A0)
            # self.x0 = self.hexagonal_lattice(self.n_c,self.n_c,noise=noise)
            # self.x0 = self.x0[self.x0.max(axis=1) < L*0.95]
            self.x0 += 1e-3
            np.argsort(self.x0.max(axis=1))

            self.x0 = self.x0[np.argsort(self.x0.max(axis=1))[:int(self.L**2/self.A0)]]
            self.x = self.x0
            self.n_c = self.x0.shape[0]
            self.n_C = self.n_c



        def make_init_boundary(self,L,r,noise=0.005):
            self.make_init(L,noise=noise)
            self._triangulate(self.x0)
            circular_mask = (self.x0[:,0] - self.L/2)**2 + (self.x0[:,1] - self.L/2)**2 <= (r*L)**2
            neighs = []
            for i, tri in enumerate(self.tris):
                In = 0
                for c in tri:
                    if circular_mask[c]:
                        In +=1
                if In !=0:
                    for c in tri:
                        neighs.append(c)
            kept = list(np.nonzero(circular_mask)[0])
            boundary_particles = list(set(neighs).difference(set(list(kept))))

            self.x = self.x[kept+boundary_particles]
            self.x0 = self.x.copy()
            self.n_c = self.x0.shape[0]
            self.n_C = len(kept)



        def make_init_boundary_periodic(self,L,r,noise=0.005):
            self.make_init(L,noise=noise)
            self._triangulate_periodic(self.x0)
            circular_mask = (self.x0[:,0] - self.L/2)**2 + (self.x0[:,1] - self.L/2)**2 <= (r*L)**2
            self.x = np.vstack((self.x[circular_mask],self.x[~circular_mask]))
            self.x0 = self.x.copy()
            self.n_c = self.x0.shape[0]
            self.n_C = circular_mask.sum()



        def set_interaction(self,W = 0.16*np.array([[2, 0.5], [0.5, 2]]),pE = 0.5,c_types=None,randomize=True):
            if c_types is None:
                nE = int(self.n_C*pE)
                N_dict = {"E": nE, "T": self.n_C - nE,}

                c_types = np.zeros(self.n_C, dtype=np.int32)
                j = 0
                for k, c_type in enumerate(N_dict):
                    j1 = N_dict[c_type]
                    c_types[j:j + j1] = k
                    j += j1
                if randomize is True:
                    np.random.shuffle(c_types)

            if self.n_c!=self.n_C:
                c_types = np.concatenate((c_types,np.repeat(-1,self.n_c-self.n_C)))

            cell_i, cell_j = np.meshgrid(c_types, c_types, indexing="ij")
            J = W[cell_i, cell_j]
            self.J = J
            self.c_types = c_types


        def set_interaction_boundary(self,W = 0.16*np.array([[2, 0.5], [0.5, 2]]),pE = 0.5,Wb = [0.16,0.16],b_extra = 3):
            self.b_extra = b_extra

            nE = int(self.n_C*pE)
            N_dict = {"E": nE, "T": self.n_C - nE}

            c_types = np.zeros(self.n_C, dtype=np.int32)
            j = 0
            for k, c_type in enumerate(N_dict):
                j1 = N_dict[c_type]
                c_types[j:j + j1] = k
                j += j1
            np.random.shuffle(c_types)

            c_types_all = np.concatenate((c_types,np.repeat(-1,self.n_c-self.n_C)))

            cell_i, cell_j = np.meshgrid(c_types, c_types, indexing="ij")
            J = W[cell_i, cell_j]
            self.J = J
            self.c_types = c_types
            self.c_types_all = c_types_all


            #Save a larger matrix from which to sample. This defines the interaction strengths of cells with the boundary
            self.J_large = np.zeros((self.n_c*self.b_extra,self.n_c*self.b_extra))
            for i, c_type in enumerate(self.c_types):
                self.J_large[i] = Wb[c_type]
                self.J_large[:,i] = Wb[c_type]
            self.J_large[:self.n_C,:self.n_C] = self.J



        def set_interaction_boundary_ETX(self,W = 0.16*np.array([[2, 0.5], [0.5, 2]]),pE = 0.3,pX=0.3):
            nE = int(self.n_C*pE)
            nX = int(self.n_C*pX)

            N_dict = {"E": nE, "T": self.n_C-nE-nX,"X":pX}

            c_types = np.zeros(self.n_C, dtype=np.int32)
            j = 0
            for k, c_type in enumerate(N_dict):
                j1 = N_dict[c_type]
                c_types[j:j + j1] = k
                j += j1
            np.random.shuffle(c_types)

            c_types_all = np.concatenate((c_types,np.repeat(-1,self.n_c-self.n_C)))

            cell_i, cell_j = np.meshgrid(c_types, c_types, indexing="ij")
            J = W[cell_i, cell_j]
            self.J = J
            self.c_types = c_types
            self.c_types_all = c_types_all



        def get_vertex(self):
            """
            Get vertex locations, given cell centroid positions and triangulation. I.e. calculate the circumcentres of
            each triangle
            :return V: Vertex coordinates (nv x 2)
            """
            V = circumcenter(self.Cents)
            return V

        def get_vertex_periodic(self):
            """
            Get vertex locations, given cell centroid positions and triangulation. I.e. calculate the circumcentres of
            each triangle

            :return V: Vertex coordinates (nv x 2)
            """
            V = circumcenter_periodic(self.Cents,self.L)
            return V

        def assign_vertices(self):
            """
            Generate the CV_matrix, an (nc x nv x 3) array, considering the relationship between cells and vertices/triangulation.
            Essentially an array expression of the triangulation

            Uses the stored self.tris, the (nv x 3) array denoting the triangulation.

            :return self.CV_matrix: array representation of the triangulation (nc x nv x 3)
            """
            CV_matrix = np.zeros((self.n_c, self.n_v, 3))
            for i in range(3):
               CV_matrix[self.tris[:, i], np.arange(self.n_v), i] = 1
            self.CV_matrix = CV_matrix
            return self.CV_matrix



        def set_t_span(self,dt,tfin):
            """
            Set the temporal running parameters

            :param dt: Time-step (np.float32)
            :param tfin: Final time-step (np.float32)
            :return self.t_span: Vector of times considered in the simulation (nt x 1)
            """
            self.dt, self.tfin = dt,tfin
            self.t_span = np.arange(0,tfin,dt)
            return self.t_span

        def check_forces(self,x,F):
            """
            Plot the forces (quiver) on each cell (voronoi)

            To be used as a quick check.

            :param x: Cell coordinates (nc x 2)
            :param F: Forces on each cell (nc x 2)
            """
            Vor = Voronoi(x)
            fig, ax = plt.subplots()
            ax.set(aspect=1)
            voronoi_plot_2d(Vor, ax=ax)
            ax.scatter(x[:, 0], x[:, 1])
            ax.quiver(x[:, 0], x[:, 1], F[:, 0], F[:, 1])
            fig.show()

        def voronoi_finite_polygons_2d(self, vor, radius=None):
            """
            Reconstruct infinite voronoi regions in a 2D diagram to finite
            regions.

            Parameters
            ----------
            vor : Voronoi
                Input diagram
            radius : float, optional
                Distance to 'points at infinity'.

            Returns
            -------
            regions : list of tuples
                Indices of vertices in each revised Voronoi regions.
            vertices : list of tuples
                Coordinates for revised Voronoi vertices. Same as coordinates
                of input vertices, with 'points at infinity' appended to the
                end.

            """

            if vor.points.shape[1] != 2:
                raise ValueError("Requires 2D input")

            new_regions = []
            new_vertices = vor.vertices.tolist()

            center = vor.points.mean(axis=0)
            if radius is None:
                radius = vor.points.ptp().max()

            # Construct a map containing all ridges for a given point
            all_ridges = {}
            for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
                all_ridges.setdefault(p1, []).append((p2, v1, v2))
                all_ridges.setdefault(p2, []).append((p1, v1, v2))

            # Reconstruct infinite regions
            for p1, region in enumerate(vor.point_region):
                vertices = vor.regions[region]

                if all(v >= 0 for v in vertices):
                    # finite region
                    new_regions.append(vertices)
                    continue

                # reconstruct a non-finite region
                ridges = all_ridges[p1]
                new_region = [v for v in vertices if v >= 0]

                for p2, v1, v2 in ridges:
                    if v2 < 0:
                        v1, v2 = v2, v1
                    if v1 >= 0:
                        # finite ridge: already in the region
                        continue

                    # Compute the missing endpoint of an infinite ridge

                    t = vor.points[p2] - vor.points[p1]  # tangent
                    t /= np.linalg.norm(t)
                    n = np.array([-t[1], t[0]])  # normal

                    midpoint = vor.points[[p1, p2]].mean(axis=0)
                    direction = np.sign(np.dot(midpoint - center, n)) * n
                    far_point = vor.vertices[v2] + direction * radius

                    new_region.append(len(new_vertices))
                    new_vertices.append(far_point.tolist())

                # sort region counterclockwise
                vs = np.asarray([new_vertices[v] for v in new_region])
                c = vs.mean(axis=0)
                angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
                new_region = np.array(new_region)[np.argsort(angles)]

                # finish
                new_regions.append(new_region.tolist())

            return new_regions, np.asarray(new_vertices)


        def plot_vor(self,x,ax,tri=False):
            """
            Plot the Voronoi.

            Takes in a set of cell locs (x), tiles these 9-fold, plots the full voronoi, then crops to the field-of-view

            :param x: Cell locations (nc x 2)
            :param ax: matplotlib axis
            """

            L = self.L
            grid_x, grid_y = np.mgrid[-1:2, -1:2]
            grid_x[0, 0], grid_x[1, 1] = grid_x[1, 1], grid_x[0, 0]
            grid_y[0, 0], grid_y[1, 1] = grid_y[1, 1], grid_y[0, 0]
            y = np.vstack([x + np.array([i * L, j * L]) for i, j in np.array([grid_x.ravel(), grid_y.ravel()]).T])

            c_types_print = np.tile(self.c_types,9)
            bleed = 0.1
            c_types_print = c_types_print[(y<L*(1+bleed)).all(axis=1)+(y>-L*bleed).all(axis=1)]
            y = y[(y<L*(1+bleed)).all(axis=1)+(y>-L*bleed).all(axis=1)]
            regions, vertices = self.voronoi_finite_polygons_2d(Voronoi(y))


            ax.set(aspect=1,xlim=(0,self.L),ylim=(0,self.L))
            if type(self.c_types) is list:
                # ax.scatter(x[:, 0], x[:, 1],color="grey",zorder=1000)
                for region in regions:
                    polygon = vertices[region]
                    plt.fill(*zip(*polygon), alpha=0.4, color="grey")

            else:
                cols = self.cols
                if self.plot_scatter is True:
                    for j,i in enumerate(np.unique(self.c_types)):
                        ax.scatter(x[self.c_types==i, 0], x[self.c_types==i, 1],color=cols[i],zorder=1000)
                patches = []
                for i, region in enumerate(regions):
                    patches.append(Polygon(vertices[region], True,facecolor=cols[c_types_print[i]],ec=(1,1,1,1)))

                p = PatchCollection(patches, match_original=True)
                # p.set_array(c_types_print)
                ax.add_collection(p)
            if tri is not False:
                for TRI in tri:
                    for j in range(3):
                        a, b = TRI[j], TRI[np.mod(j + 1, 3)]
                        if (a >= 0) and (b >= 0):
                            X = np.stack((x[a], x[b])).T
                            ax.plot(X[0], X[1], color="black")




        def plot_vor_boundary(self,x,ax,tri=False):
            """
            Plot the Voronoi.

            Takes in a set of cell locs (x), tiles these 9-fold, plots the full voronoi, then crops to the field-of-view

            :param x: Cell locations (nc x 2)
            :param ax: matplotlib axis
            :param tri: Is either a (n_v x 3) np.ndarray of dtype **np.int64** defining the triangulation.
                Or **False** where the triangulation is not plotted
            """

            x = x[~np.isnan(x[:,0])]
            c_types_print = np.ones(x.shape[0],dtype=np.int32)*-1
            c_types_print[:self.n_C] = self.c_types
            regions, vertices = self.voronoi_finite_polygons_2d(Voronoi(x))


            ax.set(aspect=1,xlim=(0,self.L),ylim=(0,self.L))
            if type(self.c_types) is list:
                # ax.scatter(x[:, 0], x[:, 1],color="grey",zorder=1000)
                for region in regions:
                    polygon = vertices[region]
                    plt.fill(*zip(*polygon), alpha=0.4, color="grey")

            else:
                cols = self.cols
                patches = []
                if self.plot_scatter is True:
                    ax.scatter(x[:self.n_C, 0], x[:self.n_C, 1], color="black", zorder=1000)
                    ax.scatter(x[self.n_C:, 0], x[self.n_C:, 1], color="grey", zorder=1000)

                for i, region in enumerate(regions):
                    patches.append(Polygon(vertices[region], True,facecolor=cols[c_types_print[i]],edgecolor="white",alpha=0.5))

                p = PatchCollection(patches, match_original=True)
                # p.set_array(c_types_print)
                ax.add_collection(p)
            if tri is not False:
                for TRI in tri:
                    for j in range(3):
                        a, b = TRI[j], TRI[np.mod(j + 1, 3)]
                        if (a >= 0) and (b >= 0):
                            X = np.stack((x[a], x[b])).T
                            ax.plot(X[0], X[1], color="black")


        def plot_vor_colored(self,x,ax,cmap):
            """
            Plot the Voronoi.

            Takes in a set of cell locs (x), tiles these 9-fold, plots the full voronoi, then crops to the field-of-view

            :param x: Cell locations (nc x 2)
            :param ax: matplotlib axis
            """

            L = self.L
            grid_x, grid_y = np.mgrid[-1:2, -1:2]
            grid_x[0, 0], grid_x[1, 1] = grid_x[1, 1], grid_x[0, 0]
            grid_y[0, 0], grid_y[1, 1] = grid_y[1, 1], grid_y[0, 0]
            y = np.vstack([x + np.array([i * L, j * L]) for i, j in np.array([grid_x.ravel(), grid_y.ravel()]).T])

            cmap_print = np.tile(cmap.T,9).T
            bleed = 0.1
            cmap_print = cmap_print[(y<L*(1+bleed)).all(axis=1)+(y>-L*bleed).all(axis=1)]
            y = y[(y<L*(1+bleed)).all(axis=1)+(y>-L*bleed).all(axis=1)]
            regions, vertices = self.voronoi_finite_polygons_2d(Voronoi(y))


            ax.set(aspect=1,xlim=(0,self.L),ylim=(0,self.L))
            if type(self.c_types) is list:
                # ax.scatter(x[:, 0], x[:, 1],color="grey",zorder=1000)
                for region in regions:
                    polygon = vertices[region]
                    plt.fill(*zip(*polygon), alpha=0.4, color="grey")

            else:
                patches = []
                for i, region in enumerate(regions):
                    patches.append(Polygon(vertices[region], True,facecolor=cmap_print[i],edgecolor="white",alpha=0.5))

                p = PatchCollection(patches, match_original=True)
                # p.set_array(c_types_print)
                ax.add_collection(p)


        def generate_noise(self):
            if type(self.noise) is list:
                theta_noise = np.cumsum(np.random.normal(0, np.sqrt(2 * self.Dr * self.dt), (self.n_t, self.n_c)), axis=0) + np.random.uniform(0,np.pi*2,self.n_c)
                self.noise = np.dstack((np.cos(theta_noise), np.sin(theta_noise)))
                if self.cell_movement_mask is not None:
                    self.noise[:,~self.cell_movement_mask] = self.noise[:,~self.cell_movement_mask]*0
                if self.no_noise_time is not None:
                    self.noise[:self.no_noise_time] = 0*self.noise[:self.no_noise_time]

        def generate_noise_boundary(self):
            n_c_extra = int(self.n_c*self.b_extra)
            theta_noise = np.cumsum(np.random.normal(0, np.sqrt(2 * self.Dr * self.dt), (self.n_t, self.n_C)), axis=0)
            noise_cells = np.dstack((np.cos(theta_noise), np.sin(theta_noise)))
            noise = np.zeros((self.n_t,n_c_extra,2))
            noise[:,:self.n_C]= noise_cells
            self.noise = noise

        def remove_repeats(self,tri,n_c):
            """
            For a given triangulation (nv x 3), remove repeated entries (i.e. rows)

            The triangulation is first re-ordered, such that the first cell id referenced is the smallest. Achieved via
            the function order_tris. (This preserves the internal order -- i.e. CCW)

            Then remove repeated rows via lexsort.

            NB: order of vertices changes via the conventions of lexsort

            Inspired by...
            https://stackoverflow.com/questions/31097247/remove-duplicate-rows-of-a-numpy-array

            :param tri: (nv x 3) matrix, the triangulation
            :return: triangulation minus the repeated entries (nv* x 3) (where nv* is the new # vertices).
            """
            tri = order_tris(np.mod(tri,n_c))
            sorted_tri = tri[np.lexsort(tri.T), :]
            row_mask = np.append([True], np.any(np.diff(sorted_tri, axis=0), 1))
            return sorted_tri[row_mask]

        def reset_k2s(self):
            self.k2s = get_k2_boundary(self.tris, self.v_neighbours).ravel()
            self.v_neighbours_flat = self.v_neighbours.ravel()
            self.b_neighbour_mask = (self.k2s > 0) * (self.v_neighbours_flat > 0)

        def triangulate(self,x,recalc_angles=False):
            self.Angles = tri_angles(x, self.tris)

            if type(self.k2s) is list:
                self._triangulate(x)
                if recalc_angles is True:
                    self.Angles = tri_angles(x, self.tris)
            elif not ((self.Angles[self.v_neighbours_flat, self.k2s] + self.Angles.ravel()) < np.pi)[self.b_neighbour_mask].all():
                self._triangulate(x)
                if recalc_angles is True:
                    self.Angles = tri_angles(x, self.tris)
            else:
                self.Cents = x[self.tris]
                self.vs = self.get_vertex()
                self.neighbours = self.vs[self.v_neighbours]


        def _triangulate(self,x):
            """

            Calculates the triangulation on the set of points x.

            Stores:
                self.n_v = number of vertices (int32)
                self.tris = triangulation of the vertices (nv x 3) matrix.
                    Cells are stored in CCW order. As a convention, the first entry has the smallest cell id
                    (Which entry comes first is, in and of itself, arbitrary, but is utilised elsewhere)
                self.vs = coordinates of each vertex; (nv x 2) matrix
                self.v_neighbours = vertex ids (i.e. rows of self.vs) corresponding to the 3 neighbours of a given vertex (nv x 3).
                    In CCW order, where vertex i {i=0..2} is opposite cell i in the corresponding row of self.tris
                self.neighbours = coordinates of each neighbouring vertex (nv x 3 x 2) matrix

            :param x: (nc x 2) matrix with the coordinates of each cell
            """

            t = tr.triangulate({"vertices": x},"-n")
            tri = t["triangles"]
            neighbours = t["neighbors"]

            b_cells = np.zeros(self.n_c)
            b_cells[self.n_C:] = 1

            three_b_cell_mask = b_cells[tri].sum(axis=1)==3
            tri = tri[~three_b_cell_mask]

            neigh_map = np.cumsum(~three_b_cell_mask)-1
            neigh_map[three_b_cell_mask] = -1
            neigh_map = np.concatenate((neigh_map,[-1]))

            neighbours = neighbours[~three_b_cell_mask]
            neighbours = neigh_map[neighbours]

            #6. Store outputs
            self.tris = tri
            self.n_v = tri.shape[0]
            self.Cents = x[self.tris]
            self.vs = self.get_vertex()


            #7. Manually calculate the neighbours. See doc_string for conventions.
            self.v_neighbours = neighbours
            self.neighbours = self.vs[neighbours]
            self.neighbours[neighbours == -1] = np.nan

            self.reset_k2s()


        def triangulate_periodic(self,x):
            self.Angles = tri_angles_periodic(x, self.tris, self.L)
            if type(self.k2s) is list:
                self._triangulate_periodic(x)
                self.k2s = get_k2(self.tris, self.v_neighbours)
            else:
                if (self.k2s>=3).sum()!=0:
                    self._triangulate_periodic(x)
                    self.k2s = get_k2(self.tris, self.v_neighbours)
                else:
                    mask = ((self.Angles[self.v_neighbours, self.k2s] + self.Angles) < np.pi)
                    if not mask.all():
                        self.equiangulate(x,mask)
                        # self._triangulate_periodic(x)

                        # self.k2s = get_k2(self.tris, self.v_neighbours)
                    else:
                        self.Cents = x[self.tris]
                        self.vs = self.get_vertex_periodic()
                        self.neighbours = self.vs[self.v_neighbours]

        def equiangulate(self,x,mask):
            """

            Fill this in properly later ...

            Consider the sum of the angles opposite every interface. If this is >180, then equiangulate.

            mask defines the cells/angles for which the sum with a neighbouring cell/angle is >180. These come in pairs

            Equiangulation works by looping through the following, until there exist no such pairs:
                1. Pick an edge for which the angles > 180. This is defined by "chosen_cell" and "chosen_opposite_cell", which are actually triangles.
                2. Replace the triangle entries for each of these triangles, such that the edge is swapped from the four cells
                3. Recompute the neighbours, but only for these two triangles, and their surrounding (4) neighbours (=6)
                4. Recalculate the angles and the mask and repeat.

            Notes:
                -- One worry is that equiangulation fails. May be important in the future to include a fail-safe back up of recomputation.
                -- Would be good to jit this function

            :param x:
            :param mask:
            :return:
            """

            timeout = 100
            k = 0
            while (not mask.all())and(k<timeout):

                changed_tris,j = np.nonzero(~mask)
                chosen_cell = changed_tris[0]
                cell_mask = np.zeros(3,dtype=np.bool)
                cell_mask[j[0]] = True
                chosen_opposite_cell = self.v_neighbours[chosen_cell,cell_mask][0]


                cells = np.roll(self.tris[chosen_cell],-j[0])
                opposite_cells = self.tris[chosen_opposite_cell]
                opposite_cells = np.roll(opposite_cells, - self.k2s[chosen_cell,cell_mask])


                self.tris[chosen_cell] = cells[0], opposite_cells[0],cells[2]
                self.tris[chosen_opposite_cell] = opposite_cells[0],cells[0], opposite_cells[2]

                self.Angles[[chosen_cell,chosen_opposite_cell]] = tri_angles_periodic(x, self.tris[[chosen_cell,chosen_opposite_cell]], self.L)
                # self.Angles = tri_angles_periodic(x,self.tris,self.L)
                self.Cents = x[self.tris]
                self.vs = self.get_vertex_periodic()


                modify_neighbours = np.concatenate([self.v_neighbours[chosen_cell],self.v_neighbours[chosen_opposite_cell]])
                modify_neighbours.sort()
                self.v_neighbours[modify_neighbours] = -1


                n_neigh = get_neighbours(self.tris,self.v_neighbours,Range = modify_neighbours)
                self.v_neighbours = n_neigh
                self.neighbours = self.vs[n_neigh]

                self.k2s = get_k2(self.tris, self.v_neighbours)
                if (self.k2s>=3).sum()!=0:
                    self._triangulate_periodic(x)
                    self.k2s = get_k2(self.tris, self.v_neighbours)
                    mask[:] = True
                else:
                    mask = ((self.Angles[self.v_neighbours, self.k2s] + self.Angles) < np.pi)
                k+=1
            if k == timeout:
                self._triangulate_periodic(x)
                self.k2s = get_k2(self.tris, self.v_neighbours)


        def _triangulate_periodic(self,x):
            """
            Calculates the periodic triangulation on the set of points x.

            Stores:
                self.n_v = number of vertices (int32)
                self.tris = triangulation of the vertices (nv x 3) matrix.
                    Cells are stored in CCW order. As a convention, the first entry has the smallest cell id
                    (Which entry comes first is, in and of itself, arbitrary, but is utilised elsewhere)
                self.vs = coordinates of each vertex; (nv x 2) matrix
                self.v_neighbours = vertex ids (i.e. rows of self.vs) corresponding to the 3 neighbours of a given vertex (nv x 3).
                    In CCW order, where vertex i {i=0..2} is opposite cell i in the corresponding row of self.tris
                self.neighbours = coordinates of each neighbouring vertex (nv x 3 x 2) matrix

            :param x: (nc x 2) matrix with the coordinates of each cell
            """

            #1. Tile cell positions 9-fold to perform the periodic triangulation
            #   Calculates y from x. y is (9nc x 2) matrix, where the first (nc x 2) are the "true" cell positions,
            #   and the rest are translations
            y = make_y(x,self.L*self.grid_xy)


            #2. Perform the triangulation on y
            #   The **triangle** package (tr) returns a dictionary, containing the triangulation.
            #   This triangulation is extracted and saved as tri
            t = tr.triangulate({"vertices": y})
            tri = t["triangles"]

            # Del = Delaunay(y)
            # tri = Del.simplices
            n_c = x.shape[0]

            #3. Find triangles with **at least one** cell within the "true" frame (i.e. with **at least one** "normal cell")
            #   (Ignore entries with -1, a quirk of the **triangle** package, which denotes boundary triangles
            #   Generate a mask -- one_in -- that considers such triangles
            #   Save the new triangulation by applying the mask -- new_tri
            tri = tri[(tri != -1).all(axis=1)]
            one_in = (tri<n_c).any(axis=1)
            new_tri = tri[one_in]

            #4. Remove repeats in new_tri
            #   new_tri contains repeats of the same cells, i.e. in cases where triangles straddle a boundary
            #   Use remove_repeats function to remove these. Repeats are flagged up as entries with the same trio of
            #   cell ids, which are transformed by the mod function to account for periodicity. See function for more details
            n_tri = self.remove_repeats(new_tri,n_c)

            # tri_same = (self.tris == n_tri).all()

            #6. Store outputs
            self.n_v = n_tri.shape[0]
            self.tris = n_tri
            self.Cents = x[self.tris]
            self.vs = self.get_vertex_periodic()

            #7. Manually calculate the neighbours. See doc_string for conventions.
            n_neigh = get_neighbours(n_tri)
            self.v_neighbours = n_neigh
            self.neighbours = self.vs[n_neigh]

        def check_boundary(self,x):
            """
            For a non-periodic simulation using boundary particles, dynamically update the number/position of particles
            to preserve cell shape continuity while also minimizing the number of boundary particles.

            Provided the cell aggregate is completely contiguous, then **check_boundary** ensures boundary particles
            form a single ring (i.e. where, within the set of triangles featuring two boundary cells, each boundary cell
            is represented in two such triangles)

            Performs two steps.

            1. Add extra boundary cells. Calculate the angle that "real" cells make with pairs of boundary cells (within triangulation).
                Reflect the "real" cell over the line made by the pair of boundary cells if this angle > 90 degs

            2. Remove extra cells.
                Remove boundary cells that are not connected to at least one "real" cell.

            :param x: Cell centroids (n_c x 2)
            :return: Updated cell centroids (n_c x 2)
            """
            b_cells = np.zeros(self.n_c)
            b_cells[self.n_C:] = 1
            vBC = b_cells[self.tris]
            considered_triangles = vBC.sum(axis=1) == 2
            add_extra = ((self.Angles*(1-vBC)>np.pi/2).T*considered_triangles.T).T
            if add_extra.any():
                I,J = np.nonzero(add_extra)
                for k,i in enumerate(I):
                    j = J[k]
                    xs = x[self.tris[i]]
                    re = xs[np.mod(j-1,3)] - xs[np.mod(j+1,3)]
                    re = re/np.linalg.norm(re)
                    re = np.array([re[1],-re[0]])
                    rpe = xs[j]
                    x_new = 2*np.dot(xs[np.mod(j-1,3)]-rpe,re)*re + rpe
                    x = np.vstack((x,x_new))
                self.n_c = x.shape[0]
                self._triangulate(x)
                self.assign_vertices()

            C = get_C_boundary(self.n_c,self.CV_matrix)
            #
            # #Remove extra cells
            # keep_mask = C[self.n_C:, :self.n_C].sum(axis=1)>0 #I'm assuming this is the same thing. This removes all boundary centroids that are not connected to at least one real centroid.
            # if keep_mask.any():
            #     c_keep = np.nonzero(keep_mask)[0]
            #     x = np.concatenate((x[:self.n_C],x[c_keep + self.n_C]))
            #     self.n_c = x.shape[0]
            #     self._triangulate(x)
            #     self.assign_vertices()
            #

            #Remove all boundary particles not connected to exactly two other boundary particles
            remove_mask = C[self.n_C:, self.n_C:].sum(axis=1)!=2
            if remove_mask.any():
                c_keep = np.nonzero(~remove_mask)[0]
                x = np.concatenate((x[:self.n_C],x[c_keep + self.n_C]))
                self.n_c = x.shape[0]
                self._triangulate(x)
                self.assign_vertices()
                self.Angles = tri_angles(x, self.tris)
            #
            # remove_mask = C[self.n_C:, self.n_C:].sum(axis=1)==0
            # if remove_mask.any():
            #     c_keep = np.nonzero(~remove_mask)[0]
            #     x = np.concatenate((x[:self.n_C],x[c_keep + self.n_C]))
            #     self.n_c = x.shape[0]
            #     self._triangulate(x)
            #     self.assign_vertices()
            #     self.Angles = tri_angles(x, self.tris)


            return x

        #
        # def check_boundary(self,x):
        #       ##ARCHIVE
        #     """
        #     For a non-periodic simulation using boundary particles, dynamically update the number/position of particles
        #     to preserve cell shape continuity while also minimizing the number of boundary particles.
        #
        #     Provided the cell aggregate is completely contiguous, then **check_boundary** ensures boundary particles
        #     form a single ring (i.e. where, within the set of triangles featuring two boundary cells, each boundary cell
        #     is represented in two such triangles)
        #
        #     Performs two steps.
        #
        #     1. Add extra boundary cells. Calculate the angle that "real" cells make with pairs of boundary cells (within triangulation).
        #         Reflect the "real" cell over the line made by the pair of boundary cells if this angle > 90 degs
        #
        #     2. Remove extra cells.
        #         Remove boundary cells that are not connected to at least one "real" cell.
        #
        #     :param x: Cell centroids (n_c x 2)
        #     :return: Updated cell centroids (n_c x 2)
        #     """
        #     b_cells = np.zeros(self.n_c)
        #     b_cells[self.n_C:] = 1
        #     vBC = b_cells[self.tris]
        #     considered_triangles = vBC.sum(axis=1) == 2
        #     add_extra = ((self.Angles*(1-vBC)>np.pi/2).T*considered_triangles.T).T
        #     if add_extra.any():
        #         I,J = np.nonzero(add_extra)
        #         for k,i in enumerate(I):
        #             j = J[k]
        #             xs = x[self.tris[i]]
        #             xs = np.roll(xs, -j, axis=0)
        #             perp = np.array([-xs[1,1]+xs[2,1],xs[1,0]-xs[2,0]])
        #             perp = perp/np.sqrt((perp**2).sum())
        #             d = np.cross(xs[2]-xs[0],xs[1]-xs[0])/np.sqrt(((xs[2]-xs[1])**2).sum())
        #             x_new = xs[0] -2*d*perp
        #             x = np.vstack((x,x_new))
        #         self.n_c = x.shape[0]
        #         self._triangulate(x)
        #         self.assign_vertices()
        #
        #     #Remove extra cells
        #     C = get_C(self.n_c,self.CV_matrix)
        #     # C = ((C + C.T)==1).astype(np.int32) #<-- should this be uncommented??
        #     keep_mask = C[self.n_C:, :self.n_C].sum(axis=1)>0 #I'm assuming this is the same thing. This removes all boundary centroids that are not connected to at least one real centroid.
        #     if keep_mask.any():
        #         c_keep = np.nonzero(keep_mask)[0]
        #         x = np.concatenate((x[:self.n_C],x[c_keep + self.n_C]))
        #         self.n_c = x.shape[0]
        #         self._triangulate(x)
        #         self.assign_vertices()
        #     return x

        def get_P(self,neighbours, vs):
            """
            Calculates perimeter of each cell

            :param neighbours: (nv x 3 x 2) matrix considering the coordinates of each neighbouring vertex of each vertex
            :param vs: (nv x 2) matrix considering coordinates of each vertex
            :return: self.P saves the areas of each cell
            """
            self.P = get_P(vs, neighbours, self.CV_matrix, self.n_c)
            return self.P

        def get_P_periodic(self,neighbours, vs):
            """
            Identical to **get_P** but accounts for periodic triangulation

            Calculates perimeter of each cell (considering periodic boundary conditions)

            :param neighbours: (nv x 3 x 2) matrix considering the coordinates of each neighbouring vertex of each vertex
            :param vs: (nv x 2) matrix considering coordinates of each vertex
            :return: self.P saves the areas of each cell
            """
            self.P = get_P_periodic(vs, neighbours, self.CV_matrix, self.L, self.n_c)
            return self.P


        def get_A(self,neighbours, vs):
            """
            Calculates area of each cell

            :param neighbours: (nv x 3 x 2) matrix considering the coordinates of each neighbouring vertex of each vertex
            :param vs: (nv x 2) matrix considering coordinates of each vertex
            :return: self.A saves the areas of each cell
            """
            self.A = get_A(vs, neighbours, self.CV_matrix, self.n_c)
            return self.A

        def get_A_periodic(self,neighbours, vs):
            """
            Identical to **get_A** but accounts for periodic triangulation.

            Calculates area of each cell (considering periodic boundary conditions)

            :param neighbours: (nv x 3 x 2) matrix considering the coordinates of each neighbouring vertex of each vertex
            :param vs: (nv x 2) matrix considering coordinates of each vertex
            :return: self.A saves the areas of each cell
            """
            self.A = get_A_periodic(vs, neighbours, self.Cents, self.CV_matrix, self.L, self.n_c)
            return self.A



        def get_F_periodic(self,neighbours,vs):
            """
            Calculate the forces acting on each cell via the SPV formalism.

            Detailed explanations of each stage are described in line, but overall the strategy leverages the chain-rule
            (vertex-wise) decomposition of the expression for forces acting on each cell. Using this, contributions from
            each vertex is calculated **on the triangulation**, without the need for explicitly calculating the voronoi
            polygons. Hugely improves efficiency

            :param neighbours: Positions of neighbouring vertices (n_v x 3 x 2)
            :param vs: Positions of vertices (n_v x 2)
            :return: F
            """
            J_CW = self.J[self.tris, roll_forward(self.tris)]
            J_CCW = self.J[self.tris, roll_reverse(self.tris)]
            F = get_F_periodic(vs, neighbours, self.tris, self.CV_matrix, self.n_v, self.n_c, self.L, J_CW, J_CCW, self.A, self.P, self.Cents, self.kappa_A, self.kappa_P, self.A0, self.P0)
            return F


        def get_F_periodic_param(self,neighbours,vs):
            """

            FIX
            Calculate the forces acting on each cell via the SPV formalism.

            Detailed explanations of each stage are described in line, but overall the strategy leverages the chain-rule
            (vertex-wise) decomposition of the expression for forces acting on each cell. Using this, contributions from
            each vertex is calculated **on the triangulation**, without the need for explicitly calculating the voronoi
            polygons. Hugely improves efficiency

            :param neighbours: Positions of neighbouring vertices (n_v x 3 x 2)
            :param vs: Positions of vertices (n_v x 2)
            :return: F
            """
            J_CW = self.J[self.tris, roll_forward(self.tris)]
            J_CCW = self.J[self.tris, roll_reverse(self.tris)]
            F = get_F_periodic_param(vs, neighbours, self.tris, self.CV_matrix, self.n_v, self.n_c, self.L, J_CW, J_CCW, self.A, self.P, self.Cents, self.kappa_A, self.kappa_P, self.A0, self.P0)
            return F


        def get_F(self,neighbours,vs):
            """
            Identical to **get_F_periodic** but instead accounts for boundaries and neglects periodic triangulation.

            Calculate the forces acting on each cell via the SPV formalism.

            Detailed explanations of each stage are described in line, but overall the strategy leverages the chain-rule
            (vertex-wise) decomposition of the expression for forces acting on each cell. Using this, contributions from
            each vertex is calculated **on the triangulation**, without the need for explicitly calculating the voronoi
            polygons. Hugely improves efficiency

            :param neighbours: Positions of neighbouring vertices (n_v x 3 x 2)
            :param vs: Positions of vertices (n_v x 2)
            :return: F
            """
            J = self.J_large[:self.n_c,:self.n_c]
            # J[:self.n_C,:self.n_C] = self.J
            # J[self.n_C:,self.n_C:] = 0
            J_CW = J[self.tris, roll_forward(self.tris)]
            J_CCW = J[self.tris, roll_reverse(self.tris)]
            F = get_F(vs, neighbours, self.tris, self.CV_matrix, self.n_v, self.n_c, self.L, J_CW, J_CCW, self.A, self.P, self.Cents, self.kappa_A, self.kappa_P, self.A0, self.P0,self.n_C,self.kappa_B,self.l_b0)
            return F

        def simulate(self,print_every=1000,variable_param=False,equiangulate=True):
            """
            Evolve the SPV.

            Stores:
                self.x_save = Cell centroids for each time-step (n_t x n_c x 2), where n_t is the number of time-steps
                self.tri_save = Triangulation for each time-step (n_t x n_v x 3)


            :param print_every: integer value to skip printing progress every "print_every" iterations.
            :param variable_param: Set this to True if kappa_A,kappa_P are vectors rather than single values
            :return: self.x_save
            """
            if variable_param is True:
                F_get = self.get_F_periodic_param
            else:
                F_get = self.get_F_periodic
            if equiangulate is True:
                triangulate = self.triangulate_periodic
            else:
                triangulate = self._triangulate_periodic
            n_t = self.t_span.size
            self.n_t = n_t
            x = self.x0.copy()
            self._triangulate_periodic(x)
            self.x = x.copy()
            self.x_save = np.zeros((n_t,self.n_c,2))
            self.tri_save = np.zeros((n_t,self.tris.shape[0],3),dtype=np.int32)
            self.generate_noise()
            for i in range(n_t):
                if i % print_every == 0:
                    print(i / n_t * 100, "%")
                triangulate(x)
                self.tri_save[i] = self.tris
                self.assign_vertices()
                self.get_A_periodic(self.neighbours,self.vs)
                self.get_P_periodic(self.neighbours,self.vs)
                F = F_get(self.neighbours,self.vs)
                F_soft = weak_repulsion(self.Cents,self.a,self.k, self.CV_matrix,self.n_c,self.L)
                x += self.dt*(F + F_soft + self.v0*self.noise[i])
                x = np.mod(x,self.L)
                self.x = x
                self.x_save[i] = x
            print("Simulation complete")
            return self.x_save

        def beta_t(self,t):
            return _beta_t(self.beta_max, self.beta_min,self.tau,t)


        def simulate_dynamic_beta(self,print_every=1000,variable_param=False,equiangulate=True):
            """
            Evolve the SPV.

            Stores:
                self.x_save = Cell centroids for each time-step (n_t x n_c x 2), where n_t is the number of time-steps
                self.tri_save = Triangulation for each time-step (n_t x n_v x 3)


            :param print_every: integer value to skip printing progress every "print_every" iterations.
            :param variable_param: Set this to True if kappa_A,kappa_P are vectors rather than single values
            :return: self.x_save
            """
            if variable_param is True:
                F_get = self.get_F_periodic_param
            else:
                F_get = self.get_F_periodic
            if equiangulate is True:
                triangulate = self.triangulate_periodic
            else:
                triangulate = self._triangulate_periodic
            self.J0 = self.J.copy()
            n_t = self.t_span.size
            self.n_t = n_t
            x = self.x0.copy()
            self._triangulate_periodic(x)
            self.x = x.copy()
            self.x_save = np.zeros((n_t,self.n_c,2))
            self.tri_save = np.zeros((n_t,self.tris.shape[0],3),dtype=np.int32)
            self.generate_noise()
            for i, t in enumerate(self.t_span):
                if i % print_every == 0:
                    print(i / n_t * 100, "%")
                triangulate(x)
                self.tri_save[i] = self.tris
                self.assign_vertices()
                self.get_A_periodic(self.neighbours,self.vs)
                self.get_P_periodic(self.neighbours,self.vs)
                self.J = self.J0*self.beta_t(t)
                F = F_get(self.neighbours,self.vs)

                F_soft = weak_repulsion(self.Cents,self.a,self.k, self.CV_matrix,self.n_c,self.L)
                x += self.dt*(F + F_soft + self.v0*self.noise[i])
                x = np.mod(x,self.L)
                self.x = x
                self.x_save[i] = x
            print("Simulation complete")
            return self.x_save


        def get_t1_forward_cells(self):
            tc_type = self.c_types[self.tris]
            two_type1_tris = tc_type.sum(axis=1) == 2
            Ls = np.nonzero(two_type1_tris)[0]
            Is, Js = np.nonzero(tc_type[two_type1_tris] == 1)
            type1_b_cells = self.tris[Ls[Is], Js]
            type0_cell = tc_type[two_type1_tris] == 0
            Is, Js = np.nonzero(type0_cell)
            # near_boundary_cells = np.ones_like(Is,dtype=np.int64)*-1
            self.forward_cells = np.ones((Is.size), dtype=np.int64) * -1
            k = 0
            for i, j in zip(Is, Js):
                m = Ls[i]
                opposite_tri = self.v_neighbours[m, j]
                opposite_cell = np.roll(self.tris[opposite_tri], -self.k2s[m, j])[0]
                if (self.c_types[opposite_cell] == 1) * (opposite_cell not in type1_b_cells):
                    # near_boundary_cells[k] = opposite_cell
                    ordered_tri = np.roll(self.tris[m], -j)
                    self.forward_cells[k] = opposite_cell
                    k += 1
            self.forward_cells = self.forward_cells[:k - 1]
            self.n_t1_cells = self.forward_cells.shape[0]

        def get_t1_reverse_cells(self):
            tc_type = self.c_types[self.tris]
            two_type1_tris = tc_type.sum(axis=1) == 2
            Ls = np.nonzero(two_type1_tris)[0]
            Is, Js = np.nonzero(tc_type[two_type1_tris] == 1)
            self.reverse_cells = np.unique(self.tris[Ls[Is], Js])
            for i, cll in enumerate(self.reverse_cells):
                neighs = np.unique(self.tris[np.nonzero(self.tris==cll)[0]])
                n_a_neighs = (1-self.c_types[neighs]).sum()
                if n_a_neighs!=2:
                    self.reverse_cells[i] = -1

            self.reverse_cells = self.reverse_cells[self.reverse_cells!=-1]
            self.n_t1_cells = self.reverse_cells.shape[0]
            #
            # tc_type = self.c_types[self.tris]
            # two_type0_tris = tc_type.sum(axis=1) == 1
            # Ls = np.nonzero(two_type0_tris)[0]
            # Is, Js = np.nonzero(tc_type[two_type0_tris] == 0)
            # self.reverse_cells = np.unique(self.tris[Ls[Is], Js])
            # self.n_t1_cells = self.reverse_cells.shape[0]




        def initialize_t1(self,i,t1_type = "forward"):
            self.t1_type = t1_type
            self._triangulate_periodic(self.x)
            self.triangulate_periodic(self.x)
            self.assign_vertices()
            if t1_type == "forward":
                self.get_t1_forward_cells()
                self.mobile_i = self.forward_cells[i]
            if t1_type == "reverse":
                self.get_t1_reverse_cells()
                self.mobile_i = self.reverse_cells[i]
            self.c_types[self.mobile_i] = 0


        def simulate_t1(self,print_every=1000,variable_param=False,equiangulate=True):
            """
            Evolve the SPV.

            Stores:
                self.x_save = Cell centroids for each time-step (n_t x n_c x 2), where n_t is the number of time-steps
                self.tri_save = Triangulation for each time-step (n_t x n_v x 3)


            :param print_every: integer value to skip printing progress every "print_every" iterations.
            :param variable_param: Set this to True if kappa_A,kappa_P are vectors rather than single values
            :return: self.x_save
            """
            if variable_param is True:
                F_get = self.get_F_periodic_param
            else:
                F_get = self.get_F_periodic
            if equiangulate is True:
                triangulate = self.triangulate_periodic
            else:
                triangulate = self._triangulate_periodic

            n_t = self.t_span.size
            self.n_t = n_t
            x = self.x0.copy()
            self._triangulate_periodic(x)
            self.x = x.copy()
            self.x_save = np.zeros((n_t,self.n_c,2))
            self.tri_save = np.zeros((n_t,self.tris.shape[0],3),dtype=np.int32)
            self.generate_noise()
            for i,t in enumerate(self.t_span):
                if i % print_every == 0:
                    print(i / n_t * 100, "%")
                triangulate(x)
                self.tri_save[i] = self.tris
                self.assign_vertices()
                self.get_A_periodic(self.neighbours,self.vs)
                self.get_P_periodic(self.neighbours,self.vs)
                F = F_get(self.neighbours,self.vs)
                F_soft = weak_repulsion(self.Cents,self.a,self.k, self.CV_matrix,self.n_c,self.L)
                x += self.dt*(F + F_soft)
                x += self.dt*(self.v0*0.01*self.noise[i])
                if (t > self.no_movement_time):
                    tri_i,tri_k,complete = get_t1_dir(self.tris,self.c_types,self.v_neighbours,self.neighbours,self.L,self.vs,self.mobile_i,self.n_c,self.CV_matrix,t1_type=self.t1_type)
                    if tri_i != -1:
                        tri_j = self.v_neighbours[tri_i, tri_k]
                        M_comp = np.zeros((self.n_c, 2))
                        M_comp[self.tris[tri_i]] += get_M(tri_i, tri_j, self.vs, self.x, self.tris, self.L).T
                        M_comp[self.tris[tri_j]] += get_M(tri_j, tri_i, self.vs, self.x, self.tris, self.L).T
                        M_comp = (M_comp.T / np.linalg.norm(M_comp, axis=1)).T
                        M_comp[np.isnan(M_comp)] = 0
                        x += M_comp*self.dt*self.v0
                    if complete * (self.t1_time is False):
                        self.t1_time = t
                x = np.mod(x,self.L)
                self.x = x
                self.x_save[i] = x

            print("Simulation complete")
            return self.x_save




        def profile_function(self,function):
            lp = LineProfiler()
            lp_wrapper = lp(function)
            lp_wrapper()
            lp.print_stats()
            return

        def simulate_boundary(self,print_every=1000,do_F_bound=True):
            """
            Evolve the SPV but using boundaries.

            Stores:
                self.x_save = Cell centroids for each time-step (n_t x n_c x 2), where n_t is the number of time-steps
                self.tri_save = Triangulation for each time-step (n_t x n_v x 3)


            :param print_every: integer value to skip printing progress every "print_every" iterations.
            :param b_extra: Set this to >1. Defines the size of x_save to account for variable numbers of (boundary) cells.
                if b_extra = 2, then x_save.shape[1] = 2*n_c (at t=0)
            :return: self.x_save
            """
            n_t = self.t_span.size
            self.n_t = n_t
            x = self.x0.copy()
            self._triangulate(x)
            self.assign_vertices()
            x = self.check_boundary(x)
            self.x = x.copy()
            self.x_save = np.ones((n_t,int(self.n_c*self.b_extra),2))*np.nan
            self.tri_save = -np.ones((n_t,int(self.tris.shape[0]*self.b_extra),3),dtype=np.int32)
            self.generate_noise_boundary()
            if do_F_bound is True:
                for i in range(n_t):
                    if i % print_every == 0:
                        print(i / n_t * 100, "%")
                    self.triangulate(x,recalc_angles=True)
                    self.assign_vertices()
                    x = self.check_boundary(x)
                    self.tri_save[i,:self.tris.shape[0]] = self.tris
                    self.get_A(self.neighbours,self.vs)
                    self.get_P(self.neighbours,self.vs)
                    F = self.get_F(self.neighbours,self.vs)
                    # F_bend = get_F_bend(self.n_c, self.CV_matrix, self.n_C, x, self.zeta)
                    F_soft = weak_repulsion_boundary(self.Cents,self.a,self.k, self.CV_matrix,self.n_c,self.n_C)
                    F_bound = boundary_tension(self.Gamma_bound,self.n_C,self.n_c,self.Cents,self.CV_matrix)
                    x += self.dt*(F + F_soft + self.v0*self.noise[i,:x.shape[0]] + F_bound)
                                  # + F_bend + F_bound

                    self.x = x
                    self.x_save[i,:x.shape[0]] = x
            else:
                for i in range(n_t):
                    if i % print_every == 0:
                        print(i / n_t * 100, "%")
                    self.triangulate(x, recalc_angles=True)
                    self.assign_vertices()
                    x = self.check_boundary(x)
                    self.tri_save[i, :self.tris.shape[0]] = self.tris
                    self.get_A(self.neighbours, self.vs)
                    self.get_P(self.neighbours, self.vs)
                    F = self.get_F(self.neighbours, self.vs)
                    F_soft = weak_repulsion_boundary(self.Cents, self.a, self.k, self.CV_matrix, self.n_c, self.n_C)
                    x += self.dt * (F + F_soft + self.v0 * self.noise[i, :x.shape[0]])

                    self.x = x
                    self.x_save[i, :x.shape[0]] = x
            print("Simulation complete")
            return self.x_save




        def simulate_GRN(self,print_every=1000):
            """
            Evolve the SPV.

            Stores:
                self.x_save = Cell centroids for each time-step (n_t x n_c x 2), where n_t is the number of time-steps
                self.tri_save = Triangulation for each time-step (n_t x n_v x 3)


            :param print_every: integer value to skip printing progress every "print_every" iterations.
            :return: self.x_save
            """
            n_t = self.t_span.size
            self.n_t = n_t
            x = self.x0.copy()
            self._triangulate_periodic(x)
            self.x = x.copy()
            self.x_save = np.zeros((n_t,self.n_c,2))
            self.E_save = np.zeros((n_t,self.n_c))
            E = self.Sender * self.sender_val
            self.E_save[0] = E
            i_past = int(self.dT / self.dt)
            print(i_past)
            self.tri_save = np.zeros((n_t,self.tris.shape[0],3),dtype=np.int32)
            self.generate_noise()

            for i in range(n_t):
                if i % print_every == 0:
                    print(i / n_t * 100, "%")
                self.triangulate_periodic(x)
                self.tri_save[i] = self.tris
                self.assign_vertices()
                self.get_A_periodic(self.neighbours,self.vs)
                self.get_P_periodic(self.neighbours,self.vs)
                F = self.get_F_periodic(self.neighbours,self.vs)
                F_soft = weak_repulsion(self.Cents,self.a,self.k, self.CV_matrix,self.n_c,self.L)
                x += self.dt*(F + F_soft + self.v0*self.noise[i])
                x = np.mod(x,self.L)
                self.x = x
                self.x_save[i] = x


                ##Simulate the GRN
                if i <= i_past:
                    E = self.GRN_step(self.E_save[0],E)*(~self.Sender*self.sender_val) + self.Sender * self.sender_val
                else:
                    E = self.GRN_step(self.E_save[i-i_past],E)*(~self.Sender*self.sender_val) + self.Sender * self.sender_val
                self.E_save[i] = E
            print("Simulation complete")
            return self.x_save


        def GRN_step(self,A_in,A):
            """
            Perform one time-step of the GRN simulation

            Ebar_i = SUM_{j in adjoining edges of cell i} l_ij A_j/P_j
            dA_i/dt = 1/tau*(E_bar_i^p/(E_bar_i^p + B0^p + (delT*A)^p))

            :param A: vector of n_c, defining value of the variable A
            :return: Output after one time-step
            """
            l_matrix = get_l_interface(self.n_v, self.n_c, self.neighbours, self.vs, self.CV_matrix, self.L)
            E_bar = l_matrix@(A_in/self.P)
            dtA = 1/self.tau * (self.leak +
                                self.alpha*(E_bar**self.p)/(E_bar**self.p + self.K**self.p + (self.delT*A_in)**self.p)
                                - A)
            A_out = A + self.dt*dtA
            return A_out

        def get_self_self(self):
            """
            Stores the percentage of self-self contacts (in self.self_self)
            """
            self.self_self = np.zeros(self.n_t)
            for i, tri in enumerate(self.tri_save):
                C_types = self.c_types[tri]
                C_self_self = (C_types-np.roll(C_types,1,axis=1))==0
                self.self_self[i] = np.sum(C_self_self)/C_self_self.size

        def get_self_self_by_cell(self):
            """
            Stores the percentage of self-self contacts (in self.self_self)
            """
            self.p_self_self = np.zeros(self.n_t)
            self.p_self_self_t = np.zeros((self.n_t,self.n_c))
            for i, tri in enumerate(self.tri_save):
                C_types = self.c_types[tri]
                C_self_self = (C_types-np.roll(C_types,1,axis=1))==0
                C_self_self = C_self_self*1.0 #+ np.roll(C_self_self,-1,axis=1)*1.0
                c_self_self = np.zeros(self.n_c)
                for j in range(3):
                    c_self_self += np.dot(self.CV_matrix[:,:,j],C_self_self[:,j])
                p_self_self = c_self_self/np.sum(self.CV_matrix,axis=(1,2))
                self.p_self_self_t[i] = p_self_self
                self.p_self_self[i] = p_self_self.mean()

        def get_self_self_interface(self,nT = 100,t_sel = None):
            """
            Calculates the fraction of a cell's perimeter that is a boundary with another cell type (on average)
            Saves this to self.self_self_interface

            Additionally calculates the total interface length for all cells, saved to self.total_interface_length

            :param nT: Number of time-steps to consider
            :return: self.self_self_interface and time-point selection
            """
            if t_sel is None:
                t_sel = np.linspace(0,self.n_t-1,nT).astype(int)
            else:
                nT = t_sel.size
            I,J = np.meshgrid(self.c_types,self.c_types,indexing="ij")
            SS = (I==J)*1.0

            self.self_self_interface = np.zeros((nT,self.n_c))
            self.total_interface_length = np.zeros(nT)
            for j, t in enumerate(t_sel):
                x = self.x_save[t]
                self._triangulate_periodic(x)
                self.assign_vertices()
                SS_CCW = SS[self.tris, roll_reverse(self.tris)]
                P = self.get_P_periodic(self.neighbours,self.vs)
                h_j = np.empty((self.n_v, 3, 2))
                for i in range(3):
                    h_j[:, i] = self.vs
                h_jm1 = np.dstack((roll_forward(self.neighbours[:, :, 0]), roll_forward(self.neighbours[:, :, 1])))

                l_jm1 = np.mod(h_j - h_jm1 + self.L / 2,self.L) - self.L / 2
                l_jm1_norm = np.sqrt(l_jm1[:, :, 0] ** 2 + l_jm1[:, :, 1] ** 2)

                self_self_length = SS_CCW * l_jm1_norm
                c_ss_length = np.zeros(self.n_c)
                for i in range(3):
                    c_ss_length += np.dot(self.CV_matrix[:,:,i],self_self_length[:,i])
                self.self_self_interface[j] = c_ss_length/P
                self.total_interface_length[j] = P.sum() - c_ss_length.sum()
            return self.self_self_interface, self.t_span[t_sel]

        def get_num_islands(self,nT=100):
            """
            Get the number of islands, defined as contiguous regions of a given cell type.

            Considers only cases where there are two cell types.

            A_islands is the number of islands of cell_type == 0, and B_islands for cell_type == 1

            :param nT: Number of time-steps to consider (evenly spaced)
            :return: A_islands, B_islands
            """
            t_sel = np.linspace(0,self.n_t-1,nT).astype(int)
            A_islands,B_islands = np.zeros(nT,dtype=np.int32),np.zeros(nT,dtype=np.int32)
            for i, t in enumerate(t_sel):
                tri = self.tri_save[t]
                Adj = np.zeros((self.n_c,self.n_c),dtype=np.float32)
                Adj[tri,np.roll(tri,-1,axis=1)] = 1
                AdjA = Adj[self.c_types==0][:, self.c_types==0]
                AdjB = Adj[self.c_types==1][:, self.c_types==1]
                A_islands[i],B_islands[i] = connected_components(csgraph=csr_matrix(AdjA), directed=False)[0],connected_components(csgraph=csr_matrix(AdjB), directed=False)[0]
            return A_islands, B_islands

        def get_num_boundaries(self, nT=100):
            """
            Get the number of islands, defined as contiguous regions of a given cell type.
            Considers only cases where there are two cell types.
            A_islands is the number of islands of cell_type == 0, and B_islands for cell_type == 1
            :param nT: Number of time-steps to consider (evenly spaced)
            :return: A_islands, B_islands
            """
            t_sel = np.linspace(0, self.n_t - 1, nT).astype(int)
            num_boundaries = np.zeros(nT, dtype=np.int32)
            for i, t in enumerate(t_sel):
                tri = self.tri_save[t]
                tri_types = self.c_types[tri]

                # Find heterotypic edges by comparing each cell with its CW neighbour in the triangulation
                het_neighbours = tri_types != np.roll(tri_types, 1, axis=1)
                v_neighbours = get_neighbours(tri)

                # For each triangle (i.e. vertex), find the corresponding neighbouring vertex that makes up the above heterotypic edges
                het_v_neighbour_mask = np.roll(het_neighbours, 1, axis=1)
                het_v_neighbours = v_neighbours[het_v_neighbour_mask]

                # Get the other vertex that makes up the edge (i.e. that of the triangle in question)
                het_v = np.repeat(np.arange(self.n_v), 3).reshape((self.n_v, 3))[het_v_neighbour_mask]

                # Build an adjacency matrix of vertex pairs (i.e. edges) that are members of a boundary
                Adj = np.zeros((self.n_v, self.n_v))
                Adj[het_v, het_v_neighbours] = 1

                # Reduce the adjacency matrix to consider only vertices that are involved in a boundary
                v_mask = (Adj.T @ np.ones(self.n_v)) != 0

                # Count the number of boundaries
                num_boundaries[i] = connected_components(csgraph=csr_matrix(Adj[v_mask][:, v_mask]), directed=False)[0]

            return num_boundaries


        def animate(self,n_frames = 100,file_name=None, dir_name="plots",an_type="periodic",tri=False):
            """
            Animate simulation, saving to mp4.

            :param n_frames: Number of frames to simulate (evenly spaced)
            :param file_name: File-name of the simulation (if None, then uses the time)
            :param dir_name: Directory name to save simulation within
            :param an_type: Animation-type -- either "periodic" or "boundary"
            :return:
            """
            if an_type == "periodic":
                plot_fn = self.plot_vor
            if an_type == "boundary":
                plot_fn = self.plot_vor_boundary

            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 1, 1)

            skip = int((self.x_save.shape[0])/n_frames)
            def animate(i):
                ax1.cla()
                if tri is True:
                    plot_fn(self.x_save[skip*i],ax1,tri=self.tri_save[skip*i])
                else:
                    plot_fn(self.x_save[skip*i],ax1,tri=False)
                if self.plot_forces is True:
                    x = self.x_save[skip*i]
                    mask = ~np.isnan(self.x_save[skip*i,:,0]) * ~np.isnan(self.x_save[skip*i+1,:,0])
                    x = x[mask]
                    F = self.x_save[skip*i+1,mask] - self.x_save[skip*i,mask]
                    ax1.quiver(x[:,0],x[:,1],F[:,0],F[:,1])
                ax1.set(aspect=1, xlim=(0, self.L), ylim=(0, self.L))

            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=15, bitrate=1800)
            if file_name is None:
                file_name = "animation %d" % time.time()
            an = animation.FuncAnimation(fig, animate, frames=n_frames, interval=200)
            an.save("%s/%s.mp4" % (dir_name, file_name), writer=writer, dpi=264)


        def normalize(self,x,xmin,xmax):
            """
            Perform a normalization between [0,1].

            x_norm = (x-xmin)/(xmax -xmin)
            :param x: Value to be normalized
            :param xmin: Minimum value (set to 0)
            :param xmax: Maximum value (set to 1)
            :return: x_norm
            """
            return (x-xmin)/(xmax-xmin)

        def animate_GRN(self, n_frames=100, file_name=None, dir_name="plots"):
            """
            Animate the simulation, saving to an mp4 file.

            Parameters
            ----------
            n_frames : int
                Number of frames to animate. Spaced evenly throughout **x_save**

            file_name : str
                Name of the file. If **None** given, generates file-name based on the time of simulation

            dir_name: str
                Directory name to save the plot.


            """
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 1, 1)

            skip = int((self.x_save.shape[0]) / n_frames)
            E_sample = self.E_save[::skip]
            E_min,E_max = E_sample.min(),E_sample.max()

            def animate(i):
                ax1.cla()
                cmap = plt.cm.plasma(self.normalize(E_sample[i],E_min,E_max))
                self.plot_vor_colored(self.x_save[skip * i], ax1,cmap)
                ax1.set(aspect=1, xlim=(0, self.L), ylim=(0, self.L))

            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=15, bitrate=1800)
            if file_name is None:
                file_name = "animation %d" % time.time()
            an = animation.FuncAnimation(fig, animate, frames=n_frames, interval=200)
            an.save("%s/%s.mp4" % (dir_name, file_name), writer=writer, dpi=264)


@jit(nopython=True,cache=True)
def dhdr(rijk):
    """
    Calculates ∂h_j/dr_i the Jacobian for all cells in each triangulation

    Last two dims: ((dhx/drx,dhx/dry),(dhy/drx,dhy/dry))

    These are lifted from Mathematica

    :param rijk_: (n_v x 3 x 2) np.float32 array of cell centroid positions for each cell in each triangulation (first two dims follow order of triangulation)
    :param vs: (n_v x 2) np.float32 array of vertex positions, corresponding to each triangle in the triangulation
    :param L: Domain size (np.float32)
    :return: Jacobian for each cell of each triangulation (n_v x 3 x 2 x 2) np.float32 array (where the first 2 dims follow the order of the triangulation.
    """
    DHDR = np.empty(rijk.shape + (2,))
    for i in range(3):
        ax,ay = rijk[:,np.mod(i,3),0],rijk[:,np.mod(i,3),1]
        bx, by = rijk[:, np.mod(i+1,3), 0], rijk[:, np.mod(i+1,3), 1]
        cx, cy = rijk[:, np.mod(i+2,3), 0], rijk[:, np.mod(i+2,3), 1]
        #dhx/drx
        DHDR[:, i, 0, 0] = (ax * (by - cy)) / ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy)) - ((by - cy) * (
                (ax ** 2 + ay ** 2) * (by - cy) + (bx ** 2 + by ** 2) * (-ay + cy) + (ay - by) * (
                cx ** 2 + cy ** 2))) / (2. * ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy)) ** 2)

        #dhy/drx
        DHDR[:, i, 0,1] = (bx ** 2 + by ** 2 - cx ** 2 + 2 * ax * (-bx + cx) - cy ** 2) / (
                    2. * ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy))) - ((by - cy) * (
                    (bx ** 2 + by ** 2) * (ax - cx) + (ax ** 2 + ay ** 2) * (-bx + cx) + (-ax + bx) * (
                        cx ** 2 + cy ** 2))) / (2. * ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy)) ** 2)

        #dhx/dry
        DHDR[:, i, 1, 0] = (-bx ** 2 - by ** 2 + cx ** 2 + 2 * ay * (by - cy) + cy ** 2) / (
                2. * ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy))) - ((-bx + cx) * (
                (ax ** 2 + ay ** 2) * (by - cy) + (bx ** 2 + by ** 2) * (-ay + cy) + (ay - by) * (
                cx ** 2 + cy ** 2))) / (2. * ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy)) ** 2)

        #dhy/dry
        DHDR[:, i, 1,1] = (ay * (-bx + cx)) / ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy)) - ((-bx + cx) * (
                    (bx ** 2 + by ** 2) * (ax - cx) + (ax ** 2 + ay ** 2) * (-bx + cx) + (-ax + bx) * (
                        cx ** 2 + cy ** 2))) / (2. * ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy)) ** 2)


    return DHDR

#
# # @jit(nopython=True,cache=True)
# def dhdr_periodic_new(Cents,vs,L):
#     n_v = Cents.shape[0]
#     r_dx = roll_forward(np.mod(roll_forward(Cents[:,:,0])-Cents[:,:,0] + L/2,L)-L/2)
#     r_dy = roll_forward(np.mod(roll_forward(Cents[:,:,1])-Cents[:,:,1] + L/2,L)-L/2)
#     r_d = np.dstack((r_dx,r_dy))
#     #Double check this
#     l = np.sqrt(r_dx**2 + r_dy**2)
#
#     lambd = l**2 * (roll_reverse(l**2)+roll_forward(l**2)-l**2)
#
#     dl2_drp = np.zeros((n_v,3,3,2))
#     mult = np.array((0,1,-1))
#     for i in range(3):
#         for j in range(2):
#             dl2_drp[:,i,:,j] = np.outer(r_d[:,i,j],np.roll(mult,i))
#
#     dL2_drp = np.zeros((n_v,3,2))
#     for i in range(2):
#         dL2_drp[:,:,i] = 2*(roll_forward(r_d[:,:,i])-roll_reverse(r_d[:,:,i]))
#
#     dlambd_dr = np.zeros((n_v,3,3,2))
#     for i in range(3):
#         for j in range(3):
#             for k in range(2):
#                 if i ==j:
#                     dlambd_dr[:,i,j] = ((2*l[:,j]**2).T * (r_d[:,np.mod(j-1,3)] - r_d[:,np.mod(j+1,3)]).T).T
#                 if np.mod(i-1,3)==j:
#                     dlambd_dr[:, i, j] = (-2*(l[:,np.mod(j-1,3)]**2 + l[:,np.mod(j+1,3)]**2 - l[:,j]**2).T*r_d[:,j].T + (2*l[:,j]**2).T * r_d[:,np.mod(j+1,3)].T).T
#                 if np.mod(i+1,3)==j:
#                     dlambd_dr[:, i, j] = (2*(l[:,np.mod(j-1,3)]**2 + l[:,np.mod(j+1,3)]**2 - l[:,j]**2).T*r_d[:,j].T - (2*l[:,j]**2).T * r_d[:,np.mod(j-1,3)].T).T
#
#     dLambd_dr = (-4*(roll_forward(l)**2 + roll_reverse(l)**2 - l**2).T * r_d.T + 4*(roll_reverse(l)**2 + l**2 - roll_forward(l)**2).T*np.dstack((roll_forward(r_d[:,:,0]),roll_forward(r_d[:,:,1]))).T).T
#
#     Lambd = lambd[:,0] + lambd[:,1] + lambd[:,2]
#
#     d_lambd_Lambd_dr = np.zeros((n_v,3,3,2))
#     for i in range(3):
#         d_lambd_Lambd_dr[:,i] = ((1 / Lambd ** 2) * (Lambd.T * dlambd_dr[:,i].T - lambd[:,i] * dLambd_dr.T)).T
#
#     equal_component = Cents[:,np.newaxis,:,np.newaxis,:]*d_lambd_Lambd_dr[:,:,:,:,np.newaxis]
#     equal_component = equal_component[:,0] + equal_component[:,1] + equal_component[:,2]
#     unequal_component = ((lambd.T/Lambd).T)[:,:,np.newaxis,np.newaxis]*np.eye(2)[np.newaxis,np.newaxis,:,:]
#     Dhdr = equal_component + unequal_component
#     return Dhdr

@jit(nopython=True,cache=True)
def dhdr_periodic(rijk_,vs,L):
    """
    Same as **dhdr** apart from accounts for periodic triangulation.

    Calculates ∂h_j/dr_i the Jacobian for all cells in each triangulation

    Last two dims: ((dhx/drx,dhx/dry),(dhy/drx,dhy/dry))

    These are lifted from Mathematica

    :param rijk_: (n_v x 3 x 2) np.float32 array of cell centroid positions for each cell in each triangulation (first two dims follow order of triangulation)
    :param vs: (n_v x 2) np.float32 array of vertex positions, corresponding to each triangle in the triangulation
    :param L: Domain size (np.float32)
    :return: Jacobian for each cell of each triangulation (n_v x 3 x 2 x 2) np.float32 array (where the first 2 dims follow the order of the triangulation.
    """
    rijk = np.empty_like(rijk_)
    for i in range(3):
        rijk[:,i,:] = np.remainder(rijk_[:,i] - vs + L/2,L) - L/2

    DHDR = np.empty(rijk.shape + (2,))
    for i in range(3):
        ax,ay = rijk[:,np.mod(i,3),0],rijk[:,np.mod(i,3),1]
        bx, by = rijk[:, np.mod(i+1,3), 0], rijk[:, np.mod(i+1,3), 1]
        cx, cy = rijk[:, np.mod(i+2,3), 0], rijk[:, np.mod(i+2,3), 1]
        #dhx/drx
        DHDR[:, i, 0, 0] = (ax * (by - cy)) / ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy)) - ((by - cy) * (
                (ax ** 2 + ay ** 2) * (by - cy) + (bx ** 2 + by ** 2) * (-ay + cy) + (ay - by) * (
                cx ** 2 + cy ** 2))) / (2. * ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy)) ** 2)

        #dhy/drx
        DHDR[:, i, 0,1] = (bx ** 2 + by ** 2 - cx ** 2 + 2 * ax * (-bx + cx) - cy ** 2) / (
                    2. * ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy))) - ((by - cy) * (
                    (bx ** 2 + by ** 2) * (ax - cx) + (ax ** 2 + ay ** 2) * (-bx + cx) + (-ax + bx) * (
                        cx ** 2 + cy ** 2))) / (2. * ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy)) ** 2)

        #dhx/dry
        DHDR[:, i, 1, 0] = (-bx ** 2 - by ** 2 + cx ** 2 + 2 * ay * (by - cy) + cy ** 2) / (
                2. * ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy))) - ((-bx + cx) * (
                (ax ** 2 + ay ** 2) * (by - cy) + (bx ** 2 + by ** 2) * (-ay + cy) + (ay - by) * (
                cx ** 2 + cy ** 2))) / (2. * ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy)) ** 2)

        #dhy/dry
        DHDR[:, i, 1,1] = (ay * (-bx + cx)) / ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy)) - ((-bx + cx) * (
                    (bx ** 2 + by ** 2) * (ax - cx) + (ax ** 2 + ay ** 2) * (-bx + cx) + (-ax + bx) * (
                        cx ** 2 + cy ** 2))) / (2. * ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy)) ** 2)


    return DHDR




@jit(nopython=True)
def get_neighbours(tri,neigh=None,Range=None):
    """
    Given a triangulation, find the neighbouring triangles of each triangle.

    By convention, the column i in the output -- neigh -- corresponds to the triangle that is opposite the cell i in that triangle.

    Can supply neigh, meaning the algorithm only fills in gaps (-1 entries)

    :param tri: Triangulation (n_v x 3) np.int32 array
    :param neigh: neighbourhood matrix to update {Optional}
    :return: (n_v x 3) np.int32 array, storing the three neighbouring triangles. Values correspond to the row numbers of tri
    """
    n_v = tri.shape[0]
    if neigh is None:
        neigh = np.ones_like(tri,dtype=np.int32)*-1
    if Range is None:
        Range = np.arange(n_v)
    tri_compare = np.concatenate((tri.T, tri.T)).T.reshape((-1, 3, 2))
    for j in Range:#range(n_v):
        tri_sample_flip = np.flip(tri[j])
        tri_i = np.concatenate((tri_sample_flip,tri_sample_flip)).reshape(3,2)
        for k in range(3):
            if neigh[j,k]==-1:
                neighb,l = np.nonzero((tri_compare[:,:,0]==tri_i[k,0])*(tri_compare[:,:,1]==tri_i[k,1]))
                neighb,l = neighb[0],l[0]
                neigh[j,k] = neighb
                neigh[neighb,np.mod(2-l,3)] = j
    return neigh

@jit(nopython=True,cache=True)
def order_tris(tri):
    """
    For each triangle (i.e. row in **tri**), order cell ids in ascending order
    :param tri: Triangulation (n_v x 3) np.int32 array
    :return: the ordered triangulation
    """
    nv = tri.shape[0]
    for i in range(nv):
        Min = np.argmin(tri[i])
        tri[i] = tri[i,Min],tri[i,np.mod(Min+1,3)],tri[i,np.mod(Min+2,3)]
    return tri

@jit(nopython=True,cache=True)
def circumcenter_periodic(C,L):
    """
    Find the circumcentre (i.e. vertex position) of each triangle in the triangulation.

    :param C: Cell centroids for each triangle in triangulation (n_c x 3 x 2) np.float32 array
    :param L: Domain size (np.float32)
    :return: Circumcentres/vertex-positions (n_v x 2) np.float32 array
    """
    ri, rj, rk = C.transpose(1,2,0)
    r_mean = (ri+rj+rk)/3
    disp = r_mean - L / 2
    ri,rj,rk = np.mod(ri-disp,L),np.mod(rj-disp,L),np.mod(rk-disp,L)
    ax, ay = ri
    bx, by = rj
    cx, cy = rk
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (
            ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (
            bx - ax)) / d
    vs = np.empty((ax.size,2),dtype=np.float64)
    vs[:,0],vs[:,1] = ux,uy
    vs = np.mod(vs+disp.T,L)
    return vs


@jit(nopython=True,cache=True)
def circumcenter(C):
    """
    Find the circumcentre (i.e. vertex position) of each triangle in the triangulation.

    :param C: Cell centroids for each triangle in triangulation (n_c x 3 x 2) np.float32 array
    :param L: Domain size (np.float32)
    :return: Circumcentres/vertex-positions (n_v x 2) np.float32 array
    """
    ri, rj, rk = C.transpose(1,2,0)
    ax, ay = ri
    bx, by = rj
    cx, cy = rk
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (
            ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (
            bx - ax)) / d
    vs = np.empty((ax.size,2),dtype=np.float64)
    vs[:,0],vs[:,1] = ux,uy
    return vs

@jit(nopython=True,cache=True)
def tri_angles(x, tri):
    """
    Find angles that make up each triangle in the triangulation. By convention, column i defines the angle
    corresponding to cell centroid i

    :param x: Cell centroids (n_c x 2) np.float32 array
    :param tri: Triangulation (n_v x 3) np.int32 array
    :param L: Domain size (np.float32)
    :return: tri_angles (n_v x 3) np.flaot32 array (in radians)
    """
    three = np.array([0,1,2])
    i_b = np.mod(three + 1, 3)
    i_c = np.mod(three + 2, 3)

    C = np.empty((tri.shape[0],3,2))
    for i, TRI in enumerate(tri):
        C[i] = x[TRI]
    a2 = (C[:, i_b, 0] - C[:, i_c, 0]) ** 2 + (C[:, i_b, 1] - C[:, i_c, 1]) ** 2
    b2 = (C[:, :, 0] - C[:, i_c, 0] ) ** 2 + (C[:, :, 1] - C[:, i_c, 1]) ** 2
    c2 = (C[:, i_b, 0] - C[:, :, 0] ) ** 2 + (C[:, i_b, 1] - C[:, :, 1]) ** 2

    cos_Angles = (b2 + c2 - a2) / (2 * np.sqrt(b2) * np.sqrt(c2))
    Angles = np.arccos(cos_Angles)
    return Angles


@jit(nopython=True, cache=True)
def tri_angles_periodic(x, tri, L):
    """
    Same as **tri_angles** apart from accounts for periodic triangulation (i.e. the **L**)

    Find angles that make up each triangle in the triangulation. By convention, column i defines the angle
    corresponding to cell centroid i

    :param x: Cell centroids (n_c x 2) np.float32 array
    :param tri: Triangulation (n_v x 3) np.int32 array
    :param L: Domain size (np.float32)
    :return: tri_angles (n_v x 3) np.flaot32 array (in radians)
    """
    three = np.array([0, 1, 2])
    i_b = np.mod(three + 1, 3)
    i_c = np.mod(three + 2, 3)

    C = np.empty((tri.shape[0], 3, 2))
    for i, TRI in enumerate(tri):
        C[i] = x[TRI]
    a2 = (np.mod(C[:, i_b, 0] - C[:, i_c, 0] + L / 2, L) - L / 2) ** 2 + (
            np.mod(C[:, i_b, 1] - C[:, i_c, 1] + L / 2, L) - L / 2) ** 2
    b2 = (np.mod(C[:, :, 0] - C[:, i_c, 0] + L / 2, L) - L / 2) ** 2 + (
            np.mod(C[:, :, 1] - C[:, i_c, 1] + L / 2, L) - L / 2) ** 2
    c2 = (np.mod(C[:, i_b, 0] - C[:, :, 0] + L / 2, L) - L / 2) ** 2 + (
            np.mod(C[:, i_b, 1] - C[:, :, 1] + L / 2, L) - L / 2) ** 2

    cos_Angles = (b2 + c2 - a2) / (2 * np.sqrt(b2) * np.sqrt(c2))
    Angles = np.arccos(cos_Angles)
    return Angles


@jit(nopython=True,cache=True)
def get_k2(tri, v_neighbours):
    """
    To determine whether a given neighbouring pair of triangles needs to be re-triangulated, one considers the sum of
    the pair angles of the triangles associated with the cell centroids that are **not** themselves associated with the
    adjoining edge. I.e. these are the **opposite** angles.

    Given one cell centroid/angle in a given triangulation, k2 defines the column index of the cell centroid/angle in the **opposite** triangle

    :param tri: Triangulation (n_v x 3) np.int32 array
    :param v_neighbours: Neighbourhood matrix (n_v x 3) np.int32 array
    :return:
    """
    three = np.array([0, 1, 2])
    nv = tri.shape[0]
    k2s = np.empty((nv, 3), dtype=np.int32)
    for i in range(nv):
        for k in range(3):
            neighbour = v_neighbours[i, k]
            k2 = ((v_neighbours[neighbour] == i) * three).sum()
            k2s[i, k] = k2
    return k2s


@jit(nopython=True,cache=True)
def get_k2_boundary(tri, v_neighbours):
    """
    Same as **get_k2** but fills in -1 if the k2 neighbour is undefined (-1)

    To determine whether a given neighbouring pair of triangles needs to be re-triangulated, one considers the sum of
    the pair angles of the triangles associated with the cell centroids that are **not** themselves associated with the
    adjoining edge. I.e. these are the **opposite** angles.

    Given one cell centroid/angle in a given triangulation, k2 defines the column index of the cell centroid/angle in the **opposite** triangle

    :param tri: Triangulation (n_v x 3) np.int32 array
    :param v_neighbours: Neighbourhood matrix (n_v x 3) np.int32 array
    :return:
    """
    three = np.array([0, 1, 2])
    nv = tri.shape[0]
    k2s = np.empty((nv, 3), dtype=np.int32)
    for i in range(nv):
        for k in range(3):
            neighbour = v_neighbours[i, k]
            if neighbour == -1:
                k2s[i,k] = -1
            else:
                k2 = ((v_neighbours[neighbour] == i) * three).sum()
                k2s[i, k] = k2
    return k2s


@jit(nopython=True,cache=True)
def make_y(x,Lgrid_xy):
    """
    Makes the (9) tiled set of coordinates used to perform the periodic triangulation.

    :param x: Cell centroids (n_c x 2) np.float32 array
    :param Lgrid_xy: (9 x 2) array defining the displacement vectors for each of the 9 images of the tiling
    :return: Tiled set of coordinates (9n_c x 2) np.float32 array
    """
    n_c = x.shape[0]
    y = np.empty((n_c*9,x.shape[1]))
    for k in range(9):
        y[k*n_c:(k+1)*n_c] = x + Lgrid_xy[k]
    return y


@jit(nopython=True,cache=True)
def get_A_periodic(vs,neighbours,Cents,CV_matrix,L,n_c):
    """
    Calculates area of each cell (considering periodic boundary conditions)

    :param vs: (nv x 2) matrix considering coordinates of each vertex
    :param neighbours: (nv x 3 x 2) matrix considering the coordinates of each neighbouring vertex of each vertex
    :param Cents: (nv x 3 x 2) array considering cell centroids of each cell involved in each triangulation
    :param CV_matrix: cell-vertex binary matrix (n_c x n_v x 3) (where the 3rd dimension prescribes the order)
    :param L: Domain size np.float32
    :param n_c: Number of cells (np.int32)
    :return: self.A saves the areas of each cell
    """
    AA_mat = np.empty((neighbours.shape[0], neighbours.shape[1]))
    for i in range(3):
        Neighbours = np.remainder(neighbours[:, np.mod(i + 2, 3)] - Cents[:, i] + L / 2, L) -L/2
        Vs = np.remainder(vs - Cents[:, i] + L / 2, L) - L/2
        AA_mat[:, i] = 0.5 * (Neighbours[:, 0] * Vs[:, 1] - Neighbours[:, 1] * Vs[:, 0])
    A = np.zeros((n_c))
    for i in range(3):
        A += np.asfortranarray(CV_matrix[:, :, i])@np.asfortranarray(AA_mat[:, i])
    return A


@jit(nopython=True,cache=True)
def get_A(vs,neighbours,CV_matrix,n_c):
    """
    Calculates area of each cell (considering periodic boundary conditions)

    :param vs: (nv x 2) matrix considering coordinates of each vertex
    :param neighbours: (nv x 3 x 2) matrix considering the coordinates of each neighbouring vertex of each vertex
    :param Cents: (nv x 3 x 2) array considering cell centroids of each cell involved in each triangulation
    :param CV_matrix: cell-vertex binary matrix (n_c x n_v x 3) (where the 3rd dimension prescribes the order)
    :param L: Domain size np.float32
    :param n_c: Number of cells (np.int32)
    :return: self.A saves the areas of each cell
    """
    AA_mat = np.empty((neighbours.shape[0], neighbours.shape[1]))
    for i in range(3):
        Neighbours = neighbours[:, np.mod(i + 2, 3)]
        AA_mat[:, i] = 0.5 * (Neighbours[:, 0] * vs[:, 1] - Neighbours[:, 1] * vs[:, 0])
    AA_flat = AA_mat.ravel()
    AA_flat[np.isnan(AA_flat)] = 0
    AA_mat = AA_flat.reshape(AA_mat.shape)
    A = np.zeros((n_c))
    for i in range(3):
        A += np.asfortranarray(CV_matrix[:, :, i])@np.asfortranarray(AA_mat[:, i])
    return A


@jit(nopython=True,cache=True)
def get_P_periodic(vs,neighbours,CV_matrix,L,n_c):
    """
    Finds perimeter of each cell (given periodic boundary conditions)

    :param vs: (nv x 2) matrix considering coordinates of each vertex
    :param neighbours: (nv x 3 x 2) matrix considering the coordinates of each neighbouring vertex of each vertex
    :param CV_matrix: cell-vertex binary matrix (n_c x n_v x 3) (where the 3rd dimension prescribes the order)
    :param L: Domain size np.float32
    :param n_c: Number of cells (np.int32)
    :return: P (n_c x 1) np.float32 array of perimeters for each cell
    """
    P_m = np.empty((neighbours.shape[0], neighbours.shape[1]))
    for i in range(3):
        Neighbours = np.remainder(neighbours[:, i] - vs + L / 2, L) - L / 2
        P_m[:, i] = np.sqrt((Neighbours[:, 0]) ** 2 + (Neighbours[:, 1]) ** 2)  # * self.boundary_mask

    PP_mat = np.zeros(P_m.shape)
    for i in range(3):
        PP_mat[:, i] = (P_m[:, np.mod(i + 1, 3)] + P_m[:, np.mod(i + 2, 3)]) / 2

    P = np.zeros((n_c))
    for i in range(3):
        P += np.asfortranarray(CV_matrix[:, :, i])@np.asfortranarray(PP_mat[:, i])
    return P



@jit(nopython=True,cache=True)
def get_P(vs,neighbours,CV_matrix,n_c):
    """
    Finds perimeter of each cell (given periodic boundary conditions)

    :param vs: (nv x 2) matrix considering coordinates of each vertex
    :param neighbours: (nv x 3 x 2) matrix considering the coordinates of each neighbouring vertex of each vertex
    :param CV_matrix: cell-vertex binary matrix (n_c x n_v x 3) (where the 3rd dimension prescribes the order)
    :param L: Domain size np.float32
    :param n_c: Number of cells (np.int32)
    :return: P (n_c x 1) np.float32 array of perimeters for each cell
    """
    P_m = np.empty((neighbours.shape[0], neighbours.shape[1]))
    for i in range(3):
        Neighbours = neighbours[:, i] - vs
        P_m[:, i] = np.sqrt((Neighbours[:, 0]) ** 2 + (Neighbours[:, 1]) ** 2)  # * self.boundary_mask

    PP_mat = np.zeros(P_m.shape)
    for i in range(3):
        PP_mat[:, i] = P_m[:, np.mod(i + 2, 3)]

        # PP_mat[:, i] = (P_m[:, np.mod(i + 1, 3)] + P_m[:, np.mod(i + 2, 3)]) / 2

    PP_flat = PP_mat.ravel()
    PP_flat[np.isnan(PP_flat)] = 0
    PP_mat = PP_flat.reshape(PP_mat.shape)

    P = np.zeros((n_c))
    for i in range(3):
        P += np.asfortranarray(CV_matrix[:, :, i])@np.asfortranarray(PP_mat[:, i])
    return P


@jit(nopython=True,cache=True)
def roll_forward(x):
    """
    Jitted equivalent to np.roll(x,1,axis=1)
    :param x:
    :return:
    """
    return np.column_stack((x[:,2],x[:,:2]))

@jit(nopython=True,cache=True)
def roll_reverse(x):
    """
    Jitted equivalent to np.roll(x,-1,axis=1)
    :param x:
    :return:
    """
    return np.column_stack((x[:,1:3],x[:,0]))

@jit(nopython=True,cache=True)
def get_F_periodic(vs, neighbours,tris,CV_matrix,n_v,n_c,L,J_CW,J_CCW,A,P,X,kappa_A,kappa_P,A0,P0):

    h_j = np.empty((n_v, 3, 2))
    for i in range(3):
        h_j[:, i] = vs
    h_jm1 = np.dstack((roll_forward(neighbours[:,:,0]),roll_forward(neighbours[:,:,1])))
    h_jp1 = np.dstack((roll_reverse(neighbours[:,:,0]),roll_reverse(neighbours[:,:,1])))


    dAdh_j = 0.5*(np.mod(h_jp1 - h_jm1 + L / 2, L) - L / 2)
    dAdh_j = np.dstack((dAdh_j[:,:,1],-dAdh_j[:,:,0]))

    l_jm1 = np.mod(h_j - h_jm1 + L / 2, L) - L / 2
    l_jp1 = np.mod(h_j - h_jp1 + L / 2, L) - L / 2
    l_jm1_norm, l_jp1_norm = np.sqrt(l_jm1[:,:,0] ** 2 + l_jm1[:,:,1] ** 2), np.sqrt(l_jp1[:,:,0] ** 2 +  l_jp1[:,:,1] ** 2)
    dPdh_j = (l_jm1.T/l_jm1_norm.T + l_jp1.T/l_jp1_norm.T).T

    dljidh_j = (l_jm1.T * J_CCW.T/l_jm1_norm.T + l_jp1.T * J_CW.T/l_jp1_norm.T).T


    ## 3. Find areas and perimeters of the cells and restructure data wrt. the triangulation
    vA = A[tris.ravel()].reshape(tris.shape)
    vP = P[tris.ravel()].reshape(tris.shape)

    # 4. Calculate ∂h/∂r. This is for cell i (which may or may not be cell j, and triangulates with cell j)
    # This last two dims are a Jacobinan (2x2) matrix, defining {x,y} for h and r. See function description for details
    DHDR = dhdr_periodic(X, vs, L)  # order is wrt cell i

    # 5. Now calculate the force component for each vertex, with respect to the 3 neighbouring cells
    #   This is essentially decomposing the chain rule of the expression of F for each cell by vertex
    #   M_sum is a (nv,2,3) matrix. This is the force contribution for each cell of a given triangle/vertex (3rd dim). Considering {x,y} components (2nd dim)
    #       Within the function, this calculates (direct and indirect) contributions of each cell wrt each other cell (i.e. 3x3), then adds them together
    M = np.zeros((2, n_v, 2, 3))
    for i in range(3):
        for j in range(3):
            for Fdim in range(2):
                M[:, :, Fdim, i] += DHDR[:, i, Fdim].T * \
                                    (kappa_A * (vA[:, j] - A0) * dAdh_j[:, j].T
                                     + kappa_P * (vP[:, j]-P0) * dPdh_j[:, j].T
                                     + dljidh_j[:,j].T)
    M = M[0] + M[1]


    # 6. Compile force components wrt. cells by using the cell-to-vertex connection matrix.
    #       Force on cell_i = SUM_{vertices of cell i} {forces at each vertex wrt. cell i}
    dEdr = np.zeros((n_c, 2))
    for i in range(3):
        dEdr += np.asfortranarray(CV_matrix[:, :, i])@np.asfortranarray(M[:, :, i])
    F = -dEdr

    return F

#
# @jit(nopython=True,cache=True)
# def get_F_periodic_param(vs, neighbours,tris,CV_matrix,n_v,n_c,L,J_CW,J_CCW,A,P,X,kappa_A,kappa_P,A0,P0):
#
#     h_j = np.empty((n_v, 3, 2))
#     for i in range(3):
#         h_j[:, i] = vs
#     h_jm1 = np.dstack((roll_forward(neighbours[:,:,0]),roll_forward(neighbours[:,:,1])))
#     h_jp1 = np.dstack((roll_reverse(neighbours[:,:,0]),roll_reverse(neighbours[:,:,1])))
#
#
#     dAdh_j = np.mod(h_jp1 - h_jm1 + L / 2, L) - L / 2
#     dAdh_j = np.dstack((dAdh_j[:,:,1],-dAdh_j[:,:,0]))
#
#     l_jm1 = np.mod(h_j - h_jm1 + L / 2, L) - L / 2
#     l_jp1 = np.mod(h_j - h_jp1 + L / 2, L) - L / 2
#     l_jm1_norm, l_jp1_norm = np.sqrt(l_jm1[:,:,0] ** 2 + l_jm1[:,:,1] ** 2), np.sqrt(l_jp1[:,:,0] ** 2 +  l_jp1[:,:,1] ** 2)
#     dPdh_j = (l_jm1.T/l_jm1_norm.T + l_jp1.T/l_jp1_norm.T).T
#
#     dljidh_j = (l_jm1.T * J_CCW.T/l_jm1_norm.T + l_jp1.T * J_CW.T/l_jp1_norm.T).T
#
#
#     ## 3. Find areas and perimeters of the cells and restructure data wrt. the triangulation
#     vA = A[tris.ravel()].reshape(tris.shape)
#     vP = P[tris.ravel()].reshape(tris.shape)
#     vKa = kappa_A[tris.ravel()].reshape(tris.shape)
#     vKp = kappa_P[tris.ravel()].reshape(tris.shape)
#     # 4. Calculate ∂h/∂r. This is for cell i (which may or may not be cell j, and triangulates with cell j)
#     # This last two dims are a Jacobinan (2x2) matrix, defining {x,y} for h and r. See function description for details
#     DHDR = dhdr_periodic(X, vs, L)  # order is wrt cell i
#
#     # 5. Now calculate the force component for each vertex, with respect to the 3 neighbouring cells
#     #   This is essentially decomposing the chain rule of the expression of F for each cell by vertex
#     #   M_sum is a (nv,2,3) matrix. This is the force contribution for each cell of a given triangle/vertex (3rd dim). Considering {x,y} components (2nd dim)
#     #       Within the function, this calculates (direct and indirect) contributions of each cell wrt each other cell (i.e. 3x3), then adds them together
#     M = np.zeros((2, n_v, 2, 3))
#     for i in range(3):
#         for j in range(3):
#             for Fdim in range(2):
#                 M[:, :, Fdim, i] += DHDR[:, i, Fdim].T * \
#                                     (vKa[:,j] * (vA[:, j] - A0) * dAdh_j[:, j].T
#                                      + vKp[:,j] * (vP[:, j]-P0) * dPdh_j[:, j].T
#                                      + dljidh_j[:,j].T)
#     M = M[0] + M[1]
#
#
#     # 6. Compile force components wrt. cells by using the cell-to-vertex connection matrix.
#     #       Force on cell_i = SUM_{vertices of cell i} {forces at each vertex wrt. cell i}
#     dEdr = np.zeros((n_c, 2))
#     for i in range(3):
#         dEdr += np.asfortranarray(CV_matrix[:, :, i])@np.asfortranarray(M[:, :, i])
#     F = -dEdr
#
#     return F


@jit(nopython=True,cache=True)
def get_F(vs, neighbours,tris,CV_matrix,n_v,n_c,L,J_CW,J_CCW,A,P,X,kappa_A,kappa_P,A0,P0,n_C,kappa_B,l_b0):


    h_j = np.empty((n_v, 3, 2))
    for i in range(3):
        h_j[:, i] = vs
    h_jm1 = np.dstack((roll_forward(neighbours[:,:,0]),roll_forward(neighbours[:,:,1])))
    h_jp1 = np.dstack((roll_reverse(neighbours[:,:,0]),roll_reverse(neighbours[:,:,1])))

    dAdh_j = h_jp1 - h_jm1
    dAdh_j = np.dstack((dAdh_j[:,:,1],-dAdh_j[:,:,0]))

    l_jm1 = h_j - h_jm1
    l_jp1 = h_j - h_jp1
    l_jm1_norm, l_jp1_norm = np.sqrt(l_jm1[:,:,0] ** 2 + l_jm1[:,:,1] ** 2), np.sqrt(l_jp1[:,:,0] ** 2 +  l_jp1[:,:,1] ** 2)
    dPdh_j = (l_jm1.T/l_jm1_norm.T + l_jp1.T/l_jp1_norm.T).T

    dljidh_j = (l_jm1.T * J_CCW.T/l_jm1_norm.T + l_jp1.T * J_CW.T/l_jp1_norm.T).T




    ## 3. Find areas and perimeters of the cells and restructure data wrt. the triangulation
    # vA = A[tris.ravel()].reshape(tris.shape)
    # vP = P[tris.ravel()].reshape(tris.shape)


    real_cell = np.zeros(n_c)
    real_cell[:n_C] = 1

    vdA = ((A-A0)*real_cell)[tris.ravel()].reshape(tris.shape)
    vdP = ((P-P0)*real_cell)[tris.ravel()].reshape(tris.shape)

    # 4. Calculate ∂h/∂r. This is for cell i (which may or may not be cell j, and triangulates with cell j)
    # This last two dims are a Jacobinan (2x2) matrix, defining {x,y} for h and r. See function description for details
    DHDR = dhdr(X)  # order is wrt cell i

    vRC = real_cell[tris.ravel()].reshape(tris.shape)
    bEdge = vRC*roll_forward(1-vRC) + (1-vRC)*roll_forward(vRC) + vRC*roll_reverse(1-vRC) + (1-vRC)*roll_reverse(vRC)

    # 5. Now calculate the force component for each vertex, with respect to the 3 neighbouring cells
    #   This is essentially decomposing the chain rule of the expression of F for each cell by vertex
    #   M_sum is a (nv,2,3) matrix. This is the force contribution for each cell of a given triangle/vertex (3rd dim). Considering {x,y} components (2nd dim)
    #       Within the function, this calculates (direct and indirect) contributions of each cell wrt each other cell (i.e. 3x3), then adds them together
    M = np.zeros((2, n_v, 2, 3))
    for i in range(3):
        for j in range(3):
            for Fdim in range(2):
                M[:, :, Fdim, i] += DHDR[:, i, Fdim].T * \
                                    (kappa_A * vdA[:,j] * dAdh_j[:, j].T
                                     + kappa_P * vdP[:,j] * dPdh_j[:, j].T
                                     + vRC[:,j]*dljidh_j[:,j].T
                                     # + dljidh_j[:, j].T
                                     + kappa_B*bEdge[:,j]*(l_jp1_norm[:,j] - l_b0)*dPdh_j[:,j].T)
    M = M[0] + M[1]


    M_flat = M.ravel()
    M_flat[np.isnan(M_flat)] = 0
    M = M_flat.reshape(M.shape)

    # 6. Compile force components wrt. cells by using the cell-to-vertex connection matrix.
    #       Force on cell_i = SUM_{vertices of cell i} {forces at each vertex wrt. cell i}
    dEdr = np.zeros((n_c, 2))
    for i in range(3):
        dEdr += np.asfortranarray(CV_matrix[:, :, i])@np.asfortranarray(M[:, :, i])
    F = -dEdr

    return F

@jit(nopython=True,cache=True)
def weak_repulsion(Cents,a,k, CV_matrix,n_c,L):
    """
    Additional "soft" pair-wise repulsion at short range to prevent unrealistic and sudden changes in triangulation.

    Repulsion is on the imediate neighbours (i.e. derived from the triangulation)

    And is performed respecting periodic boudnary conditions (system size = L)

    Suppose l_{ij} = \| r_i - r_j \
    F_soft = -k(l_{ij} - 2a)(r_i - r_j) if l_{ij} < 2a; and =0 otherwise

    :param Cents: Cell centroids on the triangulation (n_v x 3 x 2) **np.ndarray** of dtype **np.float64**
    :param a: Cut-off distance of spring-like interaction (**np.float64**)
    :param k: Strength of spring-like interaction (**np.float64**)
    :param CV_matrix: Cell-vertex matrix representation of the triangulation (n_c x n_v x 3)
    :param n_c: Number of cells (**np.int64**)
    :param L: Domain size/length (**np.float64**)
    :return: F_soft
    """
    CCW = np.dstack((roll_reverse(Cents[:,:,0]),roll_reverse(Cents[:,:,1])))#np.column_stack((Cents[:,1:3],Cents[:,0].reshape(-1,1,2)))
    displacement = np.mod(Cents - CCW + L/2,L) - L/2
    rij = np.sqrt(displacement[:,:,0]**2 + displacement[:,:,1]**2)
    norm_disp = (displacement.T/rij.T).T
    V_soft_mag = -k*(rij - 2*a)*(rij<2*a)
    V_soft_CCW = (V_soft_mag.T*norm_disp.T).T
    V_soft_CW = -(roll_forward(V_soft_mag).T*norm_disp.T).T
    V_soft = V_soft_CW + V_soft_CCW
    F_soft = np.zeros((n_c, 2))
    for i in range(3):
        F_soft += np.asfortranarray(CV_matrix[:, :, i])@np.asfortranarray(V_soft[:, i])
    return F_soft

@jit(nopython=True,cache=True)
def get_l_interface(n_v,n_c, neighbours, vs, CV_matrix,L):
    """
    Get the length of the interface between each pair of cells.

    LI[i,j] = length of interface between cell i and j = L[j,i] (if using periodic triangulation)

    :param n_v: Number of vertices (**np.int64**)
    :param n_c: Number of cells (**np.int64**
    :param neighbours: Position of the three neighbouring vertices (n_v x 3 x 2)
    :param vs: Positions of vertices (n_v x 3)
    :param CV_matrix: Cell-vertex matrix representation of triangulation (n_c x n_v x 3)
    :param L: Domain size (**np.float32**)
    :return:
    """
    h_j = np.empty((n_v, 3, 2))
    for i in range(3):
        h_j[:, i] = vs
    h_jp1 = np.dstack((roll_reverse(neighbours[:,:,0]),roll_reverse(neighbours[:,:,1])))
    l = np.mod(h_j - h_jp1 + L/2,L) - L/2
    l = np.sqrt(l[:,:,0]**2 + l[:,:,1]**2)
    LI = np.zeros((n_c,n_c),dtype=np.float32)
    for i in range(3):
        LI+= np.asfortranarray(l[:,i]*CV_matrix[:,:,i])@np.asfortranarray(CV_matrix[:,:,np.mod(i+2,3)].T)
    return LI


def get_l_interface_boundary(n_v,n_c, neighbours, vs, CV_matrix):
    """
    Same as **get_l_interface** but accounts for boundary cells

    Get the length of the interface between each pair of cells.

    LI[i,j] = length of interface between cell i and j = L[j,i]

    Note: the difference lies in the making the LI matrix symmetric.

    :param n_v: Number of vertices (**np.int64**)
    :param n_c: Number of cells (**np.int64**
    :param neighbours: Position of the three neighbouring vertices (n_v x 3 x 2)
    :param vs: Positions of vertices (n_v x 3)
    :param CV_matrix: Cell-vertex matrix representation of triangulation (n_c x n_v x 3)
    :return:
    """
    h_j = np.empty((n_v, 3, 2))
    for i in range(3):
        h_j[:, i] = vs
    h_jp1 = np.dstack((roll_reverse(neighbours[:,:,0]),roll_reverse(neighbours[:,:,1])))
    l = h_j - h_jp1
    l = np.sqrt(l[:,:,0]**2 + l[:,:,1]**2)
    LI = np.zeros((n_c,n_c),dtype=np.float32)
    for i in range(3):
        LI+= np.asfortranarray(l[:,i]*CV_matrix[:,:,i])@np.asfortranarray(CV_matrix[:,:,np.mod(i+2,3)].T)
    LI = np.dstack((LI,LI.T)).max(axis=2)
    return LI

@jit(nopython=True,cache=True)
def weak_repulsion_boundary(Cents,a,k, CV_matrix,n_c,n_C):
    """
    Identical to **weak_repulsion** apart from without periodic boundary conditions

    Additional "soft" pair-wise repulsion at short range to prevent unrealistic and sudden changes in triangulation.

    Repulsion is on the imediate neighbours (i.e. derived from the triangulation)

    Suppose l_{ij} = \| r_i - r_j \
    F_soft = -k(l_{ij} - 2a)(r_i - r_j) if l_{ij} < 2a; and =0 otherwise

    :param Cents: Cell centroids on the triangulation (n_v x 3 x 2) **np.ndarray** of dtype **np.float64**
    :param a: Cut-off distance of spring-like interaction (**np.float64**)
    :param k: Strength of spring-like interaction (**np.float64**)
    :param CV_matrix: Cell-vertex matrix representation of the triangulation (n_c x n_v x 3)
    :param n_c: Number of cells (**np.int64**)
    :return: F_soft
    """
    CCW = np.dstack((roll_reverse(Cents[:,:,0]),roll_reverse(Cents[:,:,1])))#np.column_stack((Cents[:,1:3],Cents[:,0].reshape(-1,1,2)))
    CCW_displacement = Cents - CCW
    rij = np.sqrt(CCW_displacement[:,:,0]**2 + CCW_displacement[:,:,1]**2)
    norm_disp = (CCW_displacement.T/rij.T).T
    V_soft_mag = -k*(rij - 2*a)*(rij<2*a)
    V_soft_CCW = (V_soft_mag.T*norm_disp.T).T
    V_soft_CW = -(roll_forward(V_soft_mag).T*norm_disp.T).T
    V_soft = V_soft_CW + V_soft_CCW
    F_soft = np.zeros((n_c, 2))
    for i in range(3):
        F_soft += np.asfortranarray(CV_matrix[:, :, i])@np.asfortranarray(V_soft[:, i])
    F_soft[n_C:] = 0
    return F_soft
@jit(nopython=True,cache=True)

def boundary_tension(Gamma_bound,n_C,n_c,Cents,CV_matrix):
    boundary_cells = np.zeros(n_c)
    boundary_cells[n_C:] = 1
    CCW = np.dstack((roll_reverse(Cents[:,:,0]),roll_reverse(Cents[:,:,1])))#np.column_stack((Cents[:,1:3],Cents[:,0].reshape(-1,1,2)))
    CW = np.dstack((roll_forward(Cents[:,:,0]),roll_forward(Cents[:,:,1])))#np.column_stack((Cents[:,1:3],Cents[:,0].reshape(-1,1,2)))
    CCW_displacement = Cents - CCW
    CW_displacement = Cents - CW

    forces = -CW_displacement - CCW_displacement
    F_boundary = np.zeros((n_c, 2))
    for i in range(3):
        F_boundary += np.asfortranarray(CV_matrix[:, :, i])@np.asfortranarray(forces[:,i])
    F_boundary[:n_C] = 0
    return Gamma_bound*F_boundary

@jit(nopython=True,cache=True)
def get_C(n_c,CV_matrix):
    """
    Generates a cell-cell interaction matrix (binary) (n_c x n_c).

    If entry C[i,j] is 1, means that cell j is the CW neighbour of cell i in one of the triangles of the triangulation

    Note: if the triangulation is not periodic, this directionality will result in asymmetric entries of rows/cols
    associated with boundary cells. To generate a symmetric interaction matrix, perform (C + C.T)!=0

    :param n_c: Number of cells **np.int64**
    :param CV_matrix: Cell-vertex matrix representation of triangulation (n_c x n_v x 3)
    :return:
    """
    C = np.zeros((n_c, n_c), dtype=np.float32)
    for i in range(3):
        C += np.asfortranarray(CV_matrix[:, :, i]) @ np.asfortranarray(CV_matrix[:, :, np.mod(i + 2, 3)].T)
    C = (C != 0).astype(np.int32)
    return C


@jit(nopython=True,cache=True)
def get_C_boundary(n_c,CV_matrix):
    """
    Generates a cell-cell interaction matrix (binary) (n_c x n_c).

    If entry C[i,j] is 1, means that cell j is the CW neighbour of cell i in one of the triangles of the triangulation

    Note: if the triangulation is not periodic, this directionality will result in asymmetric entries of rows/cols
    associated with boundary cells. To generate a symmetric interaction matrix, perform (C + C.T)!=0

    :param n_c: Number of cells **np.int64**
    :param CV_matrix: Cell-vertex matrix representation of triangulation (n_c x n_v x 3)
    :return:
    """
    C = np.zeros((n_c, n_c), dtype=np.float32)
    for i in range(3):
        C += np.asfortranarray(CV_matrix[:, :, i]) @ np.asfortranarray(CV_matrix[:, :, np.mod(i + 2, 3)].T)
        C += np.asfortranarray(CV_matrix[:, :, i]) @ np.asfortranarray(CV_matrix[:, :, np.mod(i + 1, 3)].T)
    C = (C != 0).astype(np.int32)
    return C

@jit(nopython=True,cache=True)
def get_F_bend(n_c,CV_matrix,n_C,x,zeta):
    """
    Get the spatial differential of the bending energy i.e. the bending force -- F_bend

    If E_bend = Sum_i{zeta_i * cos(theta_i))
    where cos(theta_i) = (r_{ji}•r_{ki})/(|r_{ji}||r_{ki}|)

    Then F_bend|_{cell_i} = - \partial E_bend / \partial r_i

    Relies on the function **dcosthetadr** which specifies:
        \partial cos(theta_i) / \partial r_i
        \partial cos(theta_i) / \partial r_j
        \partial cos(theta_i) / \partial r_k

    :param n_c: Number of cells including boundary cells **np.int64**
    :param CV_matrix: Cell-vertex matrix representation of triangulation (n_c x n_v x 3)
    :param n_C: Number of cells exclusing boundary cells **np.int64**
    :param x: Cell centroids (n_c x 2)
    :param zeta: Coefficient of bending energy **np.float64**
    :return:
    """
    C_b = np.zeros((n_c-n_C, n_c-n_C), dtype=np.float64)
    for i in range(3):
        C_b += np.asfortranarray(CV_matrix[n_C:, :, i]) @ np.asfortranarray(CV_matrix[n_C:, :, np.mod(i + 2, 3)].T)

    x_i = x[n_C:]
    x_j = np.asfortranarray(C_b)@x_i
    x_k = np.asfortranarray(C_b.T)@x_i

    dC_ri, dC_rj,dC_rk = dcosthetadr(x_i,x_j,x_k)

    F_b = zeta *(dC_ri
                  + np.asfortranarray(C_b) @ np.asfortranarray(dC_rj)
                  + np.asfortranarray(C_b.T) @ np.asfortranarray(dC_rk))

    F_bend = np.zeros((n_c, 2))
    F_bend[n_C:] = F_b

    return F_bend

@jit(nopython=True,cache=True)
def dcosthetadr(ri,rj,rk):
    """
    If cos(theta_i) = (r_{ji}•r_{ki})/(|r_{ji}||r_{ki}|)

    then this function calculates:
        \partial cos(theta_i) / \partial r_i (denoted in short hand dC_ri)
        \partial cos(theta_i) / \partial r_j (denoted in short hand dC_rj)
        \partial cos(theta_i) / \partial r_k (denoted in short hand dC_rk)

    :param ri: Array of positions of boundary cell i (n_c - n_C x 2)
    :param rj: Array positions of neighbours of i (j) (n_c - n_C x 2)
    :param rk: Array of positions of the other neighbour of i (k) (n_c - n_C x 2)
    :return: dC_ri, dC_rj,dC_rk
    """
    rij = ri-rj
    rjk = rj - rk
    rki = rk - ri
    nij = np.sqrt(rij[:,0]** 2 + rij[:,1]**2)
    njk = np.sqrt(rjk[:,0]** 2 + rjk[:,1]**2)
    nki = np.sqrt(rki[:,0]** 2 + rki[:,1]**2)

    dC_ri = (1/(2*nij*nki))*((rij.T/nij**2)*(nki**2 - njk**2 - nki**2)
                             - (rki.T/nki**2)*(nij**2 - njk**2 - nki**2))
    dC_rj = (1/(2*nij**3 * nki)*((rij.T * (njk**2 - nki**2)) + (nij**2 * (rjk.T - rki.T))))
    dC_rk = (1/(2*nki**3 * nij)* ((rki.T * (-njk**2 + nij**2) + (nki**2 * (rij.T - rjk.T)))))
    return dC_ri.T, dC_rj.T,dC_rk.T


@jit(nopython=True)
def disp_from_min_dist(x,X,L):
    disp = np.mod(X - x + L/2,L)-L/2
    dist = np.sqrt(disp[:,0]**2 + disp[:,1]**2)
    mindisti = np.argmin(dist)
    mindist = dist[mindisti]
    return disp[mindisti]/mindist, mindist
#
#
# @jit(nopython=True)
# def get_mobile_dir(tris,c_types,v_neighbours,neighbours,L,vs,mobile_i,t1_type="forward"):
#     """
#     For a single A cell moving in (forward) or out (reverse) of the main A aggregate
#
#     Mobile_i must be of type "0" (A)
#
#     :param tris:
#     :param mobile_i:
#     :param t1_type:
#     :return:
#     """
#     if t1_type == "forward":
#         tot = 2
#     if t1_type == "reverse":
#         tot = 3
#     m_tri_mask = (tris[:,0] == mobile_i)+(tris[:,1] == mobile_i)+(tris[:,2] == mobile_i)
#     m_tris_i = np.arange(tris.shape[0])[m_tri_mask]
#     direc_mat,dist_mat = np.ones((m_tris_i.size,2),dtype=np.float64)*np.nan,np.ones(m_tris_i.size,dtype=np.float64)*np.nan
#     n_a_neighbours = np.sum(1 - c_types[tris[m_tris_i].ravel()].reshape(-1,3)) - m_tris_i.size ##not a real meausre, but a proxy for number of "A" type neighbours
#     if (n_a_neighbours==0)*(t1_type == "forward") + (n_a_neighbours>0)*(t1_type=="reverse"):
#     # if (n_a_neighbours<2)*(t1_type == "forward") + (n_a_neighbours>0)*(t1_type=="reverse"):
#         ###small modification here such that minimally 2 A cell neighbours are required. Note the double counting required b.c. of the triangulation
#         for j, i in enumerate(m_tris_i):
#             tri = tris[i]
#             v = vs[i]
#             if c_types[tri].sum() ==2: #ensure the two neighbouring cells are of type "1" (B)
#                 neigh_v = neighbours[i]
#                 neighbour_tris = tris[v_neighbours[i]]
#                 neigh_mask = c_types[neighbour_tris.ravel()].reshape(-1,3)
#                 neigh_mask = (neigh_mask[:,0]+neigh_mask[:,1]+neigh_mask[:,2]) == tot
#                 non_self_mask = neighbour_tris != mobile_i
#                 non_self_mask = non_self_mask[:,0]*non_self_mask[:,1]*non_self_mask[:,2]
#                 neigh_v = neigh_v[neigh_mask*non_self_mask]
#                 # neigh_v_size = neigh_v.size
#                 if neigh_v.size == 2:
#                     disp = np.mod(neigh_v.ravel() - v + L/2,L) - L/2
#                     dist = np.sqrt(disp[0]**2 + disp[1]**2)
#                     direc = disp/dist
#                     direc_mat[j], dist_mat[j] = direc, dist
#                 if neigh_v.size > 2:
#                     direc,dist = disp_from_min_dist(v,neigh_v,L)
#                     direc_mat[j],dist_mat[j] = direc,dist
#         if np.isnan(dist_mat).all():
#             direc = np.array((0.0,0.0))
#         else:
#             real_mask = ~np.isnan(dist_mat)
#             dist_mat = dist_mat[real_mask]
#             direc_mat = direc_mat[real_mask]
#             direc = direc_mat[np.argmin(dist_mat)]
#
#     else:
#         direc = np.array((0.0,0.0))
#     return direc


#
#
# # @jit(nopython=True)
# def get_mobile_dir(tris,c_types,v_neighbours,neighbours,L,vs,mobile_i,t1_type="forward"):
#     """
#     For a single A cell moving in (forward) or out (reverse) of the main A aggregate
#
#     Mobile_i must be of type "0" (A)
#
#     :param tris:
#     :param mobile_i:
#     :param t1_type:
#     :return:
#     """
#     if t1_type == "forward":
#         oc_type = 0
#     if t1_type == "reverse":
#         oc_type = 1
#     m_tri_mask = (tris[:,0] == mobile_i)+(tris[:,1] == mobile_i)+(tris[:,2] == mobile_i)
#     m_tris_i = np.arange(tris.shape[0])[m_tri_mask]
#     direc_mat,dist_mat = np.ones((m_tris_i.size,2),dtype=np.float64)*np.nan,np.ones(m_tris_i.size,dtype=np.float64)*np.nan
#     n_a_neighbours = np.sum(1 - c_types[tris[m_tris_i].ravel()].reshape(-1,3)) - m_tris_i.size ##not a real meausre, but a proxy for number of "A" type neighbours
#     for j, i in enumerate(m_tris_i):
#         tri = tris[i]
#         v = vs[i]
#         if c_types[tri].sum() ==2: #ensure the two neighbouring cells are of type "1" (B)
#             neigh_v = neighbours[i]
#             neighbour_tris = tris[v_neighbours[i]]
#             opposite_mask = ~((neighbour_tris==tri[0])+(neighbour_tris==tri[1])+(neighbour_tris==tri[2]))
#             opposite_cells = neighbour_tris.ravel()[opposite_mask.ravel()]
#             oc_types = c_types[opposite_cells]
#             neigh_v = neigh_v[oc_types == oc_type]
#             if neigh_v.size == 2:
#                 disp = np.mod(neigh_v.ravel() - v + L/2,L) - L/2
#                 dist = np.sqrt(disp[0]**2 + disp[1]**2)
#                 direc = disp/dist
#                 direc_mat[j], dist_mat[j] = direc, dist
#             if neigh_v.size > 2:
#                 direc,dist = disp_from_min_dist(v,neigh_v,L)
#                 direc_mat[j],dist_mat[j] = direc,dist
#     if np.isnan(dist_mat).all():
#         direc = np.array((0.0,0.0))
#     else:
#         real_mask = ~np.isnan(dist_mat)
#         dist_mat = dist_mat[real_mask]
#         direc_mat = direc_mat[real_mask]
#         direc = direc_mat[np.argmin(dist_mat)]
#
#     return direc
#
#
#
# # @jit(nopython=True)
# def get_t1_dir(tris,c_types,v_neighbours,neighbours,L,vs,mobile_i,t1_type="forward"):
#     """
#     For a single A cell moving in (forward) or out (reverse) of the main A aggregate
#
#     Mobile_i must be of type "0" (A)
#
#     :param tris:
#     :param mobile_i:
#     :param t1_type:
#     :return:
#     """
#     if t1_type == "forward":
#         corr_oc_type = 0
#         nB1,nB2 = 1,2
#     if t1_type == "reverse":
#         corr_oc_type = 1
#         nB1,nB2 = 2,1
#
#     m_tri_mask = (tris[:,0] == mobile_i)+(tris[:,1] == mobile_i)+(tris[:,2] == mobile_i)
#     m_tris_i = np.arange(tris.shape[0])[m_tri_mask]
#     adjacent_cells = np.unique(tris[m_tri_mask])
#     # dist, direc, clls = np.inf,np.array((0.0,0.0)),np.array((0,1,2,3))
#     dist, direc, clls,tri_i,tri_k = np.inf,np.array((0.0,0.0)),np.array((0,1,2,3)),-1,-1
#
#     for j, i in enumerate(m_tris_i):
#         tri = tris[i]
#         v = vs[i]
#         if c_types[tri].sum() ==nB1: #ensure the two neighbouring cells are of type "1" (B)
#             neigh_v = neighbours[i]
#             neighbour_tris = tris[v_neighbours[i]]
#             opposite_mask = ~((neighbour_tris==tri[0])+(neighbour_tris==tri[1])+(neighbour_tris==tri[2]))
#             opposite_cells = neighbour_tris.ravel()[opposite_mask.ravel()]
#             for k, o_c in enumerate(opposite_cells):
#                 oc_type = c_types[o_c]
#                 if (oc_type == corr_oc_type)*(o_c not in adjacent_cells):
#                     disp_k = np.mod(neigh_v[k] - v + L / 2, L) - L / 2
#                     dist_k = np.sqrt(disp_k[0] ** 2 + disp_k[1] ** 2)
#                     if dist_k < dist:
#                         dist = dist_k
#                         direc = disp_k / dist_k
#                         clls = np.roll(tri,-np.nonzero(tri==mobile_i)[0][0])
#                         clls = np.append(clls,o_c)
#                         tri_i = i
#                         tri_k = k
#     if np.isinf(dist):
#         for j, i in enumerate(m_tris_i):
#             tri = tris[i]
#             v = vs[i]
#             if c_types[tri].sum() ==nB2: #ensure the two neighbouring cells are of type "1" (B)
#                 neigh_v = neighbours[i]
#                 neighbour_tris = tris[v_neighbours[i]]
#                 opposite_mask = ~((neighbour_tris==tri[0])+(neighbour_tris==tri[1])+(neighbour_tris==tri[2]))
#                 opposite_cells = neighbour_tris.ravel()[opposite_mask.ravel()]
#                 for k, o_c in enumerate(opposite_cells):
#                     oc_type = c_types[o_c]
#                     if (oc_type == corr_oc_type)*(o_c not in adjacent_cells):
#                         disp_k = np.mod(neigh_v[k] - v + L / 2, L) - L / 2
#                         dist_k = np.sqrt(disp_k[0] ** 2 + disp_k[1] ** 2)
#                         if dist_k < dist:
#                             dist = dist_k
#                             direc = disp_k / dist_k
#                             clls = np.roll(tri,-np.nonzero(tri==mobile_i)[0][0])
#                             clls = np.append(clls,o_c)
#                             tri_i = i
#                             tri_k = k
#     direc_cw = np.array((direc[1],-direc[0]))
#     direcs = np.stack((direc,direc_cw,-direc_cw,-direc))
#     return direcs,clls,tri_i,tri_k
#
#


@jit(nopython=True)
def get_degree(c_types,tris,n_c,CV_matrix):
    tri_types = c_types.ravel()[tris.ravel()].reshape(-1,3)
    local_degree = (tri_types==roll_forward(tri_types))*1.0 + (tri_types==roll_reverse(tri_types))*1.0
    degree = np.zeros(n_c)
    for i in range(3):
        degree += np.asfortranarray(CV_matrix[:,:,i])@np.asfortranarray(local_degree[:,i]/2)
    return degree.astype(np.int64)

# @jit(nopython=True)
def get_t1_dir(tris,c_types,v_neighbours,neighbours,L,vs,mobile_i,n_c,CV_matrix,t1_type="forward"):
    """
    For a single A cell moving in (forward) or out (reverse) of the main A aggregate

    Mobile_i must be of type "0" (A)

    :param tris:
    :param mobile_i:
    :param t1_type:
    :return:
    """

    m_tri_mask = (tris[:,0] == mobile_i)+(tris[:,1] == mobile_i)+(tris[:,2] == mobile_i)
    m_tris_i = np.arange(tris.shape[0])[m_tri_mask]
    neighs = np.unique(tris[m_tris_i].ravel())
    n_b_neighs = c_types[neighs].sum()
    n_a_neighs = (1-c_types[neighs]).sum() - 1
    tri_i, tri_k = -1,-1
    mi_degree = np.nan
    if (t1_type == "forward")*(n_a_neighs <2) + (t1_type=="reverse")*(n_a_neighs>0):
        degree = get_degree(c_types,tris,n_c,CV_matrix)
        mi_degree = degree[mobile_i]
        quartets = np.zeros((m_tris_i.size,3,4),dtype=np.int64)
        quartet_types = np.zeros((m_tris_i.size,3,4),dtype=np.int64)
        quartet_degree = np.zeros((m_tris_i.size,3,4),dtype=np.int64)

        for j, i in enumerate(m_tris_i):
            tri = tris[i]
            tri_types = c_types[tri]
            tri_degree = degree[tri]
            neighs_i = v_neighbours[i]
            neighbour_tris = tris[neighs_i]
            opposite_mask = ~((neighbour_tris==tri[0])+(neighbour_tris==tri[1])+(neighbour_tris==tri[2]))
            opposite_cells = neighbour_tris.ravel()[opposite_mask.ravel()]
            o_c_types = c_types[opposite_cells]
            o_c_degree = degree[opposite_cells]
            for k in range(3):
                quartets[j,k,:3] = np.roll(tri,-k)
                quartets[j,k,3] = opposite_cells[k]
                quartet_types[j, k, :3] = np.roll(tri_types, -k)
                quartet_types[j, k, 3] = o_c_types[k]
                quartet_degree[j, k, :3] = np.roll(tri_degree, -k)
                quartet_degree[j, k, 3] = o_c_degree[k]

        confs =  np.array(((1,1,0,1),(1,0,1,1),(0,1,1,1),(1,0,0,1)))
        changes = np.array(((1,0,0,1),(1,0,0,1),(0,-1,-1,0),(1,-1,-1,1)))

        t_changes = np.zeros_like(quartet_types)
        for conf,change in zip(confs,changes):
            mask =((quartet_types == conf).all(axis=2)+(quartet_types == (1-conf)).all(axis=2))
            # mask = mask[:,:,0] * mask[:,:,1] * mask[:,:,2] * mask[:,:,3]
            for i in range(4):
                t_changes[:,:,i] += mask * change[i]
        new_degree = t_changes + quartet_degree


        if t1_type == "forward":
            new_degree_not_0 = (new_degree != 0)
            new_degree_not_0 = new_degree_not_0[:,:,0]*new_degree_not_0[:,:,1]*new_degree_not_0[:,:,2]*new_degree_not_0[:,:,3]
            mobile_i_increase = t_changes.ravel()[(quartets == mobile_i).ravel()] * (new_degree_not_0.ravel())
            max_mi_increase_mask = mobile_i_increase == mobile_i_increase.max()
            pos = np.nonzero(max_mi_increase_mask)[0]
            dist = np.inf
            if pos.size!=0:
                tri_is,tri_ks = m_tris_i[(pos/3).astype(np.int64)],pos%3
                for tri_i_,tri_k_ in zip(tri_is,tri_ks):
                    v = vs[tri_i]
                    neigh_v = neighbours[tri_i,tri_k]
                    disp = np.mod(v - neigh_v+L/2,L)-L/2
                    dist_ = np.sqrt(disp[0]**2 + disp[1]**2)
                    if dist_ < dist:
                        tri_i,tri_k = tri_i_,tri_k_
                        dist = dist_


        if t1_type == "reverse":
            mobile_i_increase = t_changes.ravel()[(quartets == mobile_i).ravel()]
            min_mi_increase_mask = mobile_i_increase == mobile_i_increase.min()
            pos = np.nonzero(min_mi_increase_mask)[0]
            dist = np.inf
            if pos.size!=0:
                tri_is,tri_ks = m_tris_i[(pos/3).astype(np.int64)],pos%3
                for tri_i_,tri_k_ in zip(tri_is,tri_ks):
                    v = vs[tri_i]
                    neigh_v = neighbours[tri_i,tri_k]
                    disp = np.mod(v - neigh_v+L/2,L)-L/2
                    dist_ = np.sqrt(disp[0]**2 + disp[1]**2)
                    if dist_ < dist:
                        tri_i,tri_k = tri_i_,tri_k_
                        dist = dist_

    complete = False
    if t1_type == "forward":
        if tri_i ==-1:
            complete = True
        if mi_degree > 0:
            complete = True
    if t1_type == "reverse":
        if tri_i ==-1:
            complete = True
        if mi_degree == 0:
            complete = True

    return tri_i,tri_k,complete



@jit(nopython=True,cache=True)
def dhdr_periodic_single(rijk_,v,L):
    """
    Same as **dhdr** apart from accounts for periodic triangulation.

    Calculates ∂h_j/dr_i the Jacobian for all cells in each triangulation

    Last two dims: ((dhx/drx,dhx/dry),(dhy/drx,dhy/dry))

    These are lifted from Mathematica

    :param rijk_: (n_v x 3 x 2) np.float32 array of cell centroid positions for each cell in each triangulation (first two dims follow order of triangulation)
    :param vs: (n_v x 2) np.float32 array of vertex positions, corresponding to each triangle in the triangulation
    :param L: Domain size (np.float32)
    :return: Jacobian for each cell of each triangulation (n_v x 3 x 2 x 2) np.float32 array (where the first 2 dims follow the order of the triangulation.
    """
    rijk = np.empty_like(rijk_)
    for i in range(3):
        rijk[i] = np.remainder(rijk_[i] - v + L/2,L) - L/2

    DHDR = np.empty(rijk.shape + (2,))
    for i in range(3):
        ax,ay = rijk[np.mod(i,3),0],rijk[np.mod(i,3),1]
        bx, by = rijk[np.mod(i+1,3), 0], rijk[np.mod(i+1,3), 1]
        cx, cy = rijk[np.mod(i+2,3), 0], rijk[np.mod(i+2,3), 1]
        #dhx/drx
        DHDR[i, 0, 0] = (ax * (by - cy)) / ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy)) - ((by - cy) * (
                (ax ** 2 + ay ** 2) * (by - cy) + (bx ** 2 + by ** 2) * (-ay + cy) + (ay - by) * (
                cx ** 2 + cy ** 2))) / (2. * ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy)) ** 2)

        #dhy/drx
        DHDR[i, 0,1] = (bx ** 2 + by ** 2 - cx ** 2 + 2 * ax * (-bx + cx) - cy ** 2) / (
                    2. * ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy))) - ((by - cy) * (
                    (bx ** 2 + by ** 2) * (ax - cx) + (ax ** 2 + ay ** 2) * (-bx + cx) + (-ax + bx) * (
                        cx ** 2 + cy ** 2))) / (2. * ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy)) ** 2)

        #dhx/dry
        DHDR[i, 1, 0] = (-bx ** 2 - by ** 2 + cx ** 2 + 2 * ay * (by - cy) + cy ** 2) / (
                2. * ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy))) - ((-bx + cx) * (
                (ax ** 2 + ay ** 2) * (by - cy) + (bx ** 2 + by ** 2) * (-ay + cy) + (ay - by) * (
                cx ** 2 + cy ** 2))) / (2. * ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy)) ** 2)

        #dhy/dry
        DHDR[i, 1,1] = (ay * (-bx + cx)) / ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy)) - ((-bx + cx) * (
                    (bx ** 2 + by ** 2) * (ax - cx) + (ax ** 2 + ay ** 2) * (-bx + cx) + (-ax + bx) * (
                        cx ** 2 + cy ** 2))) / (2. * ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy)) ** 2)


    return DHDR



def get_M(tri_i,tri_j,vs,x,tris,L):
    v = vs[tri_i]
    neigh_v = vs[tri_j]
    direc = neigh_v - v
    direc /= np.linalg.norm(direc)
    DHDR = dhdr_periodic_single(x[tris[tri_i]],v,L)
    M = np.zeros((2, 2, 3))
    for i in range(3):
        for j in range(3):
            for Fdim in range(2):
                M[:, Fdim, i] += DHDR[i, Fdim] * direc
    M = M[0] + M[1]
    return M


@jit(nopython=True, cache=True)
def _beta_t(beta_max, beta_min, tau, t):
    if t <= tau:
        return beta_min + (beta_max - beta_min) * t / tau
    else:
        return beta_max