"""
SPV with boundaries for studying mammary gland organoids
"""

"""
1. Install the packages. This depends on the module 'voronoi_model_periodic' which has a bunch of extra stuff. Reasonably well documented, but ask me if you have qs
"""

from voronoi_model.voronoi_model_periodic import *
import numpy as np


"""
2. Set up your tissue 
"""

vor = Tissue() #Start an instance of the class
vor.make_init_boundary(L=22,r=0.15) #Make your cells. This initialises cells by making a hexagonal grid within a square box (LxL), then cropping to a circle with radius (r*L). Cell area is fixed, so play around with these to adjust the number of cells. You can also over-ride, see the module or ask me

"""
3. Set interactions. 

W is the cell-cell interaction matrix. Positive values indicate membranes under tension. W can be positive or negative. 
Generic adhesion is accounted for in the p0 term. So when W=0 for all cells, reverts back to the original SPV model. 

I use alpha to scale this matrix. But do what you want. 

Wb is the cell-boundary interaction vector. Again, higher values = more tension and less adhesion. 

b_extra guesses the number of total cells (including ghost cells). Given cell number is dynamically updating, better to err on the side of caution to avoid explosion
"""

alpha = 0.03
vor.set_interaction_boundary(W = alpha*np.array([[0, 1], [1, 0]]),pE=0.55,Wb=[0.3,0.05],b_extra = 3)

"""
4. Set up the triangulation. Need to run this to make the following work
"""

vor._triangulate(vor.x0) #This builds the triangulation and neighbourhood indices from scratch
vor.triangulate(vor.x0) #This saves a few extra things so that the triangulation can be updated. Can explain if you're interested. This is what the simulate function goes on to use
vor.assign_vertices() #This builds a cell-vertex matrix. Used in some calculations
vor.x0 = vor.check_boundary(vor.x0) #This checks the boundary like in the PLOS Active Vertex paper. Adds or removes ghost particles.

"""
5. Set your boundary parameters.

Some options, that you can play with:

(i) Boundary particles are linked by springs with a 0 rest length. Gamma_bound sets the coefficient of this spring. THis should ensure circularity and minimise curvature

(ii) Boundary edges are under additional tension. Edges have a rest length l_b0 and a coefficient kappa_B.

I set one or the other to 0. If you decide one is better than another, then we can remove the other part from the code (for speed)

"""

vor.Gamma_bound = 5e-3
vor.kappa_B = 0#0.05 #<<-- this is the value I use when I set Gamma_bound to 0
vor.l_b0 = 0.3


"""
6. Set your tissue parameters
"""
p0 = 3.95 #This is the normalised perimeter/surface area thing that they use in Bi et al. Note that p0=3.81 is the magic number in periodic, homogeneous tissues. Bigger p0 = more generic adhesion. Lower p0 = higher tension and typically more hexagonal cells
vor.A0 = 0.9 #Target area
vor.P0 = p0*np.sqrt(vor.A0) #Calculate back the target perimeter

vor.v0 = 3e-1 #Instantaneous velocity -- "activity"
vor.Dr = 40 #Persistence timescale for motility
vor.kappa_A = 0.1 #Coefficient for deviation from target area in energy functional
vor.kappa_P = 0.05 #and for perimeter

##For these, see the Active Vertex PLOS paper. At very low distances, cells show spring-like repulsion.
vor.a = 0.2 #Maximal radius to engage in repulsion
vor.k = 2 #Coefficient of repulsion.


"""
7. Simulate!
"""
vor.set_t_span(dt = 0.025,tfin = 250) #Set your time course. dt is the time step. tfin is the final time point
vor.simulate_boundary(print_every=2000) #Print_every sets the number of iterations for which to print the percentage completed. This is for people like me who are too imaptient to wait for a blinking console


"""
8. Plot an animation. 

This will make a new directory "plots" and save an animation.
"""

vor.cols = "red","green","white" #Set your colours. First two correspond to the two cell types (or more) you investigate. The final one should be set to white to hide the boundary cells
vor.plot_scatter = False #If this is true, plots dots at the cell centroids, including ghosts

vor.plot_forces = False #If this is set to true, overlays a quiver plot of the forces. Not perfect at the moment, so can ignore
vor.animate(n_frames=50,an_type="boundary",tri=False) #n_frames sets the number of frames of the animation. an_type specifies that this is a boundary simulation. and if tri=True, then plots the triangulation on top (though this is super super slow)
