#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=2:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH -J "active_vertex"   # job name
#SBATCH --output=output_fromsorted.out
#SBATCH --error=output_fromsorted.out


source activate apical_domain

python voronoi_model/SPV_energy_barrier_cluster.py "$1" "$2" "$3"