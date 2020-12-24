#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=2:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH -J "active_vertex"   # job name
#SBATCH --output=output.out
#SBATCH --error=output.out


source activate apical_domain

python voronoi_model/SPV_from_file_cluster.py "$1" "$2" "$3"