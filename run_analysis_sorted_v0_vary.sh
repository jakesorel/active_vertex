#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=0:30:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH -J "AV_jakecs"   # job name
#SBATCH --output=output_analysis_sorted.out
#SBATCH --error=output_analysis_sorted.out

source activate self-organisation-mouse

python voronoi_model/cluster_analysis_file_sorted_v0_vary.py "$1" "$2" "$3"