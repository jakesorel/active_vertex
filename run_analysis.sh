#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=0:30:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH -J "AV_jakecs"   # job name
#SBATCH --output=output_analysis_unsorted.out
#SBATCH --error=output_analysis_unsorted.out

source activate self-organisation-mouse

#python voronoi_model/cluster_analysis_file.py "$1" "$2" "$3"
#python voronoi_model/jamming_analysis_cluster.py "$1" "$2" "$3"
#python voronoi_model/cluster_analysis_file_dynamic_beta.py "$1" "$2" "$3"
#python voronoi_model/cluster_analysis_file_dynamic_beta_control.py "$1" "$2" "$3"
python voronoi_model/jamming_analysis_control_cluster.py "$1" "$2" "$3"
