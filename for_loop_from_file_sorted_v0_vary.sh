#!/bin/bash

((N = "$1"*"$1"))

mkdir from_sorted_v0_vary
mkdir from_sorted_v0_vary/x_save from_sorted_v0_vary/tri_save from_sorted_v0_vary/c_types


for i in $(seq 0 $(($N-1)))
do
    sbatch run_job_fromfile_sorted.sh "$i" "$1" "$2" "$3"
done