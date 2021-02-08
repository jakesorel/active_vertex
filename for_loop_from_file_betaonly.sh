#!/bin/bash

((N = "$1"))

mkdir from_unsorted_beta_only
mkdir from_unsorted_beta_only/x_save from_unsorted_beta_only/tri_save from_unsorted_beta_only/c_types

for i in $(seq 0 $(($N-1)))
do
    sbatch run_job_fromfile_betaonly.sh "$i" "$1" "$2" "$3"
done