#!/bin/bash

((N = "$1"*"$1"))

mkdir from_sorted
mkdir from_sorted/x_save from_sorted/tri_save from_sorted/c_types


for i in $(seq 0 $(($N-1)))
do
    sbatch run_job_fromfile_sorted.sh "$i" "$1" "$2" "$3"
done