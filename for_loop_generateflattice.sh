#!/bin/bash

((N = "$1"))


for i in $(seq 0 $(($N-1)))
do
    sbatch run_job_generateflattice.sh "$i" "$1"
done