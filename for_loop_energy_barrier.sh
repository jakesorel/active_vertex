#!/bin/bash

((N = "$1"*"$2"))

for i in $(seq 0 $(($N-1)))
do
    sbatch run_job_energy_barrier.sh "$i" "$1" "$2"
done