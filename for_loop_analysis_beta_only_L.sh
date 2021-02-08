#!/bin/bash

((N = "$1"))

mkdir from_unsorted_beta_only/autocorr

for i in $(seq 0 $(($N-1)))
do
    sbatch run_analysis_beta_only_L.sh "$i" "$1" "$2"
done