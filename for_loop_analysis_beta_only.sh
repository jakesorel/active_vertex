#!/bin/bash

((N = "$1"))

mkdir from_unsorted_beta_only/analysis

for i in $(seq 0 $(($N-1)))
do
    sbatch run_analysis_beta_only.sh "$i" "$1" "$2"
done