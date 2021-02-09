#!/bin/bash

((N = "$1"*"$1"))

mkdir from_sorted_v0_vary/analysis

for i in $(seq 0 $(($N-1)))
do
    sbatch run_analysis_sorted_v0_vary.sh "$i" "$1" "$2"
done