#!/bin/bash

((N = "$1"*"$1"))

#mkdir from_unsorted/analysis

for i in {40,132}
do
    sbatch run_analysis.sh "$i" "$1" "$2"
done