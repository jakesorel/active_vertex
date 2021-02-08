#!/bin/bash

((N = "$1"*"$1"))

mkdir from_unsorted/het_swaps

for i in $(seq 0 $(($N-1)))
do
    sbatch run_het_swap.sh "$i" "$1" "$2"
done