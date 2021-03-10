#!/bin/bash

#((N = "$1"*"$1"))
((N = "$1"))

#mkdir from_unsorted
#mkdir from_unsorted/x_save from_unsorted/tri_save from_unsorted/c_types

for i in $(seq 0 $(($N-1)))
do
    sbatch run_job_fromfile.sh "$i" "$1" "$2" "$3"
done