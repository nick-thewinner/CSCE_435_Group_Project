#!/bin/bash
# 1 node
# 2 node 
# num_proc=(64)
# 4 node
# num_proc=(128)
# 8 node
# num_proc=(256)
num_vals=(64 128 256 512 1024)
for v in "${num_vals[@]}"
do 
    sbatch bitonic.grace_job $v 1024 4
done