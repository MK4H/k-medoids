#!/usr/bin/env bash

set -eu -o pipefail

# With 16 cpus per task, only volta04 will not be fully utilized
# But the bottleneck should mostly be the GPU processing, so additional
# CPUs given on volta04 would not help us in any way
# { time srun -p volta-hp --gpus-per-task=1 --cpus-per-task=16 --mem-per-gpu=64G --ntasks=11 ./parallel.sh ; } > std.out 2> std.err



{ time srun -p volta-hp --gpus-per-task=1 --cpus-per-task=16 --mem-per-gpu=64G --ntasks=11 \
    ./parallel /mnt/home/_teaching/advpara/final-kmedoids/data/aloi.bsf ; } > std.out 2> std.err
