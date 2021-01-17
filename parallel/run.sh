#!/usr/bin/env bash

set -eu -o pipefail

# With 16 cpus per task, only volta04 will not be fully utilized
# But the bottleneck should mostly be the GPU processing, so additional
# CPUs given on volta04 would not help us in any way
srun -p volta-hp --gpus-per-task=1 --cpus-per-task=16 --mem-per-gpu=64G --ntasks=4 ./parallel.sh > std.out 2> std.err