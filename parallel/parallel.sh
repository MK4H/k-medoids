#!/usr/bin/env bash

set -eu -o pipefail

cuda-memcheck -- ./parallel --iterations 1 /mnt/home/_teaching/advpara/final-kmedoids/data/aloi_crop.bsf