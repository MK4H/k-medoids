#!/usr/bin/env bash

set -eu -o pipefail

cuda-memcheck -- ./parallel --iterations 3 /mnt/home/_teaching/advpara/final-kmedoids/data/aloi_crop.bsf