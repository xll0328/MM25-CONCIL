#!/usr/bin/env bash
# Example cleaned reproduction run.
# Before running, copy and edit:
#   cp src/utils/data_path.example.yml src/utils/data_path.yml
# and set valid dataset paths and checkpoint paths.

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$REPO_ROOT"

python reproduce_concil.py \
  -dataset cub \
  -base_ckpt /path/to/CUB.pth \
  -saved_dir results/concil_repro_cub \
  -batch_size 64 \
  -num_stages 2 \
  -class_ratio 0.5 \
  -concept_ratio 0.5 \
  -buffer_size 25000 \
  -gg1 500 \
  -gg2 1 \
  -seed 42
