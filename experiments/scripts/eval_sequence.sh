#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0


OBJ=${1:-1}
shift

# Add --verbose to print per-frame logs during the evaluation loop.

mkdir -p demo_out/sequence/obj$(printf "%02d" $OBJ)

python3 ./tools/eval_sequence.py \
  --dataset_root ./datasets/linemod/Linemod_preprocessed \
  --model        trained_checkpoints/linemod/pose_model_9_0.01310166542980859.pth \
  --refine_model trained_checkpoints/linemod/pose_refine_model_493_0.006761023565178073.pth \
  --obj          $OBJ \
  --output_dir   demo_out \
  "$@"

chmod -R a+rw demo_out
