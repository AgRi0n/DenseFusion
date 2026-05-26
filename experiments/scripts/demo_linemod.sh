#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0


OBJ=${1:-1}
IDX=${2:-0}

mkdir -p demo_out/demo/obj$(printf "%02d" $OBJ)

python3 ./tools/demo_linemod.py \
  --dataset_root ./datasets/linemod/Linemod_preprocessed \
  --model        trained_checkpoints/linemod/pose_model_9_0.01310166542980859.pth \
  --refine_model trained_checkpoints/linemod/pose_refine_model_493_0.006761023565178073.pth \
  --obj        $OBJ \
  --idx        $IDX \
  --output_dir demo_out

chmod -R a+rw demo_out
