#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0


RGB="${1:?Usage: $0 <rgb> <npz> <mask> <ply> [model] [refine_model]}"
NPZ="${2:?Usage: $0 <rgb> <npz> <mask> <ply> [model] [refine_model]}"
MASK="${3:?Usage: $0 <rgb> <npz> <mask> <ply> [model] [refine_model]}"
PLY="${4:?Usage: $0 <rgb> <npz> <mask> <ply> [model] [refine_model]}"
MODEL="${5:-trained_checkpoints/custom/pose_model_9_0.01310166542980859.pth}"
REFINE_MODEL="${6:-trained_checkpoints/custom/pose_refine_model_493_0.006761023565178073.pth}"

mkdir -p demo_out/custom

python3 ./tools/demo_custom.py \
  --num_obj      13 \
  --rgb          $RGB \
  --npz          $NPZ \
  --mask         $MASK \
  --ply          $PLY \
  --model        $MODEL \
  --refine_model $REFINE_MODEL \
  --output_dir   demo_out/custom

chmod -R a+rw demo_out/custom
