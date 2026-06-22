#!/bin/bash

SAVE_JSON_DIR="./results/Robust/robust_json"
ROBUST_CSV="./results/Robust/robust_results.csv"
ROBUST_CACHE_ROOT="./results/Robust/robust_cache"

mkdir -p "${SAVE_JSON_DIR}"
mkdir -p "$(dirname "${ROBUST_CSV}")"
mkdir -p "${ROBUST_CACHE_ROOT}"

COMMON_ARGS="\
  --gpu_id 0 \
  --load ./checkpoints/Network_best.pth \
  --save_path ./results \
  --val_batchsize 64 \
  --test_num 1000 \
  --test_real_dirs /workspace/user-data/datasets/DDAtraining/COCO-SD-2/0_real \
  --test_fake_dirs /workspace/user-data/datasets/ARForensics/ARForensics/Test_all \
  --isPatch true \
  --img_height 256 \
  --bit_mode scaling \
  --patch_size 32 \
  --patch_mode max \
  --recursive \
"

CACHE_ARGS="\
  --robust_cache_root ${ROBUST_CACHE_ROOT} \
  --robust_sample_seed 2026 \
"

# Baseline
# python test_robust.py \
#   --name "baseline" \
#   ${COMMON_ARGS} \
#   ${CACHE_ARGS} \
#   --robust_mode none \
#   --robust_csv "${ROBUST_CSV}" \
#   --save_json "${SAVE_JSON_DIR}/baseline.json"

# JPEG
for Q in 100 95 85 75 60 45
do
  python test_robust.py \
    --name "jpeg_q${Q}" \
    ${COMMON_ARGS} \
    ${CACHE_ARGS} \
    --robust_mode jpeg \
    --jpeg_quality ${Q} \
    --robust_csv "${ROBUST_CSV}" \
    --save_json "${SAVE_JSON_DIR}/jpeg_q${Q}.json"
done

# Blur
for R in 0 0.5 0.75 1.0 1.25 1.5 1.75 2.0
do
  python test_robust.py \
    --name "blur_r${R}" \
    ${COMMON_ARGS} \
    ${CACHE_ARGS} \
    --robust_mode blur \
    --blur_radius ${R} \
    --robust_csv "${ROBUST_CSV}" \
    --save_json "${SAVE_JSON_DIR}/blur_r${R}.json"
done

# Resize-Recover
for S in 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0
do
  python test_robust.py \
    --name "resize_s${S}" \
    ${COMMON_ARGS} \
    ${CACHE_ARGS} \
    --robust_mode resize \
    --resize_scale ${S} \
    --robust_csv "${ROBUST_CSV}" \
    --save_json "${SAVE_JSON_DIR}/resize_s${S}.json"
done