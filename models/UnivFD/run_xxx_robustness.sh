#!/bin/bash

# =========================
# Basic Config
# =========================

TEST_SCRIPT="test_robust.py"
REAL_DIR="/workspace/user-data/datasets/ARForensics/ARForensics/val/0_real"
PARENT_FOLDER="/workspace/user-data/datasets/ARForensics/ARForensics"
SUBFOLDERS="Test_all"

CKPT="/workspace/user-data/models/ExDA_org/EXDA/checkpoints/nips_compare/model_epoch_best.pth"
ARCH="CLIP:ViT-L/14"

RESULT_ROOT="results_test/All_Robust"
BATCH_SIZE=64
NUM_WORKERS=4
REAL_NUM_PER_DIR=1000
KEY="ARForensics"

mkdir -p ${RESULT_ROOT}

COMMON_ARGS="--real_dirs ${REAL_DIR} \
--real_num_per_dir ${REAL_NUM_PER_DIR} \
--fake_num_per_dir ${REAL_NUM_PER_DIR} \
--parent_folder ${PARENT_FOLDER} \
--subfolders ${SUBFOLDERS} \
--ckpt ${CKPT} \
--arch ${ARCH} \
--batch_size ${BATCH_SIZE} \
--num_workers ${NUM_WORKERS} \
--append_csv "

# =========================
# Clean
# =========================

echo "Running clean test..."

python ${TEST_SCRIPT} \
  ${COMMON_ARGS} \
  --result_folder "${RESULT_ROOT}/clean" \
  --key "${KEY}_clean" \
  --robust_mode clean


# =========================
# JPEG Robustness
# =========================

for Q in 100 95 85 75 60 45
do
  echo "Running JPEG robustness test: quality=${Q}"

  python ${TEST_SCRIPT} \
    ${COMMON_ARGS} \
    --result_folder "${RESULT_ROOT}" \
    --key "${KEY}_jpeg_q${Q}" \
    --robust_mode jpeg \
    --jpeg_quality ${Q}
done



# =========================
# Blur Robustness
# =========================

for R in 0 0.5 0.75 1.0 1.25 1.5 1.75 2.0
do
  echo "Running blur robustness test: radius=${R}"

  python ${TEST_SCRIPT} \
    ${COMMON_ARGS} \
    --result_folder "${RESULT_ROOT}" \
    --key "${KEY}_blur_r${R}" \
    --robust_mode blur \
    --blur_radius ${R}
done

# =========================
# Resize-Recover Robustness
# =========================

for S in 1.0 0.9 0.75 0.5 0.25
do
  echo "Running resize-recover robustness test: scale=${S}"

  python ${TEST_SCRIPT} \
    ${COMMON_ARGS} \
    --result_folder "${RESULT_ROOT}" \
    --key "${KEY}_resize_s${S}" \
    --robust_mode resize \
    --resize_scale ${S}
done

echo "All robustness tests finished."
echo "Results are saved under: ${RESULT_ROOT}"