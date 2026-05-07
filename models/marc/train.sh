#!/usr/bin/env bash
# Training Script for DINOv2-LoRA Multi-Loss Forgery Detection
# Usage:
#   bash train.sh -g 0 -a 4 -n "experiment_name"

set -euo pipefail

# ========= User Configuration =========
# 6 folders
REAL_PATH="/workspace/user-data/datasets/DDAtraining/NIPS/0_real"
REAL_PROCESSED_PATH="/workspace/user-data/datasets/DDAtraining/NIPS/camera_post/0_real"
DM_PATH="/workspace/user-data/datasets/DDAtraining/NIPS/1_vae_fake"
DM_PROCESSED_PATH="/workspace/user-data/datasets/DDAtraining/NIPS/camera_post/1_vae_fake"
AR_PATH="/workspace/user-data/datasets/DDAtraining/NIPS/1_ar_fake"
AR_PROCESSED_PATH="/workspace/user-data/datasets/DDAtraining/NIPS/camera_post/1_ar_fake"

CHECKPOINTS_DIR="./checkpoints"

# ========= Experiment Settings =========
BACKBONE_NAME="dinov2_vitl14"

LORA_RANK=8
LORA_ALPHA=1.0

OPTIM="adam"
NITER=5
BATCH_SIZE=8
ACCUM_STEPS=4
IMAGE_SIZE=336
CROP_SIZE=336
LEARNING_RATE=1e-4
WEIGHT_DECAY=0.0

NUM_THREADS=8
SEED=2026

# ========= Loss Weights =========
# 对齐新版 trainer.py / options.py
LAMBDA_BIN=1.0
LAMBDA_SRC=0.3
LAMBDA_PAIR=0.4
LAMBDA_RF=0.25
LAMBDA_DMAR=0.1
LAMBDA_CON_BIN=0.05
LAMBDA_CON_SRC=0.01

# ========= Loss Schedule =========
DMAR_START_EPOCH=1
DMAR_WARMUP_EPOCHS=2

CONTRAST_START_EPOCH=0
CONTRAST_FULL_EPOCH=1
CON_BIN_DECAY_EPOCH=2
CON_BIN_DECAY_RATIO=0.3
CON_SRC_OFF_EPOCH=2

# ========= Loss Switches =========
USE_LOSS_BIN=true
USE_LOSS_SRC=true
USE_LOSS_PAIR=true
USE_LOSS_RF=true
USE_LOSS_DMAR=true
USE_LOSS_CON_BIN=true
USE_LOSS_CON_SRC=false

# ========= Loss Hyperparameters =========
MARGIN_REAL_FAKE=0.3
MARGIN_DM_AR=0.15
TEMPERATURE=0.10

# ========= Projection Head =========
PROJ_DIM=256
PROJ_HIDDEN_DIM=512
DROPOUT=0.0

# ========= Sync Augmentation Settings =========
USE_SYNC_AUG=1
USE_RANDOM_CROP=0
SYNC_HFLIP_PROB=0.5
SYNC_BRIGHTNESS=0.1
SYNC_CONTRAST=0.1
SYNC_SATURATION=0.05
USE_SYNC_GAMMA=1
SYNC_GAMMA_MIN=0.95
SYNC_GAMMA_MAX=1.05
SYNC_BLUR_PROB=0.1
SYNC_BLUR_MIN=0.3
SYNC_BLUR_MAX=0.8

# ========= Save / Print =========
PRINT_FREQ=50
SAVE_LATEST_FREQ=400
SAVE_EPOCH_FREQ=1

# ========= Resume =========
RESUME_PATH=""
RESUME_STRICT=0
RESUME_OPTIMIZER=0
RESUME_SCHEDULER=0

# ========= DataLoader =========
SHUFFLE=1
DROP_LAST=0
PIN_MEMORY=1
STRICT_CHECK=1

# ========= Command Line Arguments =========
GPU_ID=0
EXP_SUFFIX=""

while getopts ":g:a:n:r:" opt; do
  case $opt in
    g) GPU_ID="$OPTARG" ;;
    a) ACCUM_STEPS="$OPTARG" ;;
    n) EXP_SUFFIX="$OPTARG" ;;
    r) RESUME_PATH="$OPTARG" ;;
    \?) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
    :) echo "Option -$OPTARG requires an argument." >&2; exit 1 ;;
  esac
done

# ========= Setup Flags =========
EXTRA_FLAGS=()

if [[ "${USE_SYNC_AUG}" -eq 1 ]]; then
  EXTRA_FLAGS+=(--use_sync_aug)
fi

if [[ "${USE_RANDOM_CROP}" -eq 1 ]]; then
  EXTRA_FLAGS+=(--use_random_crop)
fi

if [[ "${USE_SYNC_GAMMA}" -eq 1 ]]; then
  EXTRA_FLAGS+=(--use_sync_gamma)
fi

if [[ "${SHUFFLE}" -eq 1 ]]; then
  EXTRA_FLAGS+=(--shuffle)
fi

if [[ "${DROP_LAST}" -eq 1 ]]; then
  EXTRA_FLAGS+=(--drop_last)
fi

if [[ "${PIN_MEMORY}" -eq 1 ]]; then
  EXTRA_FLAGS+=(--pin_memory)
fi

if [[ "${STRICT_CHECK}" -eq 1 ]]; then
  EXTRA_FLAGS+=(--strict_check)
fi

if [[ -n "${RESUME_PATH}" ]]; then
  EXTRA_FLAGS+=(--resume_path "${RESUME_PATH}")
fi

if [[ "${RESUME_STRICT}" -eq 1 ]]; then
  EXTRA_FLAGS+=(--resume_strict)
fi

if [[ "${RESUME_OPTIMIZER}" -eq 1 ]]; then
  EXTRA_FLAGS+=(--resume_optimizer)
fi

if [[ "${RESUME_SCHEDULER}" -eq 1 ]]; then
  EXTRA_FLAGS+=(--resume_scheduler)
fi

# ========= Experiment Name =========
EXP_NAME="DINOv2_${BACKBONE_NAME}_LoRA${LORA_RANK}_IMG${IMAGE_SIZE}_LR${LEARNING_RATE}_BS${BATCH_SIZE}_ACC${ACCUM_STEPS}"
if [[ -n "${EXP_SUFFIX}" ]]; then
  EXP_NAME="${EXP_NAME}_${EXP_SUFFIX}"
fi

echo "============================================================"
echo ">>> Starting Training: ${EXP_NAME}"
echo ">>> GPU: ${GPU_ID}"
echo ">>> Backbone: ${BACKBONE_NAME}"
echo ">>> Batch Size: ${BATCH_SIZE}"
echo ">>> Accumulation Steps: ${ACCUM_STEPS}"
echo ">>> Learning Rate: ${LEARNING_RATE}"
echo ">>> Checkpoints Dir: ${CHECKPOINTS_DIR}"
echo ">>> Loss Weights: bin=${LAMBDA_BIN}, src=${LAMBDA_SRC}, pair=${LAMBDA_PAIR}, rf=${LAMBDA_RF}, dmar=${LAMBDA_DMAR}, con_bin=${LAMBDA_CON_BIN}, con_src=${LAMBDA_CON_SRC}"
echo ">>> Contrast Schedule: start=${CONTRAST_START_EPOCH}, full=${CONTRAST_FULL_EPOCH}, con_bin_decay=${CON_BIN_DECAY_EPOCH}, con_bin_ratio=${CON_BIN_DECAY_RATIO}, con_src_off=${CON_SRC_OFF_EPOCH}"
if [[ -n "${RESUME_PATH}" ]]; then
  echo ">>> Resume From: ${RESUME_PATH}"
fi
echo "============================================================"

python train.py \
  --gpu_ids "${GPU_ID}" \
  --name "${EXP_NAME}" \
  --checkpoints_dir "${CHECKPOINTS_DIR}" \
  --seed "${SEED}" \
  \
  --real_dir "${REAL_PATH}" \
  --real_processed_dir "${REAL_PROCESSED_PATH}" \
  --dm_dir "${DM_PATH}" \
  --dm_processed_dir "${DM_PROCESSED_PATH}" \
  --ar_dir "${AR_PATH}" \
  --ar_processed_dir "${AR_PROCESSED_PATH}" \
  \
  --num_threads "${NUM_THREADS}" \
  --batch_size "${BATCH_SIZE}" \
  --image_size "${IMAGE_SIZE}" \
  --crop_size "${CROP_SIZE}" \
  \
  --backbone_name "${BACKBONE_NAME}" \
  --lora_rank "${LORA_RANK}" \
  --lora_alpha "${LORA_ALPHA}" \
  --proj_dim "${PROJ_DIM}" \
  --proj_hidden_dim "${PROJ_HIDDEN_DIM}" \
  --dropout "${DROPOUT}" \
  \
  --lr "${LEARNING_RATE}" \
  --optim "${OPTIM}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --niter "${NITER}" \
  --accumulation_steps "${ACCUM_STEPS}" \
  \
  --lambda_bin "${LAMBDA_BIN}" \
  --lambda_src "${LAMBDA_SRC}" \
  --lambda_pair "${LAMBDA_PAIR}" \
  --lambda_rf "${LAMBDA_RF}" \
  --lambda_dmar "${LAMBDA_DMAR}" \
  --lambda_con_bin "${LAMBDA_CON_BIN}" \
  --lambda_con_src "${LAMBDA_CON_SRC}" \
  \
  --use_loss_bin "${USE_LOSS_BIN}" \
  --use_loss_src "${USE_LOSS_SRC}" \
  --use_loss_pair "${USE_LOSS_PAIR}" \
  --use_loss_rf "${USE_LOSS_RF}" \
  --use_loss_dmar "${USE_LOSS_DMAR}" \
  --use_loss_con_bin "${USE_LOSS_CON_BIN}" \
  --use_loss_con_src "${USE_LOSS_CON_SRC}" \
  \
  --dmar_start_epoch "${DMAR_START_EPOCH}" \
  --dmar_warmup_epochs "${DMAR_WARMUP_EPOCHS}" \
  --contrast_start_epoch "${CONTRAST_START_EPOCH}" \
  --contrast_full_epoch "${CONTRAST_FULL_EPOCH}" \
  --con_bin_decay_epoch "${CON_BIN_DECAY_EPOCH}" \
  --con_bin_decay_ratio "${CON_BIN_DECAY_RATIO}" \
  --con_src_off_epoch "${CON_SRC_OFF_EPOCH}" \
  \
  --margin_real_fake "${MARGIN_REAL_FAKE}" \
  --margin_dm_ar "${MARGIN_DM_AR}" \
  --temperature "${TEMPERATURE}" \
  \
  --sync_hflip_prob "${SYNC_HFLIP_PROB}" \
  --sync_brightness "${SYNC_BRIGHTNESS}" \
  --sync_contrast "${SYNC_CONTRAST}" \
  --sync_saturation "${SYNC_SATURATION}" \
  --sync_gamma_min "${SYNC_GAMMA_MIN}" \
  --sync_gamma_max "${SYNC_GAMMA_MAX}" \
  --sync_blur_prob "${SYNC_BLUR_PROB}" \
  --sync_blur_min "${SYNC_BLUR_MIN}" \
  --sync_blur_max "${SYNC_BLUR_MAX}" \
  \
  --print_freq "${PRINT_FREQ}" \
  --save_latest_freq "${SAVE_LATEST_FREQ}" \
  --save_epoch_freq "${SAVE_EPOCH_FREQ}" \
  \
  "${EXTRA_FLAGS[@]}"

echo ">>> Training finished. Weights saved to: ${CHECKPOINTS_DIR}/${EXP_NAME}"