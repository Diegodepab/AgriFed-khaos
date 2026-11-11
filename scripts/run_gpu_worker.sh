#!/usr/bin/env bash
# ==============================================================================
# KHAOS Hub - Ephemeral Zero-Footprint GPU Training Worker
# ==============================================================================
# Runs a single high-precision training job on a GPU host inside an ephemeral
# container (--rm --gpus all) and outputs only the resulting .pt and .meta.json
# files, leaving ZERO permanent services or memory footprint on the GPU machine.
#
# Usage on GPU machine:
#   ./scripts/run_gpu_worker.sh \
#     --data-dir /path/to/dataset \
#     --slug frutossecos \
#     --arch dinov2_vits14 \
#     --img-size 336 \
#     --epochs 25
# ==============================================================================

set -euo pipefail

DATA_DIR=""
SLUG="custom-model"
ARCH="dinov2_vits14"
IMG_SIZE="224"
EPOCHS="25"
BATCH_SIZE="64"
AUG="in_the_wild"
TL_STRATEGY="partial"
LOSS="weighted_ce"
IMAGE_NAME="agrifed-khaos-agrifed:latest"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --slug) SLUG="$2"; shift 2 ;;
    --arch) ARCH="$2"; shift 2 ;;
    --img-size) IMG_SIZE="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --aug) AUG="$2"; shift 2 ;;
    --tl-strategy) TL_STRATEGY="$2"; shift 2 ;;
    --loss) LOSS="$2"; shift 2 ;;
    *) echo "Unknown parameter: $1"; exit 1 ;;
  esac
done

if [[ -z "$DATA_DIR" ]]; then
  echo "[!] Error: --data-dir is required."
  exit 1
fi

ABS_DATA="$(cd "$DATA_DIR" && pwd)"
OUT_DIR="$(pwd)/models"
mkdir -p "$OUT_DIR"

echo "================================================================="
echo " KHAOS HUB - EPHEMERAL GPU WORKER (ZERO PERMANENT FOOTPRINT)"
echo "================================================================="
echo " [*] Dataset:      $ABS_DATA"
echo " [*] Output Slug:  $SLUG"
echo " [*] Architecture: $ARCH (Resolution: ${IMG_SIZE}x${IMG_SIZE})"
echo " [*] Augmentation: $AUG | Strategy: $TL_STRATEGY | Loss: $LOSS"
echo "================================================================="

docker run --rm --gpus all \
  -v "$(pwd):/workspace" \
  -v "$ABS_DATA:/dataset:ro" \
  -w /workspace \
  "$IMAGE_NAME" \
  python -u scripts/train_base.py \
    --data-dir /dataset \
    --arch "$ARCH" \
    --img-size "$IMG_SIZE" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --tl-strategy "$TL_STRATEGY" \
    --loss "$LOSS" \
    --aug "$AUG" \
    --max-samples 0 \
    --device cuda \
    --output "models/${SLUG}_base.pt" \
    --metadata "models/${SLUG}.meta.json"

echo "[+] GPU Training finished and container destroyed (--rm)."
echo "[+] Artifacts ready in: models/${SLUG}_base.pt and models/${SLUG}.meta.json"

