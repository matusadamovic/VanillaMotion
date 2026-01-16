#!/usr/bin/env bash
set -Eeuo pipefail

COMFY_ROOT="/comfyui"
TMP_ROOT="/tmp"

# RunPod Cached Models root (nie Network Volume; cached models sú tu podľa docs)
HF_CACHE_ROOT="/runpod-volume/huggingface-cache/hub"  # :contentReference[oaicite:1]{index=1}

# Odporúčam nastaviť v endpoint env presne:
# HF_MODEL_NAME=matadamovic/vanillamotionbbotmodels
HF_MODEL_NAME="${HF_MODEL_NAME:-matadamovic/vanillamotionbotmodels}"

# Lokálne testovanie / fallback (na RunPod to nechaj prázdne)
MODEL_BUNDLE="${MODEL_BUNDLE_PATH:-}"

echo "BOOT: starting"
echo "BOOT: worker python=$(which python)"
python -c "import sys; print('BOOT: worker sys.executable=', sys.executable)"
python3 -c "import sys; print('BOOT: python3 sys.executable=', sys.executable)" || true
python3 -c "import transformers; print('BOOT: python3 transformers=', transformers.__version__)" || true
echo "BOOT: HF_CACHE_ROOT=$HF_CACHE_ROOT"
echo "BOOT: HF_MODEL_NAME=$HF_MODEL_NAME"
echo "BOOT: MODEL_BUNDLE_PATH=${MODEL_BUNDLE_PATH:-<empty>}"

# Best-effort cleanup
find "$TMP_ROOT" -maxdepth 1 -type d -name "job_*" -mtime +0 -exec rm -rf {} + || true

# Resolve cached snapshot (robust: wait + pick latest complete snapshot)
WAIT_FOR_CACHE_SECONDS="${WAIT_FOR_CACHE_SECONDS:-300}"
SLEEP_STEP_SECONDS="${SLEEP_STEP_SECONDS:-5}"

required_dirs=(checkpoints loras vae clip checkpoints_gguf)

is_complete_snapshot() {
  local snap="$1"
  [ -d "$snap" ] || return 1
  for d in "${required_dirs[@]}"; do
    [ -d "$snap/$d" ] || return 1
  done
  return 0
}

pick_latest_complete_snapshot() {
  local snapshots_dir="$1"
  local snap
  for snap in $(ls -1dt "$snapshots_dir"/* 2>/dev/null); do
    if is_complete_snapshot "$snap"; then
      echo "$snap"
      return 0
    fi
  done
  return 1
}

if [ -z "$MODEL_BUNDLE" ]; then
  cache_name="models--${HF_MODEL_NAME//\//--}"
  snapshots_dir="$HF_CACHE_ROOT/$cache_name/snapshots"

  echo "BOOT: snapshots_dir=$snapshots_dir"
  deadline=$(( $(date +%s) + WAIT_FOR_CACHE_SECONDS ))

  while [ "$(date +%s)" -lt "$deadline" ]; do
    if [ -d "$snapshots_dir" ]; then
      MODEL_BUNDLE="$(pick_latest_complete_snapshot "$snapshots_dir" || true)"
      if [ -n "$MODEL_BUNDLE" ]; then
        break
      fi
      echo "BOOT: snapshots exist but none complete yet; waiting..."
    else
      echo "BOOT: snapshots_dir not present yet; waiting..."
    fi
    sleep "$SLEEP_STEP_SECONDS"
  done
fi

if [ -z "$MODEL_BUNDLE" ] || [ ! -d "$MODEL_BUNDLE" ]; then
  echo "ERROR: MODEL_BUNDLE not found/complete within timeout (${WAIT_FOR_CACHE_SECONDS}s)."
  echo "HF_CACHE_ROOT=$HF_CACHE_ROOT"
  echo "HF_MODEL_NAME=$HF_MODEL_NAME"
  ls -lah /runpod-volume || true
  ls -lah "$HF_CACHE_ROOT" || true
  [ -n "${snapshots_dir:-}" ] && ls -lah "$snapshots_dir" || true
  exit 1
fi

echo "BOOT: Using MODEL_BUNDLE=$MODEL_BUNDLE"

echo "BOOT: Using MODEL_BUNDLE=$MODEL_BUNDLE"
ls -lah "$MODEL_BUNDLE" | head -n 200 || true

# Ensure Comfy paths exist
mkdir -p "$COMFY_ROOT/input/image" "$COMFY_ROOT/output" "$COMFY_ROOT/models"

# Fail fast ak chýba základná štruktúra repa
# for d in checkpoints loras vae clip checkpoints_gguf; do
#   if [ ! -d "$MODEL_BUNDLE/$d" ]; then
#     echo "ERROR: Chyba adresar v HF modeli: $MODEL_BUNDLE/$d"
#     echo "Obsah MODEL_BUNDLE (maxdepth 2):"
#     find "$MODEL_BUNDLE" -maxdepth 2 -type d | head -n 200 || true
#     exit 1
#   fi
# done

# FIX: Replace model dirs (do NOT nest symlinks inside existing directories)
for d in checkpoints loras vae clip controlnet upscale_models embeddings checkpoints_gguf unet; do
  rm -rf "$COMFY_ROOT/models/$d"
done

ln -s "$MODEL_BUNDLE/checkpoints"      "$COMFY_ROOT/models/checkpoints"
ln -s "$MODEL_BUNDLE/loras"            "$COMFY_ROOT/models/loras"
ln -s "$MODEL_BUNDLE/vae"              "$COMFY_ROOT/models/vae"
ln -s "$MODEL_BUNDLE/clip"             "$COMFY_ROOT/models/clip"
ln -s "$MODEL_BUNDLE/controlnet"       "$COMFY_ROOT/models/controlnet"
ln -s "$MODEL_BUNDLE/upscale_models"   "$COMFY_ROOT/models/upscale_models"
ln -s "$MODEL_BUNDLE/embeddings"       "$COMFY_ROOT/models/embeddings"
ln -s "$MODEL_BUNDLE/checkpoints_gguf" "$COMFY_ROOT/models/checkpoints_gguf"

# Build /models/unet as a real directory that contains both .safetensors and .gguf
rm -rf "$COMFY_ROOT/models/unet"
mkdir -p "$COMFY_ROOT/models/unet"

# Symlink all safetensors UNETs into unet/
for f in "$MODEL_BUNDLE/checkpoints"/*.safetensors; do
  [ -e "$f" ] || continue
  ln -s "$f" "$COMFY_ROOT/models/unet/$(basename "$f")"
done

# Symlink all GGUF UNETs into unet/
for f in "$MODEL_BUNDLE/checkpoints_gguf"/*.gguf; do
  [ -e "$f" ] || continue
  ln -s "$f" "$COMFY_ROOT/models/unet/$(basename "$f")"
done


# FIX: base image expects /runpod-volume/models/*
rm -rf /runpod-volume/models || true
ln -s /comfyui/models /runpod-volume/models

# Disable ComfyUI-Manager in headless workers (avoid registry fetches)
DISABLE_COMFYUI_MANAGER="${DISABLE_COMFYUI_MANAGER:-1}"
if [ "$DISABLE_COMFYUI_MANAGER" = "1" ]; then
  mgr_src="$COMFY_ROOT/custom_nodes/ComfyUI-Manager"
  mgr_dst="$COMFY_ROOT/custom_nodes.disabled/ComfyUI-Manager"
  if [ -d "$mgr_src" ]; then
    mkdir -p "$COMFY_ROOT/custom_nodes.disabled"
    if [ ! -d "$mgr_dst" ]; then
      mv "$mgr_src" "$mgr_dst"
    fi
  fi
fi

# Ensure RIFE ckpt exists where comfyui-frame-interpolation expects it
RIFE_SRC="$COMFY_ROOT/models/upscale_models/rife49.pth"
RIFE_DST_DIR="$COMFY_ROOT/custom_nodes/comfyui-frame-interpolation/ckpts/rife"
if [ -f "$RIFE_SRC" ]; then
  mkdir -p "$RIFE_DST_DIR"
  ln -sf "$RIFE_SRC" "$RIFE_DST_DIR/rife49.pth"
fi


echo "BOOT: symlinks ready"
ls -lah "$COMFY_ROOT/models" | head -n 200 || true

export COMFY_PYTHON="$(command -v python3 || command -v python)"
exec /worker-venv/bin/python /app/runner.py
