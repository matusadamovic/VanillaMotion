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

# Resolve cached snapshot
if [ -z "$MODEL_BUNDLE" ]; then
  cache_name="models--${HF_MODEL_NAME//\//--}"
  snapshots_dir="$HF_CACHE_ROOT/$cache_name/snapshots"
  snapshots_glob="$snapshots_dir/*"

  echo "BOOT: looking for snapshots: $snapshots_glob"

  if [ ! -d "$snapshots_dir" ]; then
    echo "ERROR: snapshots dir nenajdeny: $snapshots_dir"
    echo "Debug:"
    ls -lah /runpod-volume || true
    ls -lah "$HF_CACHE_ROOT" || true
    exit 1
  fi

  MODEL_BUNDLE="$(ls -1dt $snapshots_glob 2>/dev/null | head -n 1 || true)"
fi

if [ -z "$MODEL_BUNDLE" ] || [ ! -d "$MODEL_BUNDLE" ]; then
  echo "ERROR: MODEL_BUNDLE sa nepodarilo urcit alebo neexistuje."
  echo "MODEL_BUNDLE=$MODEL_BUNDLE"
  ls -lah "$HF_CACHE_ROOT" || true
  exit 1
fi

echo "BOOT: Using MODEL_BUNDLE=$MODEL_BUNDLE"
ls -lah "$MODEL_BUNDLE" | head -n 200 || true

# Ensure Comfy paths exist
mkdir -p "$COMFY_ROOT/input/image" "$COMFY_ROOT/output" "$COMFY_ROOT/models"

# Fail fast ak chýba základná štruktúra repa
for d in checkpoints loras vae clip checkpoints_gguf; do
  if [ ! -d "$MODEL_BUNDLE/$d" ]; then
    echo "ERROR: Chyba adresar v HF modeli: $MODEL_BUNDLE/$d"
    echo "Obsah MODEL_BUNDLE (maxdepth 2):"
    find "$MODEL_BUNDLE" -maxdepth 2 -type d | head -n 200 || true
    exit 1
  fi
done

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


echo "BOOT: symlinks ready"
ls -lah "$COMFY_ROOT/models" | head -n 200 || true

export COMFY_PYTHON="$(command -v python3 || command -v python)"
exec /worker-venv/bin/python /app/runner.py
