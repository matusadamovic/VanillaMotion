#!/usr/bin/env bash
set -euo pipefail

COMFY_ROOT="/app/ComfyUI"
TMP_ROOT="/tmp"


# RunPod cached HF models path
HF_CACHE_ROOT="/runpod-volume/huggingface-cache/hub"
HF_MODEL_NAME="${HF_MODEL_NAME:-matadamovic/VanillaMotionBOTmodels}"

# Allow override for local testing
MODEL_BUNDLE="${MODEL_BUNDLE_PATH:-}"

# Best-effort cleanup starych jobov
find "$TMP_ROOT" -maxdepth 1 -type d -name "job_*" -mtime +0 -exec rm -rf {} + || true

if [ -z "$MODEL_BUNDLE" ]; then
  cache_name="models--${HF_MODEL_NAME//\//--}"
  snapshots_dir="$HF_CACHE_ROOT/$cache_name/snapshots"

  if [ ! -d "$snapshots_dir" ]; then
    echo "HF cached model snapshot nenajdeny: $snapshots_dir"
    echo "Skontroluj: (1) Model field repo ID, (2) HF token pre private repo, (3) ze caching prebehol"
    ls -lah /runpod-volume || true
    ls -lah "$HF_CACHE_ROOT" || true
    exit 1
  fi

  MODEL_BUNDLE="$(ls -1dt "$snapshots_dir"/* | head -n 1)"
fi

echo "Pouzivam MODEL_BUNDLE=$MODEL_BUNDLE"
ls -lah "$MODEL_BUNDLE" | head -n 200 || true

mkdir -p "$COMFY_ROOT/input/image" "$COMFY_ROOT/output" "$COMFY_ROOT/models"

# Mapovanie modelov do ComfyUI adresarov
ln -sfn "$MODEL_BUNDLE/checkpoints"       "$COMFY_ROOT/models/checkpoints"
ln -sfn "$MODEL_BUNDLE/loras"             "$COMFY_ROOT/models/loras"
ln -sfn "$MODEL_BUNDLE/vae"               "$COMFY_ROOT/models/vae"
ln -sfn "$MODEL_BUNDLE/clip"              "$COMFY_ROOT/models/clip"
ln -sfn "$MODEL_BUNDLE/controlnet"        "$COMFY_ROOT/models/controlnet"
ln -sfn "$MODEL_BUNDLE/upscale_models"    "$COMFY_ROOT/models/upscale_models"
ln -sfn "$MODEL_BUNDLE/embeddings"        "$COMFY_ROOT/models/embeddings"
ln -sfn "$MODEL_BUNDLE/checkpoints_gguf"  "$COMFY_ROOT/models/checkpoints_gguf"

exec /venv/bin/python /app/runner.py