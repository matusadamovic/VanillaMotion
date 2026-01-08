#!/usr/bin/env bash
set -euo pipefail

COMFY_ROOT="/app/ComfyUI"
MODEL_BUNDLE_DEFAULT="/opt/runpod/cache/models"
MODEL_BUNDLE="${MODEL_BUNDLE_PATH:-$MODEL_BUNDLE_DEFAULT}"
TMP_ROOT="/tmp"

# Best-effort cleanup starych jobov
find "$TMP_ROOT" -maxdepth 1 -type d -name "job_*" -mtime +0 -exec rm -rf {} + || true

if [ ! -d "$MODEL_BUNDLE" ]; then
  echo "Model bundle nenajdeny na $MODEL_BUNDLE"
  ls -lah /opt || true
  exit 1
fi

mkdir -p "$COMFY_ROOT/input/image" "$COMFY_ROOT/output" "$COMFY_ROOT/models"

# Mapovanie modelov do ComfyUI adresarov
ln -sfn "$MODEL_BUNDLE/checkpoints" "$COMFY_ROOT/models/checkpoints"
ln -sfn "$MODEL_BUNDLE/loras" "$COMFY_ROOT/models/loras"
ln -sfn "$MODEL_BUNDLE/vae" "$COMFY_ROOT/models/vae"
ln -sfn "$MODEL_BUNDLE/clip" "$COMFY_ROOT/models/clip"
ln -sfn "$MODEL_BUNDLE/controlnet" "$COMFY_ROOT/models/controlnet"
ln -sfn "$MODEL_BUNDLE/upscale_models" "$COMFY_ROOT/models/upscale_models"
ln -sfn "$MODEL_BUNDLE/embeddings" "$COMFY_ROOT/models/embeddings"
ln -sfn "$MODEL_BUNDLE/checkpoints_gguf" "$COMFY_ROOT/models/checkpoints_gguf"

exec /venv/bin/python /app/runner.py
