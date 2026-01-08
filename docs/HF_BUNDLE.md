## Struktura HF balika (privatny repo)
```
models/
  checkpoints/
    DasiwaWAN22I2V14BV8V1_midnightflirtHighV7.safetensors
    DasiwaWAN22I2V14BV8V1_midnightflirtLowV7.safetensors
    wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors
    wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors
  checkpoints_gguf/
    DasiwaWAN22I2V14BTastysinV8_q8High.gguf
    DasiwaWAN22I2V14BTastysinV8_q8Low.gguf
  loras/
    wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors
    wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors
  clip/
    umt5_xxl_fp8_e4m3fn_scaled.safetensors
  vae/
    wan_2.1_vae.safetensors
  upscale_models/
    remacri_original.pth
    rife49.pth
  controlnet/
  embeddings/
  optional/
    NSFW-22-H-e8.safetensors
    NSFW-22-L-e8.safetensors
```

## Git LFS a upload
```bash
git lfs install
git clone <private_repo_url> models_repo
cd models_repo
git lfs track "*.safetensors" "*.pth" "*.gguf"
# skopiruj modely do cest vyssie
git add .
git commit -m "Add ComfyUI bundle"
git push
git lfs ls-files
huggingface-cli whoami
```

## Nasadenie na RunPod cached models
- V RunPod endpointe nastav `MODEL_BUNDLE_PATH` na mount (typicky `/opt/runpod/cache/models`).
- V worker image nebalime modely; spoliehame sa na HF cache.
- Ak sa zmeni obsah HF repa, prepni na novy tag/commit a nechaj RunPod re-cache.
