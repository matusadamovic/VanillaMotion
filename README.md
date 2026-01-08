# ImgVidBot - plan a artefakty

## Co to robi
- Telegram bot (FastAPI) prijme fotku, hned posle placeholder, ulozi metadate do Postgres, spusti RunPod async job.
- GPU worker (ComfyUI headless) vezme workflow `i2v_WAN22_6step_3Samplers_GGUF_FirstLast.json`, vstupny obrazok z Telegramu, vygeneruje ~5s video, edituje povodnu spravu, vycisti /tmp.
- Ziadne binarne data sa neukladaju mimo Telegramu; DB uchovava len metadate (stav, chat_id, message_id, file_id, attempts, error, runpod_request_id).
- Modely su v jednom privatnom HF repo; RunPod cached model mount sa symlinkuje do ComfyUI adresarov cez `worker/bootstrap.sh`.

## Klucove subory
- `Dockerfile.worker` - GPU image s ComfyUI, pinned requirements, custom nodes.
- `worker/bootstrap.sh` - symlinkuje HF cache do ComfyUI, cisti /tmp, spusta runner.
- `worker/runner.py` - RunPod handler, stavy v DB, stahuje obrazok, spusta ComfyUI API, uploaduje video, fallbacky a cistenie.
- `bot/app.py` - FastAPI webhook, rate-limit, placeholder, zapis do DB, submit RunPod job, health.
- `migrations/001_init.sql` - schéma tabulky jobs a indexy.
- `infra/runpod-endpoint.json` - sablona RunPod serverless endpointu (scale-to-zero).
- `docs/HF_BUNDLE.md` - struktura HF balika, git-lfs prikazy.
- `docs/RUNBOOK.md` - prevadzkovy postup (deploy, rotacia secretov, incidenty).

## Nasadenie (kratky tahak)
1) HF modely: postup v `docs/HF_BUNDLE.md` (git-lfs, kopia modelov, push).  
2) DB: `psql $DATABASE_URL -f migrations/001_init.sql`.  
3) Worker image: `docker build -t imgvidbot-worker -f Dockerfile.worker .` a push do registry.  
4) RunPod: vytvor endpoint podla `infra/runpod-endpoint.json`, nastav secrets/env, `minWorkers=0`, `maxWorkers=1`.  
5) Bot: spust `uvicorn bot.app:app --host 0.0.0.0 --port 8000`, nastav Telegram webhook na `/webhook`.  
6) Test: posli fotku v sandbox chate, ocakavaj placeholder -> finalne video bez duplikatov.  

## Poznamky k datam a bezpecnosti
- Binarne subory len v `/tmp/job_*` pocas behu; runner a bootstrap robia best-effort cleanup.
- Ziadne ulozisko (S3/R2/B2, DB, disk, network volumes) pre obrazky/videa; Telegram je jediny storage/transport.
- Secrets len v env (TELEGRAM_BOT_TOKEN, RUNPOD_API_KEY, HF_TOKEN, DATABASE_URL); nelogovat.

## Determinizmus
- Fixovat seed v payload (ak treba) a nemenit modely/gguf/vae bez verziovania.
- Pouzivat rovnaky typ GPU a CUDA verziu ako v image (CUDA 12.1 runtime).
# VanillaMotion
