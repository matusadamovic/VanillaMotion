import asyncio
import os
import time
import uuid
from typing import Any, Dict, Optional

import httpx
import psycopg2
from fastapi import FastAPI, HTTPException, Request
from psycopg2.extras import RealDictCursor

TELEGRAM_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
RUNPOD_API_KEY = os.environ["RUNPOD_API_KEY"]
RUNPOD_ENDPOINT_ID = os.environ["RUNPOD_ENDPOINT_ID"]
RUNPOD_API_BASE = os.environ.get("RUNPOD_API_BASE", "https://api.runpod.ai")
DATABASE_URL = os.environ["DATABASE_URL"]
MAX_IMAGE_BYTES = int(os.environ.get("MAX_IMAGE_BYTES", str(10 * 1024 * 1024)))
RATE_LIMIT_SECONDS = int(os.environ.get("RATE_LIMIT_SECONDS", "30"))

# If you want to guard against webhook re-deliveries in-memory (best effort)
DEDUP_TTL_SECONDS = int(os.environ.get("DEDUP_TTL_SECONDS", "600"))

app = FastAPI()
rate_limit_cache: Dict[int, float] = {}
dedup_cache: Dict[int, float] = {}  # update_id -> timestamp


def db_conn():
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)


async def send_placeholder(chat_id: int) -> int:
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, data={"chat_id": chat_id, "text": "Spracúvam video..."})
        resp.raise_for_status()
        return resp.json()["result"]["message_id"]


async def edit_placeholder(chat_id: int, message_id: int, text: str) -> None:
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/editMessageText"
    async with httpx.AsyncClient(timeout=30) as client:
        # best-effort
        try:
            await client.post(url, data={"chat_id": chat_id, "message_id": message_id, "text": text})
        except Exception:
            pass


async def submit_runpod(payload: Dict[str, Any]) -> str:
    url = f"{RUNPOD_API_BASE}/v2/{RUNPOD_ENDPOINT_ID}/run"
    headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}"}
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, headers=headers, json={"input": payload})
        resp.raise_for_status()
        data = resp.json()
        return data.get("id") or data.get("jobId") or ""


def save_job_queued(job_id: str, chat_id: int, placeholder_id: int, file_id: str) -> None:
    with db_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO jobs (id, chat_id, placeholder_message_id, input_file_id, state, runpod_request_id)
            VALUES (%s, %s, %s, %s, 'QUEUED', NULL)
            """,
            (job_id, chat_id, placeholder_id, file_id),
        )


def set_runpod_request_id(job_id: str, runpod_id: str) -> None:
    with db_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE jobs
            SET runpod_request_id = %s, updated_at = NOW()
            WHERE id = %s
            """,
            (runpod_id, job_id),
        )


def fail_job(job_id: str, error: str) -> None:
    with db_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE jobs
            SET state = 'FAILED', error = %s, updated_at = NOW()
            WHERE id = %s
            """,
            (error, job_id),
        )


def rate_limit(chat_id: int):
    now = time.time()
    last = rate_limit_cache.get(chat_id, 0)
    if now - last < RATE_LIMIT_SECONDS:
        raise HTTPException(status_code=429, detail="Too many requests")
    rate_limit_cache[chat_id] = now


def extract_file(update: Dict[str, Any]) -> Dict[str, Any]:
    message = update.get("message") or update.get("edited_message")
    if not message:
        raise HTTPException(status_code=400, detail="No message")
    chat_id = message["chat"]["id"]
    photos = message.get("photo") or []
    if not photos:
        raise HTTPException(status_code=400, detail="No photo")
    largest = photos[-1]
    file_id = largest["file_id"]
    file_size = largest.get("file_size", 0)
    if file_size and file_size > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail="Image too large")
    return {"chat_id": chat_id, "file_id": file_id}


def _dedup(update: Dict[str, Any]) -> None:
    update_id = update.get("update_id")
    if update_id is None:
        return
    now = time.time()

    # cleanup
    if dedup_cache:
        cutoff = now - DEDUP_TTL_SECONDS
        for k, ts in list(dedup_cache.items()):
            if ts < cutoff:
                dedup_cache.pop(k, None)

    if update_id in dedup_cache:
        raise HTTPException(status_code=200, detail="Duplicate update ignored")
    dedup_cache[update_id] = now


async def process_update(update: Dict[str, Any]) -> Dict[str, Any]:
    _dedup(update)

    file_info = extract_file(update)
    chat_id = int(file_info["chat_id"])
    rate_limit(chat_id)

    placeholder_id = await send_placeholder(chat_id)
    job_id = str(uuid.uuid4())

    # IMPORTANT: insert job first to avoid worker race (worker calls mark_running)
    try:
        save_job_queued(job_id, chat_id, placeholder_id, file_info["file_id"])
    except Exception as exc:
        await edit_placeholder(chat_id, placeholder_id, f"Chyba: nepodarilo sa uložiť job do DB.\n{exc}")
        raise

    payload = {
        "job_id": job_id,
        "chat_id": chat_id,
        "placeholder_message_id": placeholder_id,  # worker môže ignorovať, nechávame kvôli kompatibilite
        "input_file_id": file_info["file_id"],
        "seed": None,
    }

    try:
        runpod_id = await submit_runpod(payload)
        if runpod_id:
            set_runpod_request_id(job_id, runpod_id)
        return {"job_id": job_id, "runpod_request_id": runpod_id}
    except Exception as exc:
        err = f"RunPod submit failed: {exc}"
        fail_job(job_id, err)
        await edit_placeholder(chat_id, placeholder_id, "Chyba: nepodarilo sa spustiť render. Skús znova o chvíľu.")
        raise


def clear_webhook():
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/deleteWebhook"
    with httpx.Client(timeout=10) as client:
        client.get(url)


async def poll_loop():
    offset = None
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
    async with httpx.AsyncClient(timeout=35) as client:
        while True:
            params = {"timeout": 30}
            if offset is not None:
                params["offset"] = offset
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            for update in data.get("result", []):
                offset = update["update_id"] + 1
                try:
                    await process_update(update)
                except HTTPException as exc:
                    # ignore dedup / rate limit etc.
                    if exc.status_code not in (200, 429):
                        print(f"polling http error: {exc.detail}")
                except Exception as exc:
                    print(f"polling error: {exc}")
            await asyncio.sleep(1)


def run_long_polling():
    clear_webhook()
    asyncio.run(poll_loop())


@app.post("/webhook")
async def webhook(request: Request):
    update = await request.json()

    # Fast-ack: schedule processing and return immediately
    async def _bg():
        try:
            await process_update(update)
        except Exception as exc:
            # log only; Telegram already got 200
            print(f"webhook background error: {exc}")

    asyncio.create_task(_bg())
    return {"ok": True}


@app.get("/healthz")
async def health():
    with db_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT 1")
    return {"status": "ok"}


if __name__ == "__main__":
    mode = os.environ.get("BOT_MODE", "polling").lower()
    if mode in ("polling", "long-polling", "long_polling", "poll"):
        run_long_polling()
    else:
        raise SystemExit("BOT_MODE must be 'polling' for long polling runs")