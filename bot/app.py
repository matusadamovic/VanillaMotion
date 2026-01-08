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
DATABASE_URL = os.environ["DATABASE_URL"]
MAX_IMAGE_BYTES = int(os.environ.get("MAX_IMAGE_BYTES", str(10 * 1024 * 1024)))
RATE_LIMIT_SECONDS = int(os.environ.get("RATE_LIMIT_SECONDS", "30"))

app = FastAPI()
rate_limit_cache: Dict[int, float] = {}


def db_conn():
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)


async def send_placeholder(chat_id: int) -> int:
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, data={"chat_id": chat_id, "text": "Spracuvam video..."})
        resp.raise_for_status()
        return resp.json()["result"]["message_id"]


async def submit_runpod(payload: Dict[str, Any]) -> str:
    url = f"https://api.runpod.io/v2/{RUNPOD_ENDPOINT_ID}/run"
    headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}"}
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, headers=headers, json={"input": payload})
        resp.raise_for_status()
        data = resp.json()
        return data.get("id") or data.get("jobId") or ""


def save_job(job_id: str, chat_id: int, placeholder_id: int, file_id: str, runpod_id: Optional[str]):
    with db_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO jobs (id, chat_id, placeholder_message_id, input_file_id, state, runpod_request_id)
            VALUES (%s, %s, %s, %s, 'QUEUED', %s)
            """,
            (job_id, chat_id, placeholder_id, file_id, runpod_id),
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


async def process_update(update: Dict[str, Any]) -> Dict[str, Any]:
    file_info = extract_file(update)
    chat_id = int(file_info["chat_id"])
    rate_limit(chat_id)

    placeholder_id = await send_placeholder(chat_id)
    job_id = str(uuid.uuid4())

    payload = {
        "job_id": job_id,
        "chat_id": chat_id,
        "placeholder_message_id": placeholder_id,
        "input_file_id": file_info["file_id"],
        "seed": None,
    }
    runpod_id = await submit_runpod(payload)
    save_job(job_id, chat_id, placeholder_id, file_info["file_id"], runpod_id)
    return {"job_id": job_id, "runpod_request_id": runpod_id}


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
                except Exception as exc:
                    print(f"polling error: {exc}")
            await asyncio.sleep(1)


def run_long_polling():
    clear_webhook()
    asyncio.run(poll_loop())


@app.post("/webhook")
async def webhook(request: Request):
    update = await request.json()
    result = await process_update(update)
    return {"ok": True, **result}


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
