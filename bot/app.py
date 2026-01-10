import asyncio
import json
import os
import time
import uuid
from typing import Any, Dict, Tuple

import httpx
import psycopg2
from psycopg2.extras import RealDictCursor

TELEGRAM_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
RUNPOD_API_KEY = os.environ["RUNPOD_API_KEY"]
RUNPOD_ENDPOINT_ID = os.environ["RUNPOD_ENDPOINT_ID"]
RUNPOD_API_BASE = os.environ.get("RUNPOD_API_BASE", "https://api.runpod.ai")
DATABASE_URL = os.environ["DATABASE_URL"]

MAX_IMAGE_BYTES = int(os.environ.get("MAX_IMAGE_BYTES", str(10 * 1024 * 1024)))
RATE_LIMIT_SECONDS = int(os.environ.get("RATE_LIMIT_SECONDS", "30"))
DEDUP_TTL_SECONDS = int(os.environ.get("DEDUP_TTL_SECONDS", "600"))

LORA_CATALOG_PATH = os.environ.get("LORA_CATALOG_PATH", "/app/loras.json")


def load_lora_catalog() -> Dict[str, Any]:
    with open(LORA_CATALOG_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not data:
        raise RuntimeError("loras.json must be a non-empty JSON object")
    return data


LORA_CATALOG = load_lora_catalog()

rate_limit_cache: Dict[int, float] = {}
dedup_cache: Dict[int, float] = {}  # update_id -> timestamp


def db_conn():
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)


# ---------------- Telegram helpers ----------------

async def tg_post(method: str, data: Dict[str, Any], timeout: float = 30.0) -> Dict[str, Any]:
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/{method}"
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, data=data)
        resp.raise_for_status()
        return resp.json()


async def send_placeholder(chat_id: int) -> int:
    r = await tg_post("sendMessage", {"chat_id": chat_id, "text": "Spracúvam…"})
    return int(r["result"]["message_id"])


async def edit_placeholder(chat_id: int, message_id: int, text: str) -> None:
    try:
        await tg_post("editMessageText", {"chat_id": chat_id, "message_id": message_id, "text": text})
    except Exception:
        return


async def answer_callback(callback_query_id: str) -> None:
    try:
        await tg_post("answerCallbackQuery", {"callback_query_id": callback_query_id}, timeout=15.0)
    except Exception:
        return


def build_lora_keyboard(job_id: str) -> str:
    rows = []
    for key, cfg in LORA_CATALOG.items():
        label = str(cfg.get("label") or key)
        rows.append([{"text": label, "callback_data": f"lora:{key}:{job_id}"}])
    return json.dumps({"inline_keyboard": rows})


async def send_lora_picker(chat_id: int, job_id: str) -> None:
    await tg_post(
        "sendMessage",
        {
            "chat_id": chat_id,
            "text": "Vyber štýl (LoRA):",
            "reply_markup": build_lora_keyboard(job_id),
        },
    )


def clear_webhook():
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/deleteWebhook"
    with httpx.Client(timeout=10) as client:
        try:
            client.get(url)
        except Exception:
            pass


# ---------------- RunPod helpers ----------------

async def submit_runpod(payload: Dict[str, Any]) -> str:
    url = f"{RUNPOD_API_BASE}/v2/{RUNPOD_ENDPOINT_ID}/run"
    headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}"}
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, headers=headers, json={"input": payload})
        resp.raise_for_status()
        data = resp.json()
        return data.get("id") or data.get("jobId") or ""


# ---------------- DB helpers ----------------

def save_job_awaiting_lora(job_id: str, chat_id: int, placeholder_id: int, file_id: str) -> None:
    with db_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO jobs (id, chat_id, placeholder_message_id, input_file_id, state, runpod_request_id, attempts)
            VALUES (%s, %s, %s, %s, 'AWAITING_LORA', NULL, 0)
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


def set_queue(job_id: str, lora_key: str) -> Dict[str, Any]:
    # Store only lora_key (optional) + move to QUEUED. No prompt in DB.
    with db_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE jobs
            SET state = 'QUEUED',
                lora_key = %s,
                updated_at = NOW()
            WHERE id = %s AND state = 'AWAITING_LORA'
            RETURNING chat_id, placeholder_message_id, input_file_id;
            """,
            (lora_key, job_id),
        )
        row = cur.fetchone()
        if not row:
            raise RuntimeError("Job not in AWAITING_LORA (already queued/running/done or missing)")
        return row


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


# ---------------- Update processing ----------------

def rate_limit(chat_id: int):
    now = time.time()
    last = rate_limit_cache.get(chat_id, 0.0)
    if now - last < RATE_LIMIT_SECONDS:
        raise RuntimeError("rate_limited")
    rate_limit_cache[chat_id] = now


def _dedup(update: Dict[str, Any]) -> None:
    update_id = update.get("update_id")
    if update_id is None:
        return

    now = time.time()
    cutoff = now - DEDUP_TTL_SECONDS
    for k, ts in list(dedup_cache.items()):
        if ts < cutoff:
            dedup_cache.pop(k, None)

    if update_id in dedup_cache:
        raise RuntimeError("duplicate_update")
    dedup_cache[update_id] = now


def extract_photo(update: Dict[str, Any]) -> Tuple[int, str]:
    msg = update.get("message") or update.get("edited_message")
    if not msg:
        raise RuntimeError("no_message")

    chat_id = int(msg["chat"]["id"])
    photos = msg.get("photo") or []
    if not photos:
        raise RuntimeError("no_photo")

    largest = photos[-1]
    file_id = largest["file_id"]
    file_size = int(largest.get("file_size") or 0)
    if file_size and file_size > MAX_IMAGE_BYTES:
        raise RuntimeError("image_too_large")

    return chat_id, file_id


async def process_callback(update: Dict[str, Any]) -> None:
    cq = update["callback_query"]
    await answer_callback(cq["id"])

    data = cq.get("data") or ""
    # expected: lora:<key>:<job_id>
    try:
        _, lora_key, job_id = data.split(":", 2)
    except ValueError:
        return

    cfg = LORA_CATALOG.get(lora_key)
    if not isinstance(cfg, dict):
        return

    row = set_queue(job_id, lora_key)

    payload = {
        "job_id": job_id,
        "chat_id": int(row["chat_id"]),
        "input_file_id": row["input_file_id"],
        "lora_key": lora_key,
        # explicit: worker does not need DB prompts nor local catalog
        "lora_filename": cfg.get("filename"),
        "lora_strength": cfg.get("strength"),
        "positive_prompt": cfg.get("positive"),
    }

    runpod_id = await submit_runpod(payload)
    if runpod_id:
        set_runpod_request_id(job_id, runpod_id)

    label = str(cfg.get("label") or lora_key)
    await edit_placeholder(int(row["chat_id"]), int(row["placeholder_message_id"]), f"Renderujem ({label})…")


async def process_update(update: Dict[str, Any]) -> Dict[str, Any]:
    _dedup(update)

    if "callback_query" in update:
        await process_callback(update)
        return {"type": "callback"}

    chat_id, file_id = extract_photo(update)
    rate_limit(chat_id)

    placeholder_id = await send_placeholder(chat_id)
    job_id = str(uuid.uuid4())

    try:
        save_job_awaiting_lora(job_id, chat_id, placeholder_id, file_id)
    except Exception as exc:
        await edit_placeholder(chat_id, placeholder_id, f"Chyba DB: {exc}")
        raise

    await edit_placeholder(chat_id, placeholder_id, "Vyber štýl (LoRA)…")
    await send_lora_picker(chat_id, job_id)

    return {"type": "photo", "job_id": job_id}


async def poll_loop():
    clear_webhook()

    offset = None
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
    async with httpx.AsyncClient(timeout=35) as client:
        while True:
            params: Dict[str, Any] = {"timeout": 30}
            if offset is not None:
                params["offset"] = offset

            try:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()
            except Exception:
                await asyncio.sleep(1)
                continue

            for update in data.get("result", []):
                offset = int(update["update_id"]) + 1
                try:
                    await process_update(update)
                except RuntimeError as exc:
                    if str(exc) not in ("duplicate_update", "rate_limited", "no_photo", "no_message"):
                        print(f"bot runtime error: {exc}")
                except Exception as exc:
                    print(f"bot error: {exc}")

            await asyncio.sleep(0.2)


def main():
    asyncio.run(poll_loop())


if __name__ == "__main__":
    main()