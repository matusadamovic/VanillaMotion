import asyncio
import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

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

# 4 buttons max (2x2 grid) + paging arrows
PAGE_SIZE = int(os.environ.get("LORA_PAGE_SIZE", "4"))  # keep 4 by default
WEIGHT_OPTIONS_RAW = os.environ.get("LORA_WEIGHT_OPTIONS", "0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")


def load_lora_catalog() -> Dict[str, Any]:
    with open(LORA_CATALOG_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not data:
        raise RuntimeError("loras.json must be a non-empty JSON object")
    return data


LORA_CATALOG = load_lora_catalog()
LORA_KEYS = sorted(LORA_CATALOG.keys())
MODEL_CHOICES = [("WAN", "wan"), ("GGUF", "gguf")]

rate_limit_cache: Dict[int, float] = {}
dedup_cache: Dict[int, float] = {}  # update_id -> timestamp


def db_conn():
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)


def _parse_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _float_eq(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


def _float_in_list(value: float, values: List[float]) -> bool:
    return any(_float_eq(value, v) for v in values)


def _parse_weight_options(raw: str) -> List[float]:
    opts: List[float] = []
    for token in raw.split(","):
        s = token.strip()
        if not s:
            continue
        try:
            v = float(s)
        except Exception:
            continue
        if not _float_in_list(v, opts):
            opts.append(v)
    return opts


WEIGHT_OPTIONS = _parse_weight_options(WEIGHT_OPTIONS_RAW) or [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def _format_weight(value: float) -> str:
    s = f"{value:.2f}".rstrip("0").rstrip(".")
    return s or "0"


def _weight_options_for(cfg: Dict[str, Any], kind: str) -> Tuple[List[float], Optional[float]]:
    default: Optional[float] = None
    if kind == "single":
        default = _parse_float(cfg.get("strength"))
    elif kind == "high":
        default = _parse_float(cfg.get("high_strength"))
    elif kind == "low":
        default = _parse_float(cfg.get("low_strength"))

    options = list(WEIGHT_OPTIONS)
    if default is not None and not _float_in_list(default, options):
        options.insert(0, default)
    return options, default


def _get_lora_cfg_by_index(idx: int) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    if idx < 0 or idx >= len(LORA_KEYS):
        return None, None
    lora_key = LORA_KEYS[idx]
    cfg = LORA_CATALOG.get(lora_key)
    if not isinstance(cfg, dict):
        return None, None
    return lora_key, cfg


# ---------------- Telegram helpers ----------------

async def tg_post(method: str, data: Dict[str, Any], timeout: float = 30.0) -> Dict[str, Any]:
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/{method}"
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, data=data)
        if resp.status_code >= 400:
            # make failures debuggable (e.g., BUTTON_DATA_INVALID)
            print("Telegram error:", resp.status_code, resp.text)
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


async def edit_message_text(chat_id: int, message_id: int, text: str, reply_markup: Optional[str] = None) -> None:
    data: Dict[str, Any] = {"chat_id": chat_id, "message_id": message_id, "text": text}
    if reply_markup is not None:
        data["reply_markup"] = reply_markup
    try:
        await tg_post("editMessageText", data, timeout=15.0)
    except Exception:
        return


async def answer_callback(callback_query_id: str) -> None:
    try:
        await tg_post("answerCallbackQuery", {"callback_query_id": callback_query_id}, timeout=15.0)
    except Exception:
        return


async def edit_keyboard(chat_id: int, message_id: int, reply_markup: str) -> None:
    # Updates only the inline keyboard; keeps text intact
    await tg_post(
        "editMessageReplyMarkup",
        {"chat_id": chat_id, "message_id": message_id, "reply_markup": reply_markup},
        timeout=15.0,
    )


def build_lora_keyboard(job_id: str, page: int = 0) -> str:
    total = len(LORA_KEYS)
    if total == 0:
        return json.dumps({"inline_keyboard": []})

    page_size = max(1, int(PAGE_SIZE))
    max_page = (total - 1) // page_size
    page = max(0, min(int(page), max_page))

    start = page * page_size
    end = min(start + page_size, total)
    chunk = LORA_KEYS[start:end]

    rows = []
    # 2x2 grid for up to 4 options
    for i in range(0, len(chunk), 2):
        row = []
        for j in range(2):
            if i + j >= len(chunk):
                break
            key = chunk[i + j]
            cfg = LORA_CATALOG[key]
            label = str(cfg.get("label") or key)

            # IMPORTANT: keep callback_data short; use index instead of key
            idx = start + (i + j)
            row.append({"text": label, "callback_data": f"l:{idx}:{job_id}"})
        rows.append(row)

    nav = []
    if page > 0:
        nav.append({"text": "‹", "callback_data": f"p:{page-1}:{job_id}"})
    nav.append({"text": f"{page+1}/{max_page+1}", "callback_data": f"noop:{job_id}"})
    if page < max_page:
        nav.append({"text": "›", "callback_data": f"p:{page+1}:{job_id}"})
    rows.append(nav)

    return json.dumps({"inline_keyboard": rows})


def build_weight_keyboard(
    job_id: str,
    lora_idx: int,
    cfg: Dict[str, Any],
    kind: str,
    *,
    high_idx: Optional[int] = None,
) -> str:
    options, default = _weight_options_for(cfg, kind)
    if not options:
        return json.dumps({"inline_keyboard": []})

    rows = []
    for i in range(0, len(options), 2):
        row = []
        for j in range(2):
            if i + j >= len(options):
                break
            value = options[i + j]
            label = _format_weight(value)
            if default is not None and _float_eq(value, default):
                label = f"{label} (default)"

            if kind == "single":
                data = f"ws:{lora_idx}:{i + j}:{job_id}"
            elif kind == "high":
                data = f"wh:{lora_idx}:{i + j}:{job_id}"
            else:
                data = f"wl:{lora_idx}:{high_idx}:{i + j}:{job_id}"
            row.append({"text": label, "callback_data": data})
        rows.append(row)

    return json.dumps({"inline_keyboard": rows})


def build_model_keyboard(
    job_id: str,
    lora_idx: int,
    *,
    is_pair: bool,
    weight_idx: Optional[int] = None,
    high_idx: Optional[int] = None,
    low_idx: Optional[int] = None,
) -> str:
    rows = []
    for i in range(0, len(MODEL_CHOICES), 2):
        row = []
        for j in range(2):
            if i + j >= len(MODEL_CHOICES):
                break
            label, key = MODEL_CHOICES[i + j]
            if is_pair:
                data = f"mp:{lora_idx}:{high_idx}:{low_idx}:{key}:{job_id}"
            else:
                data = f"ms:{lora_idx}:{weight_idx}:{key}:{job_id}"
            row.append({"text": label, "callback_data": data})
        rows.append(row)

    return json.dumps({"inline_keyboard": rows})


async def send_lora_picker(chat_id: int, job_id: str) -> None:
    await tg_post(
        "sendMessage",
        {
            "chat_id": chat_id,
            "text": "Vyber štýl (LoRA):",
            "reply_markup": build_lora_keyboard(job_id, page=0),
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
    msg = cq.get("message") or {}
    chat = msg.get("chat") or {}
    chat_id = int(chat.get("id") or 0)
    message_id = int(msg.get("message_id") or 0)

    # Supported:
    # - l:<idx>:<job_id>                 (select LoRA)
    # - p:<page>:<job_id>                (page)
    # - ws:<lora_idx>:<w_idx>:<job_id>   (single weight)
    # - wh:<lora_idx>:<h_idx>:<job_id>   (pair high)
    # - wl:<lora_idx>:<h_idx>:<l_idx>:<job_id> (pair low)
    # - ms:<lora_idx>:<w_idx>:<model>:<job_id> (single model)
    # - mp:<lora_idx>:<h_idx>:<l_idx>:<model>:<job_id> (pair model)
    # - noop:<job_id>                    (do nothing)
    parts = data.split(":")
    if len(parts) < 2:
        return

    tag = parts[0]

    if tag == "noop":
        return

    if tag == "p":
        if len(parts) != 3:
            return
        try:
            page = int(parts[1])
            job_id = parts[2]
        except Exception:
            return
        if chat_id and message_id:
            try:
                await edit_keyboard(chat_id, message_id, build_lora_keyboard(job_id, page=page))
            except Exception as exc:
                print(f"edit keyboard error: {exc}")
        return

    if tag == "l":
        if len(parts) != 3:
            return
        try:
            idx = int(parts[1])
            job_id = parts[2]
        except Exception:
            return

        lora_key, cfg = _get_lora_cfg_by_index(idx)
        if not cfg:
            return

        if not (chat_id and message_id):
            return

        label = str(cfg.get("label") or lora_key)
        lora_type = str(cfg.get("type") or "single").lower()
        if lora_type == "pair":
            reply_markup = build_weight_keyboard(job_id, idx, cfg, "high")
            await edit_message_text(chat_id, message_id, f"{label}: vyber HIGH vahu", reply_markup)
        else:
            reply_markup = build_weight_keyboard(job_id, idx, cfg, "single")
            await edit_message_text(chat_id, message_id, f"{label}: vyber vahu", reply_markup)
        return

    if tag == "ws":
        if len(parts) != 4:
            return
        try:
            lora_idx = int(parts[1])
            weight_idx = int(parts[2])
            job_id = parts[3]
        except Exception:
            return

        lora_key, cfg = _get_lora_cfg_by_index(lora_idx)
        if not cfg:
            return
        if str(cfg.get("type") or "single").lower() == "pair":
            return

        options, _ = _weight_options_for(cfg, "single")
        if weight_idx < 0 or weight_idx >= len(options):
            return

        if not (chat_id and message_id):
            return

        label = str(cfg.get("label") or lora_key)
        reply_markup = build_model_keyboard(job_id, lora_idx, is_pair=False, weight_idx=weight_idx)
        await edit_message_text(chat_id, message_id, f"{label}: vyber model (WAN/GGUF)", reply_markup)
        return

    if tag == "wh":
        if len(parts) != 4:
            return
        try:
            lora_idx = int(parts[1])
            high_idx = int(parts[2])
            job_id = parts[3]
        except Exception:
            return

        lora_key, cfg = _get_lora_cfg_by_index(lora_idx)
        if not cfg:
            return
        if str(cfg.get("type") or "single").lower() != "pair":
            return

        options, _ = _weight_options_for(cfg, "high")
        if high_idx < 0 or high_idx >= len(options):
            return

        if not (chat_id and message_id):
            return

        label = str(cfg.get("label") or lora_key)
        reply_markup = build_weight_keyboard(job_id, lora_idx, cfg, "low", high_idx=high_idx)
        await edit_message_text(chat_id, message_id, f"{label}: vyber LOW vahu", reply_markup)
        return

    if tag == "wl":
        if len(parts) != 5:
            return
        try:
            lora_idx = int(parts[1])
            high_idx = int(parts[2])
            low_idx = int(parts[3])
            job_id = parts[4]
        except Exception:
            return

        lora_key, cfg = _get_lora_cfg_by_index(lora_idx)
        if not cfg:
            return
        if str(cfg.get("type") or "single").lower() != "pair":
            return

        options_high, _ = _weight_options_for(cfg, "high")
        options_low, _ = _weight_options_for(cfg, "low")
        if high_idx < 0 or high_idx >= len(options_high):
            return
        if low_idx < 0 or low_idx >= len(options_low):
            return

        if not (chat_id and message_id):
            return

        label = str(cfg.get("label") or lora_key)
        reply_markup = build_model_keyboard(job_id, lora_idx, is_pair=True, high_idx=high_idx, low_idx=low_idx)
        await edit_message_text(chat_id, message_id, f"{label}: vyber model (WAN/GGUF)", reply_markup)
        return

    if tag == "ms":
        if len(parts) != 5:
            return
        try:
            lora_idx = int(parts[1])
            weight_idx = int(parts[2])
            model_key = parts[3]
            job_id = parts[4]
        except Exception:
            return

        if model_key not in ("wan", "gguf"):
            return

        lora_key, cfg = _get_lora_cfg_by_index(lora_idx)
        if not cfg:
            return
        if str(cfg.get("type") or "single").lower() == "pair":
            return

        options, _ = _weight_options_for(cfg, "single")
        if weight_idx < 0 or weight_idx >= len(options):
            return

        row = set_queue(job_id, lora_key)

        use_gguf = model_key == "gguf"
        payload: Dict[str, Any] = {
            "job_id": job_id,
            "chat_id": int(row["chat_id"]),
            "input_file_id": row["input_file_id"],
            "lora_key": lora_key,
            "lora_type": (cfg.get("type") or "single"),
            "positive_prompt": cfg.get("positive"),
            "use_gguf": use_gguf,
        }
        payload.update(
            {
                "lora_filename": cfg.get("filename"),
                "lora_strength": options[weight_idx],
            }
        )

        runpod_id = await submit_runpod(payload)
        if runpod_id:
            set_runpod_request_id(job_id, runpod_id)

        label = str(cfg.get("label") or lora_key)
        model_label = "GGUF" if use_gguf else "WAN"
        await edit_placeholder(
            int(row["chat_id"]),
            int(row["placeholder_message_id"]),
            f"Renderujem ({label}, {model_label})…",
        )
        if chat_id and message_id:
            await edit_keyboard(chat_id, message_id, json.dumps({"inline_keyboard": []}))
        return

    if tag == "mp":
        if len(parts) != 6:
            return
        try:
            lora_idx = int(parts[1])
            high_idx = int(parts[2])
            low_idx = int(parts[3])
            model_key = parts[4]
            job_id = parts[5]
        except Exception:
            return

        if model_key not in ("wan", "gguf"):
            return

        lora_key, cfg = _get_lora_cfg_by_index(lora_idx)
        if not cfg:
            return
        if str(cfg.get("type") or "single").lower() != "pair":
            return

        options_high, _ = _weight_options_for(cfg, "high")
        options_low, _ = _weight_options_for(cfg, "low")
        if high_idx < 0 or high_idx >= len(options_high):
            return
        if low_idx < 0 or low_idx >= len(options_low):
            return

        row = set_queue(job_id, lora_key)

        use_gguf = model_key == "gguf"
        payload = {
            "job_id": job_id,
            "chat_id": int(row["chat_id"]),
            "input_file_id": row["input_file_id"],
            "lora_key": lora_key,
            "lora_type": (cfg.get("type") or "single"),
            "positive_prompt": cfg.get("positive"),
            "use_gguf": use_gguf,
            "lora_high_filename": cfg.get("high_filename"),
            "lora_high_strength": options_high[high_idx],
            "lora_low_filename": cfg.get("low_filename"),
            "lora_low_strength": options_low[low_idx],
        }

        runpod_id = await submit_runpod(payload)
        if runpod_id:
            set_runpod_request_id(job_id, runpod_id)

        label = str(cfg.get("label") or lora_key)
        model_label = "GGUF" if use_gguf else "WAN"
        await edit_placeholder(
            int(row["chat_id"]),
            int(row["placeholder_message_id"]),
            f"Renderujem ({label}, {model_label})…",
        )
        if chat_id and message_id:
            await edit_keyboard(chat_id, message_id, json.dumps({"inline_keyboard": []}))
        return

    return


async def process_update(update: Dict[str, Any]) -> Dict[str, Any]:
    _dedup(update)

    if "callback_query" in update:
        await process_callback(update)
        return {"type": "callback"}

    chat_id, file_id = extract_photo(update)
    rate_limit(chat_id)

    placeholder_id = await send_placeholder(chat_id)
    job_id = uuid.uuid4().hex  # shorter than str(uuid.uuid4())

    try:
        save_job_awaiting_lora(job_id, chat_id, placeholder_id, file_id)
    except Exception as exc:
        await edit_placeholder(chat_id, placeholder_id, f"Chyba DB: {exc}")
        raise

    await edit_placeholder(chat_id, placeholder_id, "Vyber štýl (LoRA)…")
    try:
        await send_lora_picker(chat_id, job_id)
    except Exception as exc:
        await edit_placeholder(chat_id, placeholder_id, f"Chyba: neviem zobraziť výber štýlu ({exc})")
        raise

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
