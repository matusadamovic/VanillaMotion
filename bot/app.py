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
MODEL_CATALOG_PATH = os.environ.get("MODEL_CATALOG_PATH", "/app/models.json")

# 4 buttons max (2x2 grid) + paging arrows
PAGE_SIZE = int(os.environ.get("LORA_PAGE_SIZE", "4"))  # keep 4 by default
MODEL_PAGE_SIZE = int(os.environ.get("MODEL_PAGE_SIZE", "4"))
WEIGHT_OPTIONS_RAW = os.environ.get("LORA_WEIGHT_OPTIONS", "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")


def load_lora_catalog() -> Dict[str, Any]:
    with open(LORA_CATALOG_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not data:
        raise RuntimeError("loras.json must be a non-empty JSON object")
    return data


def load_model_catalog() -> Dict[str, Any]:
    if not os.path.exists(MODEL_CATALOG_PATH):
        return {}
    with open(MODEL_CATALOG_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise RuntimeError("models.json must be a JSON object")
    return data


def _normalize_model_type(value: Any) -> Optional[str]:
    s = str(value or "").strip().lower()
    if s in ("wan", "safetensor", "safetensors"):
        return "wan"
    if s == "gguf":
        return "gguf"
    return None


LORA_CATALOG = load_lora_catalog()
LORA_KEYS = sorted(LORA_CATALOG.keys())
MODEL_CATALOG = load_model_catalog()
MODEL_KEYS_BY_TYPE: Dict[str, List[str]] = {"wan": [], "gguf": []}
for key, cfg in MODEL_CATALOG.items():
    if not isinstance(cfg, dict):
        continue
    model_type = _normalize_model_type(cfg.get("type"))
    if not model_type:
        continue
    if not cfg.get("high_filename") or not cfg.get("low_filename"):
        continue
    MODEL_KEYS_BY_TYPE[model_type].append(key)
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


WEIGHT_OPTIONS = _parse_weight_options(WEIGHT_OPTIONS_RAW) or [
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
]

RESOLUTION_OPTIONS = [
    {"label": "1280x720", "width": 1280, "height": 720},
    {"label": "720x480", "width": 720, "height": 480},
    {"label": "640x480", "width": 640, "height": 480},
]
DEFAULT_RESOLUTION_IDX = 0

STEPS_OPTIONS = [4, 6, 8, 10, 12, 14]
DEFAULT_STEPS = 12

DEFAULT_PROMPT = (
    "face remains consistent across frames, subtle camera drift only, "
    "minimal head movement, micro-expression only, no dramatic rotation, no sudden tilt"
)


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


def _resolution_by_index(idx: int) -> Optional[Dict[str, Any]]:
    if idx < 0 or idx >= len(RESOLUTION_OPTIONS):
        return None
    return RESOLUTION_OPTIONS[idx]


def _steps_by_index(idx: int) -> Optional[int]:
    if idx < 0 or idx >= len(STEPS_OPTIONS):
        return None
    return STEPS_OPTIONS[idx]


def _get_lora_cfg_by_index(idx: int) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    if idx < 0 or idx >= len(LORA_KEYS):
        return None, None
    lora_key = LORA_KEYS[idx]
    cfg = LORA_CATALOG.get(lora_key)
    if not isinstance(cfg, dict):
        return None, None
    return lora_key, cfg


def _get_model_cfg_by_index(model_type: str, idx: int) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    keys = MODEL_KEYS_BY_TYPE.get(model_type) or []
    if idx < 0 or idx >= len(keys):
        return None, None
    model_key = keys[idx]
    cfg = MODEL_CATALOG.get(model_key)
    if not isinstance(cfg, dict):
        return None, None
    if _normalize_model_type(cfg.get("type")) != model_type:
        return None, None
    if not cfg.get("high_filename") or not cfg.get("low_filename"):
        return None, None
    return model_key, cfg


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


def build_unet_keyboard(
    job_id: str,
    model_type: str,
    *,
    is_pair: bool,
    lora_idx: int,
    weight_idx: Optional[int] = None,
    high_idx: Optional[int] = None,
    low_idx: Optional[int] = None,
    page: int = 0,
) -> str:
    keys = MODEL_KEYS_BY_TYPE.get(model_type) or []
    if not keys:
        return json.dumps({"inline_keyboard": []})

    page_size = max(1, int(MODEL_PAGE_SIZE))
    max_page = (len(keys) - 1) // page_size
    page = max(0, min(int(page), max_page))

    start = page * page_size
    end = min(start + page_size, len(keys))
    chunk = keys[start:end]

    rows = []
    for i in range(0, len(chunk), 2):
        row = []
        for j in range(2):
            if i + j >= len(chunk):
                break
            key = chunk[i + j]
            cfg = MODEL_CATALOG.get(key) or {}
            label = str(cfg.get("label") or key)

            idx = start + (i + j)
            if is_pair:
                data = f"mm:{lora_idx}:{high_idx}:{low_idx}:{model_type}:{idx}:{job_id}"
            else:
                data = f"mm:{lora_idx}:{weight_idx}:{model_type}:{idx}:{job_id}"
            row.append({"text": label, "callback_data": data})
        rows.append(row)

    nav = []
    if page > 0:
        if is_pair:
            nav.append({"text": "‹", "callback_data": f"pm:{model_type}:{page-1}:{lora_idx}:{high_idx}:{low_idx}:{job_id}"})
        else:
            nav.append({"text": "‹", "callback_data": f"pm:{model_type}:{page-1}:{lora_idx}:{weight_idx}:{job_id}"})
    nav.append({"text": f"{page+1}/{max_page+1}", "callback_data": f"noop:{job_id}"})
    if page < max_page:
        if is_pair:
            nav.append({"text": "›", "callback_data": f"pm:{model_type}:{page+1}:{lora_idx}:{high_idx}:{low_idx}:{job_id}"})
        else:
            nav.append({"text": "›", "callback_data": f"pm:{model_type}:{page+1}:{lora_idx}:{weight_idx}:{job_id}"})
    rows.append(nav)

    return json.dumps({"inline_keyboard": rows})


def build_last_frame_keyboard(
    job_id: str,
    *,
    is_pair: bool,
    lora_idx: int,
    model_type: str,
    model_idx: int,
    weight_idx: Optional[int] = None,
    high_idx: Optional[int] = None,
    low_idx: Optional[int] = None,
) -> str:
    if is_pair:
        on_data = f"lf:1:{lora_idx}:{high_idx}:{low_idx}:{model_type}:{model_idx}:{job_id}"
        off_data = f"lf:0:{lora_idx}:{high_idx}:{low_idx}:{model_type}:{model_idx}:{job_id}"
    else:
        on_data = f"lf:1:{lora_idx}:{weight_idx}:{model_type}:{model_idx}:{job_id}"
        off_data = f"lf:0:{lora_idx}:{weight_idx}:{model_type}:{model_idx}:{job_id}"

    rows = [
        [
            {"text": "Last frame: ON", "callback_data": on_data},
            {"text": "Last frame: OFF", "callback_data": off_data},
        ]
    ]
    return json.dumps({"inline_keyboard": rows})


def build_resolution_keyboard(
    job_id: str,
    *,
    is_pair: bool,
    lora_idx: int,
    model_type: str,
    model_idx: int,
    use_last_frame: bool,
    weight_idx: Optional[int] = None,
    high_idx: Optional[int] = None,
    low_idx: Optional[int] = None,
) -> str:
    rows = []
    for i in range(0, len(RESOLUTION_OPTIONS), 2):
        row = []
        for j in range(2):
            if i + j >= len(RESOLUTION_OPTIONS):
                break
            idx = i + j
            opt = RESOLUTION_OPTIONS[idx]
            label = str(opt.get("label") or "")
            if idx == DEFAULT_RESOLUTION_IDX:
                label = f"{label} (default)"

            if is_pair:
                data = f"rs:{idx}:{int(use_last_frame)}:{lora_idx}:{high_idx}:{low_idx}:{model_type}:{model_idx}:{job_id}"
            else:
                data = f"rs:{idx}:{int(use_last_frame)}:{lora_idx}:{weight_idx}:{model_type}:{model_idx}:{job_id}"
            row.append({"text": label, "callback_data": data})
        rows.append(row)

    return json.dumps({"inline_keyboard": rows})


def build_steps_keyboard(
    job_id: str,
    *,
    is_pair: bool,
    lora_idx: int,
    model_type: str,
    model_idx: int,
    use_last_frame: bool,
    resolution_idx: int,
    weight_idx: Optional[int] = None,
    high_idx: Optional[int] = None,
    low_idx: Optional[int] = None,
) -> str:
    rows = []
    for i in range(0, len(STEPS_OPTIONS), 2):
        row = []
        for j in range(2):
            if i + j >= len(STEPS_OPTIONS):
                break
            idx = i + j
            steps = STEPS_OPTIONS[idx]
            label = f"{steps}"
            if steps == DEFAULT_STEPS:
                label = f"{label} (default)"

            if is_pair:
                data = (
                    f"st:{idx}:{resolution_idx}:{int(use_last_frame)}:"
                    f"{lora_idx}:{high_idx}:{low_idx}:{model_type}:{model_idx}:{job_id}"
                )
            else:
                data = (
                    f"st:{idx}:{resolution_idx}:{int(use_last_frame)}:"
                    f"{lora_idx}:{weight_idx}:{model_type}:{model_idx}:{job_id}"
                )
            row.append({"text": label, "callback_data": data})
        rows.append(row)

    return json.dumps({"inline_keyboard": rows})


def build_prompt_keyboard(
    job_id: str,
    *,
    is_pair: bool,
    lora_idx: int,
    model_type: str,
    model_idx: int,
    use_last_frame: bool,
    resolution_idx: int,
    steps_idx: int,
    weight_idx: Optional[int] = None,
    high_idx: Optional[int] = None,
    low_idx: Optional[int] = None,
) -> str:
    if is_pair:
        yes_data = (
            f"pu:1:{steps_idx}:{resolution_idx}:{int(use_last_frame)}:"
            f"{lora_idx}:{high_idx}:{low_idx}:{model_type}:{model_idx}:{job_id}"
        )
        no_data = (
            f"pu:0:{steps_idx}:{resolution_idx}:{int(use_last_frame)}:"
            f"{lora_idx}:{high_idx}:{low_idx}:{model_type}:{model_idx}:{job_id}"
        )
    else:
        yes_data = (
            f"pu:1:{steps_idx}:{resolution_idx}:{int(use_last_frame)}:"
            f"{lora_idx}:{weight_idx}:{model_type}:{model_idx}:{job_id}"
        )
        no_data = (
            f"pu:0:{steps_idx}:{resolution_idx}:{int(use_last_frame)}:"
            f"{lora_idx}:{weight_idx}:{model_type}:{model_idx}:{job_id}"
        )

    rows = [
        [
            {"text": "Prompt: LoRA (default)", "callback_data": yes_data},
            {"text": "Prompt: Default", "callback_data": no_data},
        ]
    ]
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


async def prompt_last_frame(
    *,
    chat_id: int,
    message_id: int,
    label: str,
    job_id: str,
    is_pair: bool,
    lora_idx: int,
    model_type: str,
    model_idx: int,
    weight_idx: Optional[int] = None,
    high_idx: Optional[int] = None,
    low_idx: Optional[int] = None,
) -> None:
    if not (chat_id and message_id):
        return
    reply_markup = build_last_frame_keyboard(
        job_id,
        is_pair=is_pair,
        lora_idx=lora_idx,
        model_type=model_type,
        model_idx=model_idx,
        weight_idx=weight_idx,
        high_idx=high_idx,
        low_idx=low_idx,
    )
    await edit_message_text(chat_id, message_id, f"{label}: použiť last frame?", reply_markup)


async def prompt_resolution(
    *,
    chat_id: int,
    message_id: int,
    label: str,
    job_id: str,
    is_pair: bool,
    lora_idx: int,
    model_type: str,
    model_idx: int,
    use_last_frame: bool,
    weight_idx: Optional[int] = None,
    high_idx: Optional[int] = None,
    low_idx: Optional[int] = None,
) -> None:
    if not (chat_id and message_id):
        return
    reply_markup = build_resolution_keyboard(
        job_id,
        is_pair=is_pair,
        lora_idx=lora_idx,
        model_type=model_type,
        model_idx=model_idx,
        use_last_frame=use_last_frame,
        weight_idx=weight_idx,
        high_idx=high_idx,
        low_idx=low_idx,
    )
    await edit_message_text(chat_id, message_id, f"{label}: vyber rozlisenie", reply_markup)


async def prompt_steps(
    *,
    chat_id: int,
    message_id: int,
    label: str,
    job_id: str,
    is_pair: bool,
    lora_idx: int,
    model_type: str,
    model_idx: int,
    use_last_frame: bool,
    resolution_idx: int,
    weight_idx: Optional[int] = None,
    high_idx: Optional[int] = None,
    low_idx: Optional[int] = None,
) -> None:
    if not (chat_id and message_id):
        return
    reply_markup = build_steps_keyboard(
        job_id,
        is_pair=is_pair,
        lora_idx=lora_idx,
        model_type=model_type,
        model_idx=model_idx,
        use_last_frame=use_last_frame,
        resolution_idx=resolution_idx,
        weight_idx=weight_idx,
        high_idx=high_idx,
        low_idx=low_idx,
    )
    await edit_message_text(chat_id, message_id, f"{label}: vyber pocet krokov", reply_markup)


async def prompt_use_prompt(
    *,
    chat_id: int,
    message_id: int,
    label: str,
    job_id: str,
    is_pair: bool,
    lora_idx: int,
    model_type: str,
    model_idx: int,
    use_last_frame: bool,
    resolution_idx: int,
    steps_idx: int,
    weight_idx: Optional[int] = None,
    high_idx: Optional[int] = None,
    low_idx: Optional[int] = None,
) -> None:
    if not (chat_id and message_id):
        return
    reply_markup = build_prompt_keyboard(
        job_id,
        is_pair=is_pair,
        lora_idx=lora_idx,
        model_type=model_type,
        model_idx=model_idx,
        use_last_frame=use_last_frame,
        resolution_idx=resolution_idx,
        steps_idx=steps_idx,
        weight_idx=weight_idx,
        high_idx=high_idx,
        low_idx=low_idx,
    )
    await edit_message_text(chat_id, message_id, f"{label}: pouzit LoRA prompt?", reply_markup)


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


def _model_type_label(use_gguf: bool) -> str:
    return "GGUF" if use_gguf else "WAN"


async def _submit_single_job(
    *,
    job_id: str,
    lora_key: str,
    cfg: Dict[str, Any],
    weight: float,
    use_gguf: bool,
    use_last_frame: bool,
    video_width: Optional[int],
    video_height: Optional[int],
    total_steps: Optional[int],
    positive_prompt: Optional[str],
    model_label: Optional[str],
    model_high_filename: Optional[str],
    model_low_filename: Optional[str],
    chat_id: int,
    message_id: int,
) -> None:
    row = set_queue(job_id, lora_key)
    label = str(cfg.get("label") or lora_key)
    prompt_text = positive_prompt if isinstance(positive_prompt, str) else cfg.get("positive")
    payload: Dict[str, Any] = {
        "job_id": job_id,
        "chat_id": int(row["chat_id"]),
        "input_file_id": row["input_file_id"],
        "lora_key": lora_key,
        "lora_label": label,
        "lora_type": (cfg.get("type") or "single"),
        "positive_prompt": prompt_text,
        "use_gguf": use_gguf,
        "use_last_frame": use_last_frame,
        "video_width": video_width,
        "video_height": video_height,
        "total_steps": total_steps,
        "lora_filename": cfg.get("filename"),
        "lora_strength": weight,
    }
    if model_label:
        payload["model_label"] = model_label
    if model_high_filename:
        payload["model_high_filename"] = model_high_filename
    if model_low_filename:
        payload["model_low_filename"] = model_low_filename

    runpod_id = await submit_runpod(payload)
    if runpod_id:
        set_runpod_request_id(job_id, runpod_id)

    model_type_label = _model_type_label(use_gguf)
    suffix = f": {model_label}" if model_label else ""
    await edit_placeholder(
        int(row["chat_id"]),
        int(row["placeholder_message_id"]),
        f"Renderujem ({label}, {model_type_label}{suffix})…",
    )
    if chat_id and message_id:
        await edit_keyboard(chat_id, message_id, json.dumps({"inline_keyboard": []}))


async def _submit_pair_job(
    *,
    job_id: str,
    lora_key: str,
    cfg: Dict[str, Any],
    high_weight: float,
    low_weight: float,
    use_gguf: bool,
    use_last_frame: bool,
    video_width: Optional[int],
    video_height: Optional[int],
    total_steps: Optional[int],
    positive_prompt: Optional[str],
    model_label: Optional[str],
    model_high_filename: Optional[str],
    model_low_filename: Optional[str],
    chat_id: int,
    message_id: int,
) -> None:
    row = set_queue(job_id, lora_key)
    label = str(cfg.get("label") or lora_key)
    prompt_text = positive_prompt if isinstance(positive_prompt, str) else cfg.get("positive")
    payload: Dict[str, Any] = {
        "job_id": job_id,
        "chat_id": int(row["chat_id"]),
        "input_file_id": row["input_file_id"],
        "lora_key": lora_key,
        "lora_label": label,
        "lora_type": (cfg.get("type") or "single"),
        "positive_prompt": prompt_text,
        "use_gguf": use_gguf,
        "use_last_frame": use_last_frame,
        "video_width": video_width,
        "video_height": video_height,
        "total_steps": total_steps,
        "lora_high_filename": cfg.get("high_filename"),
        "lora_high_strength": high_weight,
        "lora_low_filename": cfg.get("low_filename"),
        "lora_low_strength": low_weight,
    }
    if model_label:
        payload["model_label"] = model_label
    if model_high_filename:
        payload["model_high_filename"] = model_high_filename
    if model_low_filename:
        payload["model_low_filename"] = model_low_filename

    runpod_id = await submit_runpod(payload)
    if runpod_id:
        set_runpod_request_id(job_id, runpod_id)

    model_type_label = _model_type_label(use_gguf)
    suffix = f": {model_label}" if model_label else ""
    await edit_placeholder(
        int(row["chat_id"]),
        int(row["placeholder_message_id"]),
        f"Renderujem ({label}, {model_type_label}{suffix})…",
    )
    if chat_id and message_id:
        await edit_keyboard(chat_id, message_id, json.dumps({"inline_keyboard": []}))


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
    # - l:<idx>:<job_id>                       (select LoRA)
    # - p:<page>:<job_id>                      (LoRA page)
    # - pm:<model>:<page>:<lora_idx>:<w_idx>:<job_id> (model page, single)
    # - pm:<model>:<page>:<lora_idx>:<h_idx>:<l_idx>:<job_id> (model page, pair)
    # - ws:<lora_idx>:<w_idx>:<job_id>         (single weight)
    # - wh:<lora_idx>:<h_idx>:<job_id>         (pair high)
    # - wl:<lora_idx>:<h_idx>:<l_idx>:<job_id> (pair low)
    # - ms:<lora_idx>:<w_idx>:<model>:<job_id> (select model type, single)
    # - mp:<lora_idx>:<h_idx>:<l_idx>:<model>:<job_id> (select model type, pair)
    # - mm:<lora_idx>:<w_idx>:<model>:<m_idx>:<job_id> (select model, single)
    # - mm:<lora_idx>:<h_idx>:<l_idx>:<model>:<m_idx>:<job_id> (select model, pair)
    # - lf:<on>:<lora_idx>:<w_idx>:<model>:<m_idx>:<job_id> (last frame, single)
    # - lf:<on>:<lora_idx>:<h_idx>:<l_idx>:<model>:<m_idx>:<job_id> (last frame, pair)
    # - rs:<r_idx>:<on>:<lora_idx>:<w_idx>:<model>:<m_idx>:<job_id> (resolution, single)
    # - rs:<r_idx>:<on>:<lora_idx>:<h_idx>:<l_idx>:<model>:<m_idx>:<job_id> (resolution, pair)
    # - st:<s_idx>:<r_idx>:<on>:<lora_idx>:<w_idx>:<model>:<m_idx>:<job_id> (steps, single)
    # - st:<s_idx>:<r_idx>:<on>:<lora_idx>:<h_idx>:<l_idx>:<model>:<m_idx>:<job_id> (steps, pair)
    # - pu:<p>:<s_idx>:<r_idx>:<on>:<lora_idx>:<w_idx>:<model>:<m_idx>:<job_id> (prompt, single)
    # - pu:<p>:<s_idx>:<r_idx>:<on>:<lora_idx>:<h_idx>:<l_idx>:<model>:<m_idx>:<job_id> (prompt, pair)
    # - noop:<job_id>                          (do nothing)
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

    if tag == "pm":
        if len(parts) not in (6, 7):
            return
        try:
            model_type = parts[1]
            page = int(parts[2])
            lora_idx = int(parts[3])
        except Exception:
            return

        if model_type not in ("wan", "gguf"):
            return
        if not (chat_id and message_id):
            return

        if len(parts) == 6:
            try:
                weight_idx = int(parts[4])
                job_id = parts[5]
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

            reply_markup = build_unet_keyboard(
                job_id,
                model_type,
                is_pair=False,
                lora_idx=lora_idx,
                weight_idx=weight_idx,
                page=page,
            )
            await edit_keyboard(chat_id, message_id, reply_markup)
            return

        try:
            high_idx = int(parts[4])
            low_idx = int(parts[5])
            job_id = parts[6]
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

        reply_markup = build_unet_keyboard(
            job_id,
            model_type,
            is_pair=True,
            lora_idx=lora_idx,
            high_idx=high_idx,
            low_idx=low_idx,
            page=page,
        )
        await edit_keyboard(chat_id, message_id, reply_markup)
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

    if tag == "mm":
        if len(parts) not in (6, 7):
            return
        try:
            lora_idx = int(parts[1])
        except Exception:
            return

        if len(parts) == 6:
            try:
                weight_idx = int(parts[2])
                model_type = parts[3]
                model_idx = int(parts[4])
                job_id = parts[5]
            except Exception:
                return

            if model_type not in ("wan", "gguf"):
                return

            lora_key, cfg = _get_lora_cfg_by_index(lora_idx)
            if not cfg:
                return
            if str(cfg.get("type") or "single").lower() == "pair":
                return

            options, _ = _weight_options_for(cfg, "single")
            if weight_idx < 0 or weight_idx >= len(options):
                return

            model_key, model_cfg = _get_model_cfg_by_index(model_type, model_idx)
            if not model_cfg:
                return

            label = str(cfg.get("label") or lora_key)
            if chat_id and message_id:
                await prompt_last_frame(
                    chat_id=chat_id,
                    message_id=message_id,
                    label=label,
                    job_id=job_id,
                    is_pair=False,
                    lora_idx=lora_idx,
                    weight_idx=weight_idx,
                    model_type=model_type,
                    model_idx=model_idx,
                )
            else:
                await _submit_single_job(
                    job_id=job_id,
                    lora_key=lora_key,
                    cfg=cfg,
                    weight=options[weight_idx],
                    use_gguf=(model_type == "gguf"),
                    use_last_frame=False,
                    video_width=RESOLUTION_OPTIONS[DEFAULT_RESOLUTION_IDX]["width"],
                    video_height=RESOLUTION_OPTIONS[DEFAULT_RESOLUTION_IDX]["height"],
                    total_steps=DEFAULT_STEPS,
                    positive_prompt=cfg.get("positive"),
                    model_label=str(model_cfg.get("label") or model_key),
                    model_high_filename=model_cfg.get("high_filename"),
                    model_low_filename=model_cfg.get("low_filename"),
                    chat_id=chat_id,
                    message_id=message_id,
                )
            return

        try:
            high_idx = int(parts[2])
            low_idx = int(parts[3])
            model_type = parts[4]
            model_idx = int(parts[5])
            job_id = parts[6]
        except Exception:
            return

        if model_type not in ("wan", "gguf"):
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

        model_key, model_cfg = _get_model_cfg_by_index(model_type, model_idx)
        if not model_cfg:
            return

        label = str(cfg.get("label") or lora_key)
        if chat_id and message_id:
            await prompt_last_frame(
                chat_id=chat_id,
                message_id=message_id,
                label=label,
                job_id=job_id,
                is_pair=True,
                lora_idx=lora_idx,
                high_idx=high_idx,
                low_idx=low_idx,
                model_type=model_type,
                model_idx=model_idx,
            )
        else:
            await _submit_pair_job(
                job_id=job_id,
                lora_key=lora_key,
                cfg=cfg,
                high_weight=options_high[high_idx],
                low_weight=options_low[low_idx],
                use_gguf=(model_type == "gguf"),
                use_last_frame=False,
                video_width=RESOLUTION_OPTIONS[DEFAULT_RESOLUTION_IDX]["width"],
                video_height=RESOLUTION_OPTIONS[DEFAULT_RESOLUTION_IDX]["height"],
                total_steps=DEFAULT_STEPS,
                positive_prompt=cfg.get("positive"),
                model_label=str(model_cfg.get("label") or model_key),
                model_high_filename=model_cfg.get("high_filename"),
                model_low_filename=model_cfg.get("low_filename"),
                chat_id=chat_id,
                message_id=message_id,
            )
        return

    if tag == "lf":
        if len(parts) not in (7, 8):
            return

        use_last_frame = str(parts[1]).strip().lower() in ("1", "true", "yes", "y", "on")

        try:
            lora_idx = int(parts[2])
        except Exception:
            return

        lora_key, cfg = _get_lora_cfg_by_index(lora_idx)
        if not cfg:
            return
        lora_type = str(cfg.get("type") or "single").lower()
        label = str(cfg.get("label") or lora_key)

        if len(parts) == 7:
            if lora_type == "pair":
                return
            try:
                weight_idx = int(parts[3])
                model_type = parts[4]
                model_idx = int(parts[5])
                job_id = parts[6]
            except Exception:
                return

            if model_type not in ("wan", "gguf"):
                return

            options, _ = _weight_options_for(cfg, "single")
            if weight_idx < 0 or weight_idx >= len(options):
                return

            await prompt_resolution(
                chat_id=chat_id,
                message_id=message_id,
                label=label,
                job_id=job_id,
                is_pair=False,
                lora_idx=lora_idx,
                weight_idx=weight_idx,
                model_type=model_type,
                model_idx=model_idx,
                use_last_frame=use_last_frame,
            )
            return

        if lora_type != "pair":
            return
        try:
            high_idx = int(parts[3])
            low_idx = int(parts[4])
            model_type = parts[5]
            model_idx = int(parts[6])
            job_id = parts[7]
        except Exception:
            return

        if model_type not in ("wan", "gguf"):
            return

        options_high, _ = _weight_options_for(cfg, "high")
        options_low, _ = _weight_options_for(cfg, "low")
        if high_idx < 0 or high_idx >= len(options_high):
            return
        if low_idx < 0 or low_idx >= len(options_low):
            return

        await prompt_resolution(
            chat_id=chat_id,
            message_id=message_id,
            label=label,
            job_id=job_id,
            is_pair=True,
            lora_idx=lora_idx,
            high_idx=high_idx,
            low_idx=low_idx,
            model_type=model_type,
            model_idx=model_idx,
            use_last_frame=use_last_frame,
        )
        return

    if tag == "rs":
        if len(parts) not in (8, 9):
            return

        try:
            resolution_idx = int(parts[1])
        except Exception:
            return
        if _resolution_by_index(resolution_idx) is None:
            return

        use_last_frame = str(parts[2]).strip().lower() in ("1", "true", "yes", "y", "on")

        try:
            lora_idx = int(parts[3])
        except Exception:
            return

        lora_key, cfg = _get_lora_cfg_by_index(lora_idx)
        if not cfg:
            return
        lora_type = str(cfg.get("type") or "single").lower()
        label = str(cfg.get("label") or lora_key)

        if len(parts) == 8:
            if lora_type == "pair":
                return
            try:
                weight_idx = int(parts[4])
                model_type = parts[5]
                model_idx = int(parts[6])
                job_id = parts[7]
            except Exception:
                return

            if model_type not in ("wan", "gguf"):
                return

            options, _ = _weight_options_for(cfg, "single")
            if weight_idx < 0 or weight_idx >= len(options):
                return

            await prompt_steps(
                chat_id=chat_id,
                message_id=message_id,
                label=label,
                job_id=job_id,
                is_pair=False,
                lora_idx=lora_idx,
                weight_idx=weight_idx,
                model_type=model_type,
                model_idx=model_idx,
                use_last_frame=use_last_frame,
                resolution_idx=resolution_idx,
            )
            return

        if lora_type != "pair":
            return
        try:
            high_idx = int(parts[4])
            low_idx = int(parts[5])
            model_type = parts[6]
            model_idx = int(parts[7])
            job_id = parts[8]
        except Exception:
            return

        if model_type not in ("wan", "gguf"):
            return

        options_high, _ = _weight_options_for(cfg, "high")
        options_low, _ = _weight_options_for(cfg, "low")
        if high_idx < 0 or high_idx >= len(options_high):
            return
        if low_idx < 0 or low_idx >= len(options_low):
            return

        await prompt_steps(
            chat_id=chat_id,
            message_id=message_id,
            label=label,
            job_id=job_id,
            is_pair=True,
            lora_idx=lora_idx,
            high_idx=high_idx,
            low_idx=low_idx,
            model_type=model_type,
            model_idx=model_idx,
            use_last_frame=use_last_frame,
            resolution_idx=resolution_idx,
        )
        return

    if tag == "st":
        if len(parts) not in (9, 10):
            return

        try:
            steps_idx = int(parts[1])
            resolution_idx = int(parts[2])
        except Exception:
            return
        if _resolution_by_index(resolution_idx) is None:
            return
        if _steps_by_index(steps_idx) is None:
            return

        use_last_frame = str(parts[3]).strip().lower() in ("1", "true", "yes", "y", "on")

        try:
            lora_idx = int(parts[4])
        except Exception:
            return

        lora_key, cfg = _get_lora_cfg_by_index(lora_idx)
        if not cfg:
            return
        lora_type = str(cfg.get("type") or "single").lower()
        label = str(cfg.get("label") or lora_key)

        if len(parts) == 9:
            if lora_type == "pair":
                return
            try:
                weight_idx = int(parts[5])
                model_type = parts[6]
                model_idx = int(parts[7])
                job_id = parts[8]
            except Exception:
                return

            if model_type not in ("wan", "gguf"):
                return

            options, _ = _weight_options_for(cfg, "single")
            if weight_idx < 0 or weight_idx >= len(options):
                return

            await prompt_use_prompt(
                chat_id=chat_id,
                message_id=message_id,
                label=label,
                job_id=job_id,
                is_pair=False,
                lora_idx=lora_idx,
                weight_idx=weight_idx,
                model_type=model_type,
                model_idx=model_idx,
                use_last_frame=use_last_frame,
                resolution_idx=resolution_idx,
                steps_idx=steps_idx,
            )
            return

        if lora_type != "pair":
            return
        try:
            high_idx = int(parts[5])
            low_idx = int(parts[6])
            model_type = parts[7]
            model_idx = int(parts[8])
            job_id = parts[9]
        except Exception:
            return

        if model_type not in ("wan", "gguf"):
            return

        options_high, _ = _weight_options_for(cfg, "high")
        options_low, _ = _weight_options_for(cfg, "low")
        if high_idx < 0 or high_idx >= len(options_high):
            return
        if low_idx < 0 or low_idx >= len(options_low):
            return

        await prompt_use_prompt(
            chat_id=chat_id,
            message_id=message_id,
            label=label,
            job_id=job_id,
            is_pair=True,
            lora_idx=lora_idx,
            high_idx=high_idx,
            low_idx=low_idx,
            model_type=model_type,
            model_idx=model_idx,
            use_last_frame=use_last_frame,
            resolution_idx=resolution_idx,
            steps_idx=steps_idx,
        )
        return

    if tag == "pu":
        if len(parts) not in (10, 11):
            return

        use_prompt = str(parts[1]).strip().lower() in ("1", "true", "yes", "y", "on")
        try:
            steps_idx = int(parts[2])
            resolution_idx = int(parts[3])
        except Exception:
            return
        if _resolution_by_index(resolution_idx) is None:
            return
        steps = _steps_by_index(steps_idx)
        if steps is None:
            return

        use_last_frame = str(parts[4]).strip().lower() in ("1", "true", "yes", "y", "on")

        try:
            lora_idx = int(parts[5])
        except Exception:
            return

        lora_key, cfg = _get_lora_cfg_by_index(lora_idx)
        if not cfg:
            return
        lora_type = str(cfg.get("type") or "single").lower()

        resolution = _resolution_by_index(resolution_idx)
        if not resolution:
            return
        video_width = int(resolution["width"])
        video_height = int(resolution["height"])
        prompt_text = (cfg.get("positive") or DEFAULT_PROMPT) if use_prompt else DEFAULT_PROMPT

        if len(parts) == 10:
            if lora_type == "pair":
                return
            try:
                weight_idx = int(parts[6])
                model_type = parts[7]
                model_idx = int(parts[8])
                job_id = parts[9]
            except Exception:
                return

            if model_type not in ("wan", "gguf"):
                return

            options, _ = _weight_options_for(cfg, "single")
            if weight_idx < 0 or weight_idx >= len(options):
                return

            model_label = None
            model_high_filename = None
            model_low_filename = None
            if model_idx >= 0:
                model_key, model_cfg = _get_model_cfg_by_index(model_type, model_idx)
                if not model_cfg:
                    return
                model_label = str(model_cfg.get("label") or model_key)
                model_high_filename = model_cfg.get("high_filename")
                model_low_filename = model_cfg.get("low_filename")

            await _submit_single_job(
                job_id=job_id,
                lora_key=lora_key,
                cfg=cfg,
                weight=options[weight_idx],
                use_gguf=(model_type == "gguf"),
                use_last_frame=use_last_frame,
                video_width=video_width,
                video_height=video_height,
                total_steps=steps,
                positive_prompt=prompt_text,
                model_label=model_label,
                model_high_filename=model_high_filename,
                model_low_filename=model_low_filename,
                chat_id=chat_id,
                message_id=message_id,
            )
            return

        if lora_type != "pair":
            return
        try:
            high_idx = int(parts[6])
            low_idx = int(parts[7])
            model_type = parts[8]
            model_idx = int(parts[9])
            job_id = parts[10]
        except Exception:
            return

        if model_type not in ("wan", "gguf"):
            return

        options_high, _ = _weight_options_for(cfg, "high")
        options_low, _ = _weight_options_for(cfg, "low")
        if high_idx < 0 or high_idx >= len(options_high):
            return
        if low_idx < 0 or low_idx >= len(options_low):
            return

        model_label = None
        model_high_filename = None
        model_low_filename = None
        if model_idx >= 0:
            model_key, model_cfg = _get_model_cfg_by_index(model_type, model_idx)
            if not model_cfg:
                return
            model_label = str(model_cfg.get("label") or model_key)
            model_high_filename = model_cfg.get("high_filename")
            model_low_filename = model_cfg.get("low_filename")

        await _submit_pair_job(
            job_id=job_id,
            lora_key=lora_key,
            cfg=cfg,
            high_weight=options_high[high_idx],
            low_weight=options_low[low_idx],
            use_gguf=(model_type == "gguf"),
            use_last_frame=use_last_frame,
            video_width=video_width,
            video_height=video_height,
            total_steps=steps,
            positive_prompt=prompt_text,
            model_label=model_label,
            model_high_filename=model_high_filename,
            model_low_filename=model_low_filename,
            chat_id=chat_id,
            message_id=message_id,
        )
        return

    if tag == "ms":
        if len(parts) != 5:
            return
        try:
            lora_idx = int(parts[1])
            weight_idx = int(parts[2])
            model_type = parts[3]
            job_id = parts[4]
        except Exception:
            return

        if model_type not in ("wan", "gguf"):
            return

        lora_key, cfg = _get_lora_cfg_by_index(lora_idx)
        if not cfg:
            return
        if str(cfg.get("type") or "single").lower() == "pair":
            return

        options, _ = _weight_options_for(cfg, "single")
        if weight_idx < 0 or weight_idx >= len(options):
            return

        use_gguf = model_type == "gguf"
        model_keys = MODEL_KEYS_BY_TYPE.get(model_type) or []
        if len(model_keys) > 1 and chat_id and message_id:
            label = str(cfg.get("label") or lora_key)
            model_type_label = _model_type_label(use_gguf)
            reply_markup = build_unet_keyboard(
                job_id,
                model_type,
                is_pair=False,
                lora_idx=lora_idx,
                weight_idx=weight_idx,
                page=0,
            )
            await edit_message_text(chat_id, message_id, f"{label}: vyber {model_type_label} model", reply_markup)
            return

        model_idx = 0 if model_keys else -1
        label = str(cfg.get("label") or lora_key)
        if chat_id and message_id:
            await prompt_last_frame(
                chat_id=chat_id,
                message_id=message_id,
                label=label,
                job_id=job_id,
                is_pair=False,
                lora_idx=lora_idx,
                weight_idx=weight_idx,
                model_type=model_type,
                model_idx=model_idx,
            )
            return

        model_label = None
        model_high_filename = None
        model_low_filename = None
        if model_idx >= 0:
            model_key, model_cfg = _get_model_cfg_by_index(model_type, model_idx)
            if model_cfg:
                model_label = str(model_cfg.get("label") or model_key)
                model_high_filename = model_cfg.get("high_filename")
                model_low_filename = model_cfg.get("low_filename")

        await _submit_single_job(
            job_id=job_id,
            lora_key=lora_key,
            cfg=cfg,
            weight=options[weight_idx],
            use_gguf=use_gguf,
            use_last_frame=False,
            video_width=RESOLUTION_OPTIONS[DEFAULT_RESOLUTION_IDX]["width"],
            video_height=RESOLUTION_OPTIONS[DEFAULT_RESOLUTION_IDX]["height"],
            total_steps=DEFAULT_STEPS,
            positive_prompt=cfg.get("positive"),
            model_label=model_label,
            model_high_filename=model_high_filename,
            model_low_filename=model_low_filename,
            chat_id=chat_id,
            message_id=message_id,
        )
        return

    if tag == "mp":
        if len(parts) != 6:
            return
        try:
            lora_idx = int(parts[1])
            high_idx = int(parts[2])
            low_idx = int(parts[3])
            model_type = parts[4]
            job_id = parts[5]
        except Exception:
            return

        if model_type not in ("wan", "gguf"):
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

        use_gguf = model_type == "gguf"
        model_keys = MODEL_KEYS_BY_TYPE.get(model_type) or []
        if len(model_keys) > 1 and chat_id and message_id:
            label = str(cfg.get("label") or lora_key)
            model_type_label = _model_type_label(use_gguf)
            reply_markup = build_unet_keyboard(
                job_id,
                model_type,
                is_pair=True,
                lora_idx=lora_idx,
                high_idx=high_idx,
                low_idx=low_idx,
                page=0,
            )
            await edit_message_text(chat_id, message_id, f"{label}: vyber {model_type_label} model", reply_markup)
            return

        model_idx = 0 if model_keys else -1
        label = str(cfg.get("label") or lora_key)
        if chat_id and message_id:
            await prompt_last_frame(
                chat_id=chat_id,
                message_id=message_id,
                label=label,
                job_id=job_id,
                is_pair=True,
                lora_idx=lora_idx,
                high_idx=high_idx,
                low_idx=low_idx,
                model_type=model_type,
                model_idx=model_idx,
            )
            return

        model_label = None
        model_high_filename = None
        model_low_filename = None
        if model_idx >= 0:
            model_key, model_cfg = _get_model_cfg_by_index(model_type, model_idx)
            if model_cfg:
                model_label = str(model_cfg.get("label") or model_key)
                model_high_filename = model_cfg.get("high_filename")
                model_low_filename = model_cfg.get("low_filename")

        await _submit_pair_job(
            job_id=job_id,
            lora_key=lora_key,
            cfg=cfg,
            high_weight=options_high[high_idx],
            low_weight=options_low[low_idx],
            use_gguf=use_gguf,
            use_last_frame=False,
            video_width=RESOLUTION_OPTIONS[DEFAULT_RESOLUTION_IDX]["width"],
            video_height=RESOLUTION_OPTIONS[DEFAULT_RESOLUTION_IDX]["height"],
            total_steps=DEFAULT_STEPS,
            positive_prompt=cfg.get("positive"),
            model_label=model_label,
            model_high_filename=model_high_filename,
            model_low_filename=model_low_filename,
            chat_id=chat_id,
            message_id=message_id,
        )
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
