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
RATE_LIMIT_SECONDS = int(os.environ.get("RATE_LIMIT_SECONDS", "5"))
DEDUP_TTL_SECONDS = int(os.environ.get("DEDUP_TTL_SECONDS", "600"))

LORA_CATALOG_PATH = os.environ.get("LORA_CATALOG_PATH", "/app/loras.json")
LORA_GROUPS_PATH = os.environ.get("LORA_GROUPS_PATH")
if not LORA_GROUPS_PATH:
    LORA_GROUPS_PATH = os.path.join(os.path.dirname(LORA_CATALOG_PATH), "lora_groups.json")
MODEL_CATALOG_PATH = os.environ.get("MODEL_CATALOG_PATH", "/app/models.json")

# 4 buttons max (2x2 grid) + paging arrows
PAGE_SIZE = int(os.environ.get("LORA_PAGE_SIZE", "4"))  # keep 4 by default
MODEL_PAGE_SIZE = int(os.environ.get("MODEL_PAGE_SIZE", "4"))
WEIGHT_OPTIONS_RAW = os.environ.get("LORA_WEIGHT_OPTIONS", "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
BATCH_TEST_WEIGHTS_RAW = os.environ.get("BATCH_TEST_WEIGHTS", "0.5,0.7,0.9")
BATCH_TEST_MODEL_KEY = os.environ.get("BATCH_TEST_MODEL_KEY", "gguf_tastysin_v8")
BATCH_TEST_VIDEO_WIDTH = int(os.environ.get("BATCH_TEST_VIDEO_WIDTH", "480"))
BATCH_TEST_VIDEO_HEIGHT = int(os.environ.get("BATCH_TEST_VIDEO_HEIGHT", "640"))
BATCH_TEST_STEPS = int(os.environ.get("BATCH_TEST_STEPS", "6"))
BATCH_TEST_RIFE = int(os.environ.get("BATCH_TEST_RIFE", "0"))
BATCH_TEST_USE_LAST_FRAME_RAW = os.environ.get("BATCH_TEST_USE_LAST_FRAME", "0")
BATCH_TEST_USE_PROMPT_RAW = os.environ.get("BATCH_TEST_USE_PROMPT", "1")
TEST5S_WEIGHT = 0.7
TEST5S_VIDEO_WIDTH = 480
TEST5S_VIDEO_HEIGHT = 720
TEST5S_STEPS = 4
TEST5S_RIFE_MULTIPLIER = 0
TEST5S_MEDIA_GROUP_SIZE = int(os.environ.get("TEST5S_MEDIA_GROUP_SIZE", "3"))
TEST5S_MEDIA_GROUP_TTL_SECONDS = int(os.environ.get("TEST5S_MEDIA_GROUP_TTL_SECONDS", "5"))
TEST5S_GROUP_TTL_SECONDS = int(os.environ.get("TEST5S_GROUP_TTL_SECONDS", "3600"))


def load_lora_catalog() -> Dict[str, Any]:
    with open(LORA_CATALOG_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not data:
        raise RuntimeError("loras.json must be a non-empty JSON object")
    return data


def load_lora_group_catalog() -> Dict[str, Any]:
    if not os.path.exists(LORA_GROUPS_PATH):
        return {}
    with open(LORA_GROUPS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise RuntimeError("lora_groups.json must be a JSON object")
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
LORA_GROUP_CATALOG = load_lora_group_catalog()
LORA_GROUP_KEYS = sorted(LORA_GROUP_CATALOG.keys())
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
DEFAULT_MODEL_KEY = "gguf_tastysin_v8"
DEFAULT_MODEL_TYPE = "gguf"

rate_limit_cache: Dict[int, float] = {}
dedup_cache: Dict[int, float] = {}  # update_id -> timestamp
EXTENDED_SESSION_TTL_SECONDS = int(os.environ.get("EXTENDED_SESSION_TTL_SECONDS", "3600"))
extended_sessions: Dict[str, Dict[str, Any]] = {}
WORKFLOW4_PARTS = 4
workflow4_sessions: Dict[str, Dict[str, Any]] = {}
test5s_album_sessions: Dict[str, Dict[str, Any]] = {}
test5s_group_jobs: Dict[str, Dict[str, Any]] = {}


def _ext_session_get(job_id: str) -> Optional[Dict[str, Any]]:
    sess = extended_sessions.get(job_id)
    if not sess:
        return None
    updated_at = float(sess.get("updated_at") or 0.0)
    if time.time() - updated_at > EXTENDED_SESSION_TTL_SECONDS:
        extended_sessions.pop(job_id, None)
        return None
    return sess


def _ext_session_touch(sess: Dict[str, Any]) -> None:
    sess["updated_at"] = time.time()


def _ext_session_clear(job_id: str) -> None:
    extended_sessions.pop(job_id, None)


def _wf4_session_get(job_id: str) -> Optional[Dict[str, Any]]:
    sess = workflow4_sessions.get(job_id)
    if not sess:
        return None
    updated_at = float(sess.get("updated_at") or 0.0)
    if time.time() - updated_at > EXTENDED_SESSION_TTL_SECONDS:
        workflow4_sessions.pop(job_id, None)
        return None
    return sess


def _wf4_session_touch(sess: Dict[str, Any]) -> None:
    sess["updated_at"] = time.time()


def _wf4_session_clear(job_id: str) -> None:
    workflow4_sessions.pop(job_id, None)


def _test5s_group_get(job_id: str) -> Optional[List[str]]:
    entry = test5s_group_jobs.get(job_id)
    if not entry:
        return None
    created_at = float(entry.get("created_at") or 0.0)
    if time.time() - created_at > TEST5S_GROUP_TTL_SECONDS:
        test5s_group_jobs.pop(job_id, None)
        return None
    job_ids = entry.get("job_ids") or []
    if not job_ids:
        test5s_group_jobs.pop(job_id, None)
        return None
    return job_ids


def _test5s_group_set(leader_job_id: str, job_ids: List[str]) -> None:
    test5s_group_jobs[leader_job_id] = {"job_ids": list(job_ids), "created_at": time.time()}


def db_conn():
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)


def _parse_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return default


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
BATCH_TEST_WEIGHTS = _parse_weight_options(BATCH_TEST_WEIGHTS_RAW) or [0.5, 0.7, 0.9]
BATCH_TEST_USE_LAST_FRAME = _parse_bool(BATCH_TEST_USE_LAST_FRAME_RAW, True)
BATCH_TEST_USE_PROMPT = _parse_bool(BATCH_TEST_USE_PROMPT_RAW, True)

RESOLUTION_OPTIONS = [
    {"label": "720x1280", "width": 720, "height": 1280},
    {"label": "480x720", "width": 480, "height": 720},
    {"label": "480x640", "width": 480, "height": 640},
]
DEFAULT_RESOLUTION_IDX = 0

STEPS_OPTIONS = [4, 6, 8, 10, 12, 14]
DEFAULT_STEPS = 12

DEFAULT_PROMPT = (
    "face remains consistent across frames, subtle camera drift only, "
    "minimal head movement, micro-expression only, no dramatic rotation, no sudden tilt"
)

DRIFT_PRESETS = [
    {
        "key": "default",
        "label": "Drift: Default",
        "speed_shift": None,
        "denoise": None,
        "overlap": None,
    },
    {
        "key": "reduce",
        "label": "Drift: Reduced",
        "speed_shift": 2,
        "denoise": 0.7,
        "overlap": 2,
    },
]
DRIFT_PRESETS_BY_KEY = {p["key"]: p for p in DRIFT_PRESETS}

INTERPOLATION_OPTIONS = [0, 4, 3, 2, 1]
DEFAULT_INTERPOLATION_CLASSIC = 3
DEFAULT_INTERPOLATION_NEW = 4


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


def _normalize_interpolation_value(value: Any, default_value: int) -> int:
    try:
        v = int(value)
    except Exception:
        return default_value
    return v if v in INTERPOLATION_OPTIONS else default_value


def _interpolation_label(value: int, default_value: int) -> str:
    label = "OFF" if value <= 0 else f"{value}x"
    if value == default_value:
        label = f"{label} (default)"
    return label


def _get_drift_preset(key: str) -> Optional[Dict[str, Any]]:
    if not key:
        return None
    return DRIFT_PRESETS_BY_KEY.get(key)


def _get_lora_cfg_by_index(idx: int) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    if idx < 0 or idx >= len(LORA_KEYS):
        return None, None
    lora_key = LORA_KEYS[idx]
    cfg = LORA_CATALOG.get(lora_key)
    if not isinstance(cfg, dict):
        return None, None
    return lora_key, cfg


def _get_lora_group_by_index(idx: int) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    if idx < 0 or idx >= len(LORA_GROUP_KEYS):
        return None, None
    group_key = LORA_GROUP_KEYS[idx]
    cfg = LORA_GROUP_CATALOG.get(group_key)
    if not isinstance(cfg, dict):
        return None, None
    return group_key, cfg


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


def _get_default_model_selection() -> Tuple[str, int, str, Dict[str, Any]]:
    gguf_keys = MODEL_KEYS_BY_TYPE.get(DEFAULT_MODEL_TYPE) or []
    if DEFAULT_MODEL_KEY in gguf_keys:
        cfg = MODEL_CATALOG.get(DEFAULT_MODEL_KEY)
        if isinstance(cfg, dict):
            return DEFAULT_MODEL_TYPE, gguf_keys.index(DEFAULT_MODEL_KEY), DEFAULT_MODEL_KEY, cfg

    for model_type in ("gguf", "wan"):
        keys = MODEL_KEYS_BY_TYPE.get(model_type) or []
        if not keys:
            continue
        model_key = keys[0]
        cfg = MODEL_CATALOG.get(model_key)
        if isinstance(cfg, dict):
            return model_type, 0, model_key, cfg

    raise RuntimeError("default_model_not_found")


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


def build_interpolation_keyboard(
    job_id: str,
    *,
    prompt_mode: int,
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
    rows = []
    for i in range(0, len(INTERPOLATION_OPTIONS), 2):
        row = []
        for j in range(2):
            if i + j >= len(INTERPOLATION_OPTIONS):
                break
            value = INTERPOLATION_OPTIONS[i + j]
            label = _interpolation_label(value, DEFAULT_INTERPOLATION_CLASSIC)
            if is_pair:
                data = (
                    f"ri:{value}:{prompt_mode}:{steps_idx}:{resolution_idx}:{int(use_last_frame)}:"
                    f"{lora_idx}:{high_idx}:{low_idx}:{model_type}:{model_idx}:{job_id}"
                )
            else:
                data = (
                    f"ri:{value}:{prompt_mode}:{steps_idx}:{resolution_idx}:{int(use_last_frame)}:"
                    f"{lora_idx}:{weight_idx}:{model_type}:{model_idx}:{job_id}"
                )
            row.append({"text": label, "callback_data": data})
        rows.append(row)
    return json.dumps({"inline_keyboard": rows})


def build_interpolation_keyboard_ext(
    job_id: str,
    *,
    tag_prefix: str = "e",
    default_value: int,
) -> str:
    rows = []
    for i in range(0, len(INTERPOLATION_OPTIONS), 2):
        row = []
        for j in range(2):
            if i + j >= len(INTERPOLATION_OPTIONS):
                break
            value = INTERPOLATION_OPTIONS[i + j]
            label = _interpolation_label(value, default_value)
            row.append({"text": label, "callback_data": f"{tag_prefix}ri:{value}:{job_id}"})
        rows.append(row)
    return json.dumps({"inline_keyboard": rows})


def build_mode_keyboard(job_id: str) -> str:
    rows = [
        [
            {"text": "Standard 5s", "callback_data": f"mode:std:{job_id}"},
            {"text": "Extended10s", "callback_data": f"mode:ext:{job_id}"},
        ]
    ]
    rows.append(
        [
            {"text": "Test5s", "callback_data": f"mode:test5s:{job_id}"},
            {"text": "Batch 5s", "callback_data": f"mode:batch:{job_id}"},
        ]
    )
    rows.append([{"text": "Workflow4", "callback_data": f"mode:new:{job_id}"}])
    return json.dumps({"inline_keyboard": rows})


def build_mode_keyboard_test5s(job_id: str) -> str:
    rows = [[{"text": "Test5s", "callback_data": f"mode:test5s:{job_id}"}]]
    return json.dumps({"inline_keyboard": rows})


def build_batch_weight_keyboard(job_id: str) -> str:
    options = list(BATCH_TEST_WEIGHTS)
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
            row.append({"text": label, "callback_data": f"bw:{i + j}:{job_id}"})
        rows.append(row)
    return json.dumps({"inline_keyboard": rows})


def build_lora_group_keyboard(job_id: str, page: int = 0, *, tag_prefix: str = "n") -> str:
    total = len(LORA_GROUP_KEYS)
    if total == 0:
        return json.dumps({"inline_keyboard": []})

    page_size = max(1, int(PAGE_SIZE))
    max_page = (total - 1) // page_size
    page = max(0, min(int(page), max_page))

    start = page * page_size
    end = min(start + page_size, total)
    chunk = LORA_GROUP_KEYS[start:end]

    rows = []
    for i in range(0, len(chunk), 2):
        row = []
        for j in range(2):
            if i + j >= len(chunk):
                break
            key = chunk[i + j]
            cfg = LORA_GROUP_CATALOG[key]
            label = str(cfg.get("label") or key)
            idx = start + (i + j)
            row.append({"text": label, "callback_data": f"{tag_prefix}g:{idx}:{job_id}"})
        rows.append(row)

    nav = []
    if page > 0:
        nav.append({"text": "‹", "callback_data": f"{tag_prefix}gp:{page-1}:{job_id}"})
    nav.append({"text": f"{page+1}/{max_page+1}", "callback_data": f"noop:{job_id}"})
    if page < max_page:
        nav.append({"text": "›", "callback_data": f"{tag_prefix}gp:{page+1}:{job_id}"})
    rows.append(nav)

    return json.dumps({"inline_keyboard": rows})


def build_lora_keyboard_ext(job_id: str, page: int = 0, *, tag_prefix: str = "e") -> str:
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
    for i in range(0, len(chunk), 2):
        row = []
        for j in range(2):
            if i + j >= len(chunk):
                break
            key = chunk[i + j]
            cfg = LORA_CATALOG[key]
            label = str(cfg.get("label") or key)
            idx = start + (i + j)
            row.append({"text": label, "callback_data": f"{tag_prefix}l:{idx}:{job_id}"})
        rows.append(row)

    nav = []
    if page > 0:
        nav.append({"text": "‹", "callback_data": f"{tag_prefix}p:{page-1}:{job_id}"})
    nav.append({"text": f"{page+1}/{max_page+1}", "callback_data": f"noop:{job_id}"})
    if page < max_page:
        nav.append({"text": "›", "callback_data": f"{tag_prefix}p:{page+1}:{job_id}"})
    rows.append(nav)

    return json.dumps({"inline_keyboard": rows})


def build_weight_keyboard_ext(job_id: str, cfg: Dict[str, Any], kind: str, *, tag_prefix: str = "e") -> str:
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
                data = f"{tag_prefix}ws:{i + j}:{job_id}"
            elif kind == "high":
                data = f"{tag_prefix}wh:{i + j}:{job_id}"
            else:
                data = f"{tag_prefix}wl:{i + j}:{job_id}"
            row.append({"text": label, "callback_data": data})
        rows.append(row)

    return json.dumps({"inline_keyboard": rows})


def build_model_keyboard_ext(job_id: str, *, tag_prefix: str = "e") -> str:
    rows = []
    for i in range(0, len(MODEL_CHOICES), 2):
        row = []
        for j in range(2):
            if i + j >= len(MODEL_CHOICES):
                break
            label, key = MODEL_CHOICES[i + j]
            row.append({"text": label, "callback_data": f"{tag_prefix}mt:{key}:{job_id}"})
        rows.append(row)

    return json.dumps({"inline_keyboard": rows})


def build_unet_keyboard_ext(job_id: str, model_type: str, page: int = 0, *, tag_prefix: str = "e") -> str:
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
            row.append({"text": label, "callback_data": f"{tag_prefix}mm:{model_type}:{idx}:{job_id}"})
        rows.append(row)

    nav = []
    if page > 0:
        nav.append({"text": "‹", "callback_data": f"{tag_prefix}pm:{model_type}:{page-1}:{job_id}"})
    nav.append({"text": f"{page+1}/{max_page+1}", "callback_data": f"noop:{job_id}"})
    if page < max_page:
        nav.append({"text": "›", "callback_data": f"{tag_prefix}pm:{model_type}:{page+1}:{job_id}"})
    rows.append(nav)

    return json.dumps({"inline_keyboard": rows})


def build_last_frame_keyboard_ext(job_id: str, *, tag_prefix: str = "e") -> str:
    rows = [
        [
            {"text": "Last frame: ON", "callback_data": f"{tag_prefix}lf:1:{job_id}"},
            {"text": "Last frame: OFF", "callback_data": f"{tag_prefix}lf:0:{job_id}"},
        ]
    ]
    return json.dumps({"inline_keyboard": rows})


def build_anchor_mode_keyboard(job_id: str, *, tag_prefix: str = "n") -> str:
    rows = [
        [
            {"text": "Stabilita: OFF", "callback_data": f"{tag_prefix}la:off:{job_id}"},
            {"text": "Anchor LF", "callback_data": f"{tag_prefix}la:anchor:{job_id}"},
        ],
        [
            {"text": "Blend LF", "callback_data": f"{tag_prefix}la:blend:{job_id}"},
        ],
    ]
    return json.dumps({"inline_keyboard": rows})


def build_resolution_keyboard_ext(job_id: str, *, tag_prefix: str = "e") -> str:
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
            row.append({"text": label, "callback_data": f"{tag_prefix}rs:{idx}:{job_id}"})
        rows.append(row)
    return json.dumps({"inline_keyboard": rows})


def build_steps_keyboard_ext(job_id: str, *, tag_prefix: str = "e") -> str:
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
            row.append({"text": label, "callback_data": f"{tag_prefix}st:{idx}:{job_id}"})
        rows.append(row)
    return json.dumps({"inline_keyboard": rows})


def build_prompt_keyboard_ext(job_id: str, *, tag_prefix: str = "e") -> str:
    rows = [
        [
            {"text": "Prompt: LoRA (default)", "callback_data": f"{tag_prefix}pu:1:{job_id}"},
            {"text": "Prompt: Default", "callback_data": f"{tag_prefix}pu:0:{job_id}"},
        ]
    ]
    if tag_prefix == "n":
        rows.append([{"text": "Len prompt (bez LoRA)", "callback_data": f"{tag_prefix}pu:2:{job_id}"}])
    return json.dumps({"inline_keyboard": rows})


def build_drift_keyboard(job_id: str, *, tag_prefix: str = "n") -> str:
    rows = []
    row = []
    for preset in DRIFT_PRESETS:
        key = str(preset.get("key") or "")
        label = str(preset.get("label") or key)
        if key == "default":
            label = f"{label} (default)"
        row.append({"text": label, "callback_data": f"{tag_prefix}d:{key}:{job_id}"})
    if row:
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


async def send_mode_picker(chat_id: int, job_id: str) -> None:
    await tg_post(
        "sendMessage",
        {
            "chat_id": chat_id,
            "text": "Vyber mod:",
            "reply_markup": build_mode_keyboard(job_id),
        },
    )


async def send_mode_picker_test5s(chat_id: int, job_id: str) -> None:
    await tg_post(
        "sendMessage",
        {
            "chat_id": chat_id,
            "text": "Album: vyber mod:",
            "reply_markup": build_mode_keyboard_test5s(job_id),
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


async def prompt_interpolation(
    *,
    chat_id: int,
    message_id: int,
    label: str,
    job_id: str,
    prompt_mode: int,
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
    reply_markup = build_interpolation_keyboard(
        job_id,
        prompt_mode=prompt_mode,
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
    text = f"{label}: interpolacia (RIFE)\nOFF = bez interpolacie"
    await edit_message_text(chat_id, message_id, text, reply_markup)


async def prompt_extended_model(
    *,
    chat_id: int,
    message_id: int,
    label: str,
    job_id: str,
    tag_prefix: str = "e",
) -> None:
    if not (chat_id and message_id):
        return
    try:
        model_type, model_idx, _model_key, _model_cfg = _get_default_model_selection()
    except Exception:
        await edit_message_text(chat_id, message_id, f"{label}: chyba modelu", json.dumps({"inline_keyboard": []}))
        return

    if tag_prefix == "n":
        sess = _wf4_session_get(job_id)
        if not sess:
            return
        sess["model_type"] = model_type
        sess["model_idx"] = model_idx
        _wf4_session_touch(sess)
        await prompt_extended_resolution(
            chat_id=chat_id,
            message_id=message_id,
            label=label,
            job_id=job_id,
            tag_prefix="n",
        )
        return

    sess = _ext_session_get(job_id)
    if not sess:
        return
    sess["model_type"] = model_type
    sess["model_idx"] = model_idx
    _ext_session_touch(sess)
    await prompt_extended_last_frame(chat_id=chat_id, message_id=message_id, label=label, job_id=job_id)


async def prompt_extended_last_frame(
    *,
    chat_id: int,
    message_id: int,
    label: str,
    job_id: str,
    tag_prefix: str = "e",
) -> None:
    if not (chat_id and message_id):
        return
    reply_markup = build_last_frame_keyboard_ext(job_id, tag_prefix=tag_prefix)
    await edit_message_text(chat_id, message_id, f"{label}: použiť last frame?", reply_markup)


async def prompt_extended_interpolation(
    *,
    chat_id: int,
    message_id: int,
    label: str,
    job_id: str,
    tag_prefix: str = "e",
    default_value: int = DEFAULT_INTERPOLATION_CLASSIC,
) -> None:
    if not (chat_id and message_id):
        return
    reply_markup = build_interpolation_keyboard_ext(
        job_id,
        tag_prefix=tag_prefix,
        default_value=default_value,
    )
    text = f"{label}: interpolacia (RIFE)\nOFF = bez interpolacie"
    await edit_message_text(chat_id, message_id, text, reply_markup)


async def prompt_extended_resolution(
    *,
    chat_id: int,
    message_id: int,
    label: str,
    job_id: str,
    tag_prefix: str = "e",
) -> None:
    if not (chat_id and message_id):
        return
    reply_markup = build_resolution_keyboard_ext(job_id, tag_prefix=tag_prefix)
    await edit_message_text(chat_id, message_id, f"{label}: vyber rozlisenie", reply_markup)


async def prompt_extended_steps(
    *,
    chat_id: int,
    message_id: int,
    label: str,
    job_id: str,
    tag_prefix: str = "e",
) -> None:
    if not (chat_id and message_id):
        return
    reply_markup = build_steps_keyboard_ext(job_id, tag_prefix=tag_prefix)
    await edit_message_text(chat_id, message_id, f"{label}: vyber pocet krokov", reply_markup)


async def prompt_extended_prompt(
    *,
    chat_id: int,
    message_id: int,
    label: str,
    job_id: str,
    tag_prefix: str = "e",
) -> None:
    if not (chat_id and message_id):
        return
    reply_markup = build_prompt_keyboard_ext(job_id, tag_prefix=tag_prefix)
    text = f"{label}: pouzit LoRA prompt?"
    if tag_prefix == "n":
        text = f"{label}: vyber prompt rezim"
    await edit_message_text(chat_id, message_id, text, reply_markup)


async def prompt_workflow4_drift(
    *,
    chat_id: int,
    message_id: int,
    label: str,
    job_id: str,
) -> None:
    if not (chat_id and message_id):
        return
    reply_markup = build_drift_keyboard(job_id, tag_prefix="n")
    text = (
        f"{label}: drift rezim\n"
        "Default = povodne, Reduced = shift 2 / denoise 0.7 / overlap 2"
    )
    await edit_message_text(chat_id, message_id, text, reply_markup)


async def prompt_workflow4_interpolation(
    *,
    chat_id: int,
    message_id: int,
    label: str,
    job_id: str,
) -> None:
    if not (chat_id and message_id):
        return
    reply_markup = build_interpolation_keyboard_ext(
        job_id,
        tag_prefix="n",
        default_value=DEFAULT_INTERPOLATION_NEW,
    )
    text = f"{label}: interpolacia (RIFE)\nOFF = bez interpolacie"
    await edit_message_text(chat_id, message_id, text, reply_markup)


async def prompt_workflow4_anchor_mode(
    *,
    chat_id: int,
    message_id: int,
    label: str,
    job_id: str,
) -> None:
    if not (chat_id and message_id):
        return
    reply_markup = build_anchor_mode_keyboard(job_id, tag_prefix="n")
    text = (
        f"{label}: stabilita (last frame)\n"
        "Anchor = nahradí prev_samples, Blend = ponechá prev_samples"
    )
    await edit_message_text(chat_id, message_id, text, reply_markup)


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


def _extended_combo_label(sess: Dict[str, Any]) -> str:
    l1 = (sess.get("lora1") or {}).get("label")
    l2 = (sess.get("lora2") or {}).get("label")
    if l1 and l2:
        return f"{l1} + {l2}"
    return l1 or l2 or "Extended10s"


def _workflow4_combo_label(sess: Dict[str, Any]) -> str:
    labels = []
    for i in range(1, WORKFLOW4_PARTS + 1):
        label = (sess.get(f"lora{i}") or {}).get("label")
        if label:
            labels.append(label)
    if len(labels) == WORKFLOW4_PARTS:
        return " + ".join(labels)
    if labels:
        return " + ".join(labels)
    return "Workflow4"


def _workflow4_prompt_for(cfg: Dict[str, Any], lora: Dict[str, Any], use_prompt: bool) -> str:
    if not use_prompt:
        return DEFAULT_PROMPT
    override = lora.get("positive")
    if isinstance(override, str) and override.strip():
        return override
    candidate = cfg.get("positive")
    return candidate if isinstance(candidate, str) and candidate.strip() else DEFAULT_PROMPT


async def _workflow4_prompt_current_weight(
    *,
    chat_id: int,
    message_id: int,
    job_id: str,
    sess: Dict[str, Any],
) -> bool:
    if not (chat_id and message_id):
        return False
    current = int(sess.get("current_lora") or 1)
    lora = sess.get(f"lora{current}") or {}
    cfg = lora.get("cfg")
    if not isinstance(cfg, dict):
        return False
    lora_type = str(lora.get("type") or "single").lower()
    label = str(lora.get("label") or "")
    if lora_type == "pair":
        reply_markup = build_weight_keyboard_ext(job_id, cfg, "high", tag_prefix="n")
        await edit_message_text(chat_id, message_id, f"{label}: vyber HIGH vahu", reply_markup)
    else:
        reply_markup = build_weight_keyboard_ext(job_id, cfg, "single", tag_prefix="n")
        await edit_message_text(chat_id, message_id, f"{label}: vyber vahu", reply_markup)
    return True


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
    rife_multiplier: Optional[int],
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
    if rife_multiplier is not None:
        payload["rife_multiplier"] = int(rife_multiplier)
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
    rife_multiplier: Optional[int],
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
    if rife_multiplier is not None:
        payload["rife_multiplier"] = int(rife_multiplier)
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


async def _submit_batch_job(
    *,
    job_id: str,
    chat_id: int,
    message_id: int,
    weight: float,
) -> None:
    row = set_queue(job_id, "batch5s")

    batch_loras: List[Dict[str, Any]] = []
    for lora_key in LORA_KEYS:
        cfg = LORA_CATALOG.get(lora_key)
        if not isinstance(cfg, dict):
            continue
        lora_type = str(cfg.get("type") or "single").lower()
        label = str(cfg.get("label") or lora_key)
        positive = cfg.get("positive")
        if lora_type == "pair":
            high_filename = cfg.get("high_filename")
            low_filename = cfg.get("low_filename")
            if not high_filename or not low_filename:
                continue
            batch_loras.append(
                {
                    "key": lora_key,
                    "label": label,
                    "type": "pair",
                    "high_filename": high_filename,
                    "low_filename": low_filename,
                    "positive": positive,
                }
            )
        else:
            filename = cfg.get("filename")
            if not filename:
                continue
            batch_loras.append(
                {
                    "key": lora_key,
                    "label": label,
                    "type": "single",
                    "filename": filename,
                    "positive": positive,
                }
            )

    if not batch_loras:
        raise RuntimeError("Batch5s: no valid LoRA entries")

    model_cfg = MODEL_CATALOG.get(BATCH_TEST_MODEL_KEY)
    if not isinstance(model_cfg, dict):
        raise RuntimeError(f"Batch5s: model not found: {BATCH_TEST_MODEL_KEY}")
    model_type = _normalize_model_type(model_cfg.get("type"))
    if model_type not in ("wan", "gguf"):
        raise RuntimeError(f"Batch5s: invalid model type: {model_cfg.get('type')}")
    if not model_cfg.get("high_filename") or not model_cfg.get("low_filename"):
        raise RuntimeError(f"Batch5s: model missing high/low filenames: {BATCH_TEST_MODEL_KEY}")

    weights = [float(weight)]

    payload: Dict[str, Any] = {
        "mode": "batch5s",
        "job_id": job_id,
        "chat_id": int(row["chat_id"]),
        "input_file_id": row["input_file_id"],
        "batch_loras": batch_loras,
        "batch_weights": weights,
        "batch_default_prompt": DEFAULT_PROMPT,
        "batch_use_prompt": bool(BATCH_TEST_USE_PROMPT),
        "use_gguf": (model_type == "gguf"),
        "use_last_frame": bool(BATCH_TEST_USE_LAST_FRAME),
        "video_width": BATCH_TEST_VIDEO_WIDTH,
        "video_height": BATCH_TEST_VIDEO_HEIGHT,
        "total_steps": BATCH_TEST_STEPS,
        "rife_multiplier": BATCH_TEST_RIFE,
        "model_label": str(model_cfg.get("label") or BATCH_TEST_MODEL_KEY),
        "model_high_filename": model_cfg.get("high_filename"),
        "model_low_filename": model_cfg.get("low_filename"),
    }

    runpod_id = await submit_runpod(payload)
    if runpod_id:
        set_runpod_request_id(job_id, runpod_id)

    total = len(batch_loras) * len(weights)
    weight_label = _format_weight(weights[0])
    await edit_placeholder(
        int(row["chat_id"]),
        int(row["placeholder_message_id"]),
        f"Batch 5s: {len(batch_loras)} LoRA (w={weight_label}) ({total} renderov)…",
    )
    if chat_id and message_id:
        await edit_message_text(chat_id, message_id, "Batch 5s: spustené", json.dumps({"inline_keyboard": []}))


async def _submit_extended_job(
    *,
    job_id: str,
    lora1_key: str,
    lora1_cfg: Dict[str, Any],
    lora1_weights: Dict[str, float],
    lora2_key: str,
    lora2_cfg: Dict[str, Any],
    lora2_weights: Dict[str, float],
    use_gguf: bool,
    use_last_frame: bool,
    video_width: Optional[int],
    video_height: Optional[int],
    total_steps: Optional[int],
    positive_prompt_1: Optional[str],
    positive_prompt_2: Optional[str],
    rife_multiplier: Optional[int],
    model_label: Optional[str],
    model_high_filename: Optional[str],
    model_low_filename: Optional[str],
    chat_id: int,
    message_id: int,
) -> None:
    combined_key = f"{lora1_key}+{lora2_key}"
    row = set_queue(job_id, combined_key)

    label1 = str(lora1_cfg.get("label") or lora1_key)
    label2 = str(lora2_cfg.get("label") or lora2_key)
    lora1_type = str(lora1_cfg.get("type") or "single")
    lora2_type = str(lora2_cfg.get("type") or "single")

    payload: Dict[str, Any] = {
        "mode": "extended10s",
        "job_id": job_id,
        "chat_id": int(row["chat_id"]),
        "input_file_id": row["input_file_id"],
        "lora_key": lora1_key,
        "lora_label": label1,
        "lora_type": lora1_type,
        "lora2_key": lora2_key,
        "lora2_label": label2,
        "lora2_type": lora2_type,
        "positive_prompt": positive_prompt_1,
        "positive_prompt_2": positive_prompt_2 or positive_prompt_1,
        "use_gguf": use_gguf,
        "use_last_frame": use_last_frame,
        "video_width": video_width,
        "video_height": video_height,
        "total_steps": total_steps,
    }
    if rife_multiplier is not None:
        payload["rife_multiplier"] = int(rife_multiplier)

    if lora1_type.lower() == "pair":
        payload["lora_high_filename"] = lora1_cfg.get("high_filename")
        payload["lora_high_strength"] = lora1_weights.get("high_weight")
        payload["lora_low_filename"] = lora1_cfg.get("low_filename")
        payload["lora_low_strength"] = lora1_weights.get("low_weight")
    else:
        payload["lora_filename"] = lora1_cfg.get("filename")
        payload["lora_strength"] = lora1_weights.get("weight")

    if lora2_type.lower() == "pair":
        payload["lora2_high_filename"] = lora2_cfg.get("high_filename")
        payload["lora2_high_strength"] = lora2_weights.get("high_weight")
        payload["lora2_low_filename"] = lora2_cfg.get("low_filename")
        payload["lora2_low_strength"] = lora2_weights.get("low_weight")
    else:
        payload["lora2_filename"] = lora2_cfg.get("filename")
        payload["lora2_strength"] = lora2_weights.get("weight")

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
        f"Renderujem ({label1} + {label2}, {model_type_label}{suffix})…",
    )
    if chat_id and message_id:
        await edit_keyboard(chat_id, message_id, json.dumps({"inline_keyboard": []}))


async def _submit_workflow4_job(
    *,
    job_id: str,
    lora1_key: str,
    lora1_cfg: Dict[str, Any],
    lora1_weights: Dict[str, float],
    lora2_key: str,
    lora2_cfg: Dict[str, Any],
    lora2_weights: Dict[str, float],
    lora3_key: str,
    lora3_cfg: Dict[str, Any],
    lora3_weights: Dict[str, float],
    lora4_key: str,
    lora4_cfg: Dict[str, Any],
    lora4_weights: Dict[str, float],
    use_lora: bool,
    use_gguf: bool,
    use_last_frame: bool,
    video_width: Optional[int],
    video_height: Optional[int],
    total_steps: Optional[int],
    drift_speed_shift: Optional[float],
    drift_denoise: Optional[float],
    drift_overlap: Optional[int],
    rife_multiplier: Optional[int],
    anchor_mode: str,
    positive_prompt_1: Optional[str],
    positive_prompt_2: Optional[str],
    positive_prompt_3: Optional[str],
    positive_prompt_4: Optional[str],
    model_label: Optional[str],
    model_high_filename: Optional[str],
    model_low_filename: Optional[str],
    chat_id: int,
    message_id: int,
) -> None:
    combined_key = f"{lora1_key}+{lora2_key}+{lora3_key}+{lora4_key}"
    row = set_queue(job_id, combined_key)

    label1 = str(lora1_cfg.get("label") or lora1_key)
    label2 = str(lora2_cfg.get("label") or lora2_key)
    label3 = str(lora3_cfg.get("label") or lora3_key)
    label4 = str(lora4_cfg.get("label") or lora4_key)
    lora1_type = str(lora1_cfg.get("type") or "single")
    lora2_type = str(lora2_cfg.get("type") or "single")
    lora3_type = str(lora3_cfg.get("type") or "single")
    lora4_type = str(lora4_cfg.get("type") or "single")

    payload: Dict[str, Any] = {
        "mode": "workflow4",
        "workflow_key": "new",
        "job_id": job_id,
        "chat_id": int(row["chat_id"]),
        "input_file_id": row["input_file_id"],
        "lora_key": lora1_key,
        "lora_label": label1,
        "lora_type": lora1_type,
        "lora2_key": lora2_key,
        "lora2_label": label2,
        "lora2_type": lora2_type,
        "lora3_key": lora3_key,
        "lora3_label": label3,
        "lora3_type": lora3_type,
        "lora4_key": lora4_key,
        "lora4_label": label4,
        "lora4_type": lora4_type,
        "positive_prompt": positive_prompt_1,
        "positive_prompt_2": positive_prompt_2 or positive_prompt_1,
        "positive_prompt_3": positive_prompt_3 or positive_prompt_1,
        "positive_prompt_4": positive_prompt_4 or positive_prompt_1,
        "use_lora": use_lora,
        "use_gguf": use_gguf,
        "use_last_frame": use_last_frame,
        "anchor_mode": anchor_mode,
        "video_width": video_width,
        "video_height": video_height,
        "total_steps": total_steps,
    }
    if drift_speed_shift is not None:
        payload["drift_speed_shift"] = float(drift_speed_shift)
    if drift_denoise is not None:
        payload["drift_denoise"] = float(drift_denoise)
    if drift_overlap is not None:
        payload["drift_overlap"] = int(drift_overlap)
    if rife_multiplier is not None:
        payload["rife_multiplier"] = int(rife_multiplier)

    if lora1_type.lower() == "pair":
        payload["lora_high_filename"] = lora1_cfg.get("high_filename")
        payload["lora_high_strength"] = lora1_weights.get("high_weight")
        payload["lora_low_filename"] = lora1_cfg.get("low_filename")
        payload["lora_low_strength"] = lora1_weights.get("low_weight")
    else:
        payload["lora_filename"] = lora1_cfg.get("filename")
        payload["lora_strength"] = lora1_weights.get("weight")

    if lora2_type.lower() == "pair":
        payload["lora2_high_filename"] = lora2_cfg.get("high_filename")
        payload["lora2_high_strength"] = lora2_weights.get("high_weight")
        payload["lora2_low_filename"] = lora2_cfg.get("low_filename")
        payload["lora2_low_strength"] = lora2_weights.get("low_weight")
    else:
        payload["lora2_filename"] = lora2_cfg.get("filename")
        payload["lora2_strength"] = lora2_weights.get("weight")

    if lora3_type.lower() == "pair":
        payload["lora3_high_filename"] = lora3_cfg.get("high_filename")
        payload["lora3_high_strength"] = lora3_weights.get("high_weight")
        payload["lora3_low_filename"] = lora3_cfg.get("low_filename")
        payload["lora3_low_strength"] = lora3_weights.get("low_weight")
    else:
        payload["lora3_filename"] = lora3_cfg.get("filename")
        payload["lora3_strength"] = lora3_weights.get("weight")

    if lora4_type.lower() == "pair":
        payload["lora4_high_filename"] = lora4_cfg.get("high_filename")
        payload["lora4_high_strength"] = lora4_weights.get("high_weight")
        payload["lora4_low_filename"] = lora4_cfg.get("low_filename")
        payload["lora4_low_strength"] = lora4_weights.get("low_weight")
    else:
        payload["lora4_filename"] = lora4_cfg.get("filename")
        payload["lora4_strength"] = lora4_weights.get("weight")

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
    combo_label = f"{label1} + {label2} + {label3} + {label4}"
    if not use_lora:
        combo_label = f"{combo_label} (prompt-only)"
    await edit_placeholder(
        int(row["chat_id"]),
        int(row["placeholder_message_id"]),
        f"Renderujem ({combo_label}, {model_type_label}{suffix})…",
    )
    if chat_id and message_id:
        await edit_keyboard(chat_id, message_id, json.dumps({"inline_keyboard": []}))


async def _workflow4_submit_from_session(
    *,
    job_id: str,
    chat_id: int,
    message_id: int,
    sess: Dict[str, Any],
    drift_preset: Optional[Dict[str, Any]],
) -> bool:
    lora1 = sess.get("lora1") or {}
    lora2 = sess.get("lora2") or {}
    lora3 = sess.get("lora3") or {}
    lora4 = sess.get("lora4") or {}
    if not lora1 or not lora2 or not lora3 or not lora4:
        return False
    cfg1 = lora1.get("cfg")
    cfg2 = lora2.get("cfg")
    cfg3 = lora3.get("cfg")
    cfg4 = lora4.get("cfg")
    if not isinstance(cfg1, dict) or not isinstance(cfg2, dict) or not isinstance(cfg3, dict) or not isinstance(cfg4, dict):
        return False
    if "use_prompt" not in sess or "use_lora" not in sess:
        return False

    if sess.get("resolution_idx") is None:
        resolution_idx = DEFAULT_RESOLUTION_IDX
    else:
        resolution_idx = int(sess.get("resolution_idx"))
    default_steps_idx = STEPS_OPTIONS.index(DEFAULT_STEPS) if DEFAULT_STEPS in STEPS_OPTIONS else 0
    if sess.get("steps_idx") is None:
        steps_idx = default_steps_idx
    else:
        steps_idx = int(sess.get("steps_idx"))
    resolution = _resolution_by_index(resolution_idx)
    steps = _steps_by_index(steps_idx)
    if not resolution or steps is None:
        return False

    model_type = str(sess.get("model_type") or "")
    model_idx = int(sess.get("model_idx") or -1)
    if model_type not in ("wan", "gguf"):
        return False

    model_label = None
    model_high_filename = None
    model_low_filename = None
    if model_idx >= 0:
        model_key, model_cfg = _get_model_cfg_by_index(model_type, model_idx)
        if not model_cfg:
            return False
        model_label = str(model_cfg.get("label") or model_key)
        model_high_filename = model_cfg.get("high_filename")
        model_low_filename = model_cfg.get("low_filename")

    use_prompt = bool(sess.get("use_prompt"))
    use_lora = bool(sess.get("use_lora"))
    prompt_text_1 = _workflow4_prompt_for(cfg1, lora1, use_prompt)
    prompt_text_2 = _workflow4_prompt_for(cfg2, lora2, use_prompt)
    prompt_text_3 = _workflow4_prompt_for(cfg3, lora3, use_prompt)
    prompt_text_4 = _workflow4_prompt_for(cfg4, lora4, use_prompt)

    weights1 = {
        "weight": lora1.get("weight"),
        "high_weight": lora1.get("high_weight"),
        "low_weight": lora1.get("low_weight"),
    }
    weights2 = {
        "weight": lora2.get("weight"),
        "high_weight": lora2.get("high_weight"),
        "low_weight": lora2.get("low_weight"),
    }
    weights3 = {
        "weight": lora3.get("weight"),
        "high_weight": lora3.get("high_weight"),
        "low_weight": lora3.get("low_weight"),
    }
    weights4 = {
        "weight": lora4.get("weight"),
        "high_weight": lora4.get("high_weight"),
        "low_weight": lora4.get("low_weight"),
    }

    drift_speed_shift = None if not drift_preset else drift_preset.get("speed_shift")
    drift_denoise = None if not drift_preset else drift_preset.get("denoise")
    drift_overlap = None if not drift_preset else drift_preset.get("overlap")
    rife_multiplier = _normalize_interpolation_value(
        sess.get("rife_multiplier"),
        DEFAULT_INTERPOLATION_NEW,
    )
    anchor_mode = str(sess.get("anchor_mode") or "off").strip().lower()
    if anchor_mode not in ("off", "anchor", "blend"):
        anchor_mode = "off"

    await _submit_workflow4_job(
        job_id=job_id,
        lora1_key=str(lora1.get("key")),
        lora1_cfg=cfg1,
        lora1_weights=weights1,
        lora2_key=str(lora2.get("key")),
        lora2_cfg=cfg2,
        lora2_weights=weights2,
        lora3_key=str(lora3.get("key")),
        lora3_cfg=cfg3,
        lora3_weights=weights3,
        lora4_key=str(lora4.get("key")),
        lora4_cfg=cfg4,
        lora4_weights=weights4,
        use_lora=use_lora,
        use_gguf=(model_type == "gguf"),
        use_last_frame=bool(sess.get("use_last_frame")),
        video_width=int(resolution["width"]),
        video_height=int(resolution["height"]),
        total_steps=int(steps),
        drift_speed_shift=drift_speed_shift,
        drift_denoise=drift_denoise,
        drift_overlap=drift_overlap,
        rife_multiplier=rife_multiplier,
        anchor_mode=anchor_mode,
        positive_prompt_1=prompt_text_1,
        positive_prompt_2=prompt_text_2,
        positive_prompt_3=prompt_text_3,
        positive_prompt_4=prompt_text_4,
        model_label=model_label,
        model_high_filename=model_high_filename,
        model_low_filename=model_low_filename,
        chat_id=chat_id,
        message_id=message_id,
    )
    _wf4_session_clear(job_id)
    return True


# ---------------- Update processing ----------------

def rate_limit(chat_id: int):
    return


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


def _test5s_album_key(chat_id: int, media_group_id: str) -> str:
    return f"{chat_id}:{media_group_id}"


async def _finalize_test5s_album(group_key: str, sess: Dict[str, Any]) -> None:
    test5s_album_sessions.pop(group_key, None)
    job_ids = sess.get("job_ids") or []
    if not job_ids:
        return
    chat_id = int(sess.get("chat_id") or 0)
    placeholder_ids = sess.get("placeholder_ids") or []
    for placeholder_id in placeholder_ids:
        await edit_placeholder(chat_id, int(placeholder_id), "Album: vyber mod…")
    leader_job_id = job_ids[0]
    _test5s_group_set(leader_job_id, job_ids)
    if chat_id:
        try:
            await send_mode_picker_test5s(chat_id, leader_job_id)
        except Exception as exc:
            for placeholder_id in placeholder_ids:
                await edit_placeholder(
                    chat_id,
                    int(placeholder_id),
                    f"Chyba: neviem zobraziť výber módu ({exc})",
                )
            raise


def _flush_test5s_group_jobs() -> None:
    now = time.time()
    for job_id, entry in list(test5s_group_jobs.items()):
        created_at = float(entry.get("created_at") or 0.0)
        if now - created_at >= TEST5S_GROUP_TTL_SECONDS:
            test5s_group_jobs.pop(job_id, None)


async def _flush_test5s_album_sessions() -> None:
    now = time.time()
    for group_key, sess in list(test5s_album_sessions.items()):
        updated_at = float(sess.get("updated_at") or 0.0)
        if now - updated_at >= TEST5S_MEDIA_GROUP_TTL_SECONDS:
            await _finalize_test5s_album(group_key, sess)


async def _handle_test5s_album_photo(chat_id: int, file_id: str, media_group_id: str) -> str:
    group_key = _test5s_album_key(chat_id, media_group_id)
    sess = test5s_album_sessions.get(group_key)
    if not sess:
        sess = {
            "chat_id": chat_id,
            "media_group_id": media_group_id,
            "job_ids": [],
            "placeholder_ids": [],
            "updated_at": time.time(),
        }
        test5s_album_sessions[group_key] = sess

    placeholder_id = await send_placeholder(chat_id)
    job_id = uuid.uuid4().hex  # shorter than str(uuid.uuid4())

    try:
        save_job_awaiting_lora(job_id, chat_id, placeholder_id, file_id)
    except Exception as exc:
        await edit_placeholder(chat_id, placeholder_id, f"Chyba DB: {exc}")
        raise

    await edit_placeholder(chat_id, placeholder_id, "Album: čakám na ďalšie fotky…")

    sess["job_ids"].append(job_id)
    sess["placeholder_ids"].append(placeholder_id)
    sess["updated_at"] = time.time()

    if len(sess["job_ids"]) >= max(1, TEST5S_MEDIA_GROUP_SIZE):
        await _finalize_test5s_album(group_key, sess)

    return job_id


def extract_photo(update: Dict[str, Any]) -> Tuple[int, str, Optional[str]]:
    msg = update.get("message") or update.get("edited_message")
    if not msg:
        raise RuntimeError("no_message")

    chat_id = int(msg["chat"]["id"])
    media_group_id = msg.get("media_group_id")
    photos = msg.get("photo") or []
    if not photos:
        raise RuntimeError("no_photo")

    largest = photos[-1]
    file_id = largest["file_id"]
    file_size = int(largest.get("file_size") or 0)
    if file_size and file_size > MAX_IMAGE_BYTES:
        raise RuntimeError("image_too_large")

    return chat_id, file_id, media_group_id


async def process_callback(update: Dict[str, Any]) -> None:
    cq = update["callback_query"]
    await answer_callback(cq["id"])

    data = cq.get("data") or ""
    msg = cq.get("message") or {}
    chat = msg.get("chat") or {}
    chat_id = int(chat.get("id") or 0)
    message_id = int(msg.get("message_id") or 0)

    # Supported:
    # - mode:<std|ext|new|batch|test5s>:<job_id> (select mode)
    # - bw:<idx>:<job_id>                      (batch weight)
    # - el:<idx>:<job_id>                      (select LoRA, extended)
    # - ep:<page>:<job_id>                     (LoRA page, extended)
    # - ews:<w_idx>:<job_id>                   (single weight, extended)
    # - ewh:<h_idx>:<job_id>                   (pair high, extended)
    # - ewl:<l_idx>:<job_id>                   (pair low, extended)
    # - emt:<model>:<job_id>                   (select model type, extended)
    # - epm:<model>:<page>:<job_id>            (model page, extended)
    # - emm:<model>:<m_idx>:<job_id>           (select model, extended)
    # - elf:<on>:<job_id>                      (last frame, extended)
    # - ers:<r_idx>:<job_id>                   (resolution, extended)
    # - est:<s_idx>:<job_id>                   (steps, extended)
    # - epu:<p>:<job_id>                       (prompt, extended)
    # - ng:<idx>:<job_id>                      (select LoRA group, workflow4)
    # - ngp:<page>:<job_id>                    (group page, workflow4)
    # - nl:<idx>:<job_id>                      (select LoRA, workflow4)
    # - np:<page>:<job_id>                     (LoRA page, workflow4)
    # - nws:<w_idx>:<job_id>                   (single weight, workflow4)
    # - nwh:<h_idx>:<job_id>                   (pair high, workflow4)
    # - nwl:<l_idx>:<job_id>                   (pair low, workflow4)
    # - nmt:<model>:<job_id>                   (select model type, workflow4)
    # - npm:<model>:<page>:<job_id>            (model page, workflow4)
    # - nmm:<model>:<m_idx>:<job_id>           (select model, workflow4)
    # - nlf:<on>:<job_id>                      (last frame, workflow4)
    # - nla:<mode>:<job_id>                    (last-frame anchor mode, workflow4)
    # - nrs:<r_idx>:<job_id>                   (resolution, workflow4)
    # - nst:<s_idx>:<job_id>                   (steps, workflow4)
    # - npu:<p>:<job_id>                       (prompt / prompt-only, workflow4)
    # - nd:<mode>:<job_id>                     (drift mode, workflow4)
    # - nri:<interp>:<job_id>                  (interpolation, workflow4)
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
    # - ri:<interp>:<p>:<s_idx>:<r_idx>:<on>:<lora_idx>:<w_idx>:<model>:<m_idx>:<job_id> (interpolation, single)
    # - ri:<interp>:<p>:<s_idx>:<r_idx>:<on>:<lora_idx>:<h_idx>:<l_idx>:<model>:<m_idx>:<job_id> (interpolation, pair)
    # - eri:<interp>:<job_id>                  (interpolation, extended)
    # - t5l:<idx>:<job_id>                     (select LoRA, test5s)
    # - t5p:<page>:<job_id>                    (LoRA page, test5s)
    # - noop:<job_id>                          (do nothing)
    parts = data.split(":")
    if len(parts) < 2:
        return

    tag = parts[0]

    if tag == "noop":
        return

    if tag == "mode":
        if len(parts) != 3:
            return
        choice = parts[1]
        job_id = parts[2]
        if not (chat_id and message_id):
            return
        if choice == "std":
            _ext_session_clear(job_id)
            _wf4_session_clear(job_id)
            await edit_message_text(chat_id, message_id, "Vyber štýl (LoRA):", build_lora_keyboard(job_id, page=0))
            return
        if choice == "test5s":
            _ext_session_clear(job_id)
            _wf4_session_clear(job_id)
            await edit_message_text(
                chat_id,
                message_id,
                "Test5s: vyber štýl (LoRA):",
                build_lora_keyboard_ext(job_id, page=0, tag_prefix="t5"),
            )
            return
        if choice == "ext":
            _wf4_session_clear(job_id)
            extended_sessions[job_id] = {"mode": "extended10s", "current_lora": 1, "rife_multiplier": None}
            _ext_session_touch(extended_sessions[job_id])
            await edit_message_text(
                chat_id,
                message_id,
                "Extended10s: vyber 1. LoRA",
                build_lora_keyboard_ext(job_id, page=0),
            )
            return
        if choice == "batch":
            _ext_session_clear(job_id)
            _wf4_session_clear(job_id)
            if not BATCH_TEST_WEIGHTS:
                await edit_message_text(
                    chat_id,
                    message_id,
                    "Batch 5s: chyba (ziadne váhy)",
                    json.dumps({"inline_keyboard": []}),
                )
                return
            await edit_message_text(
                chat_id,
                message_id,
                "Batch 5s: vyber váhu",
                build_batch_weight_keyboard(job_id),
            )
            return
        if choice == "new":
            _ext_session_clear(job_id)
            workflow4_sessions[job_id] = {
                "mode": "workflow4",
                "current_lora": 1,
                "drift_key": "default",
                "rife_multiplier": None,
                "anchor_mode": "off",
                "use_prompt": True,
                "use_lora": True,
                "use_last_frame": False,
            }
            _wf4_session_touch(workflow4_sessions[job_id])
            if LORA_GROUP_KEYS:
                await edit_message_text(
                    chat_id,
                    message_id,
                    "Workflow4: vyber skupinu",
                    build_lora_group_keyboard(job_id, page=0, tag_prefix="n"),
                )
            else:
                await edit_message_text(
                    chat_id,
                    message_id,
                    "Workflow4: vyber 1. LoRA",
                    build_lora_keyboard_ext(job_id, page=0, tag_prefix="n"),
                )
            return
        return

    if tag == "bw":
        if len(parts) != 3:
            return
        try:
            idx = int(parts[1])
            job_id = parts[2]
        except Exception:
            return
        if idx < 0 or idx >= len(BATCH_TEST_WEIGHTS):
            return
        weight = BATCH_TEST_WEIGHTS[idx]
        try:
            await _submit_batch_job(job_id=job_id, chat_id=chat_id, message_id=message_id, weight=weight)
        except Exception as exc:
            await edit_message_text(
                chat_id,
                message_id,
                f"Batch 5s: chyba ({exc})",
                json.dumps({"inline_keyboard": []}),
            )
            raise
        return

    if tag == "np":
        if len(parts) != 3:
            return
        try:
            page = int(parts[1])
            job_id = parts[2]
        except Exception:
            return
        if not _wf4_session_get(job_id):
            return
        if chat_id and message_id:
            await edit_keyboard(chat_id, message_id, build_lora_keyboard_ext(job_id, page=page, tag_prefix="n"))
        return

    if tag == "ngp":
        if len(parts) != 3:
            return
        try:
            page = int(parts[1])
            job_id = parts[2]
        except Exception:
            return
        if not _wf4_session_get(job_id):
            return
        if chat_id and message_id:
            await edit_keyboard(chat_id, message_id, build_lora_group_keyboard(job_id, page=page, tag_prefix="n"))
        return

    if tag == "ng":
        if len(parts) != 3:
            return
        try:
            idx = int(parts[1])
            job_id = parts[2]
        except Exception:
            return
        sess = _wf4_session_get(job_id)
        if not sess:
            return
        group_key, group_cfg = _get_lora_group_by_index(idx)
        if not group_cfg:
            return
        parts_cfg = group_cfg.get("parts")
        if not isinstance(parts_cfg, list) or len(parts_cfg) != WORKFLOW4_PARTS:
            return
        sess["group_key"] = group_key
        sess["group_label"] = str(group_cfg.get("label") or group_key)
        sess["current_lora"] = 1
        for i in range(1, WORKFLOW4_PARTS + 1):
            part = parts_cfg[i - 1]
            if not isinstance(part, dict):
                return
            lora_key = str(part.get("lora_key") or "").strip()
            if not lora_key:
                return
            lora_cfg = LORA_CATALOG.get(lora_key)
            if not isinstance(lora_cfg, dict):
                return
            lora_type = str(lora_cfg.get("type") or "single").lower()
            lora_entry: Dict[str, Any] = {
                "key": lora_key,
                "label": str(lora_cfg.get("label") or lora_key),
                "cfg": lora_cfg,
                "type": lora_type,
            }
            positive_override = part.get("positive")
            if isinstance(positive_override, str) and positive_override.strip():
                lora_entry["positive"] = positive_override
            sess[f"lora{i}"] = lora_entry
        _wf4_session_touch(sess)
        if chat_id and message_id:
            await _workflow4_prompt_current_weight(
                chat_id=chat_id,
                message_id=message_id,
                job_id=job_id,
                sess=sess,
            )
        return

    if tag == "nl":
        if len(parts) != 3:
            return
        try:
            idx = int(parts[1])
            job_id = parts[2]
        except Exception:
            return
        sess = _wf4_session_get(job_id)
        if not sess:
            return
        lora_key, cfg = _get_lora_cfg_by_index(idx)
        if not cfg:
            return
        lora_type = str(cfg.get("type") or "single").lower()
        label = str(cfg.get("label") or lora_key)
        current = int(sess.get("current_lora") or 1)
        sess[f"lora{current}"] = {"key": lora_key, "label": label, "cfg": cfg, "type": lora_type}
        _wf4_session_touch(sess)

        if not (chat_id and message_id):
            return
        if lora_type == "pair":
            reply_markup = build_weight_keyboard_ext(job_id, cfg, "high", tag_prefix="n")
            await edit_message_text(chat_id, message_id, f"{label}: vyber HIGH vahu", reply_markup)
        else:
            reply_markup = build_weight_keyboard_ext(job_id, cfg, "single", tag_prefix="n")
            await edit_message_text(chat_id, message_id, f"{label}: vyber vahu", reply_markup)
        return

    if tag == "nws":
        if len(parts) != 3:
            return
        try:
            weight_idx = int(parts[1])
            job_id = parts[2]
        except Exception:
            return
        sess = _wf4_session_get(job_id)
        if not sess:
            return
        current = int(sess.get("current_lora") or 1)
        lora = sess.get(f"lora{current}") or {}
        cfg = lora.get("cfg")
        if not isinstance(cfg, dict):
            return
        if str(lora.get("type") or "single").lower() == "pair":
            return
        options, _ = _weight_options_for(cfg, "single")
        if weight_idx < 0 or weight_idx >= len(options):
            return
        lora["weight"] = options[weight_idx]
        sess[f"lora{current}"] = lora
        _wf4_session_touch(sess)

        if not (chat_id and message_id):
            return
        if current < WORKFLOW4_PARTS:
            sess["current_lora"] = current + 1
            _wf4_session_touch(sess)
            if sess.get("group_key"):
                if await _workflow4_prompt_current_weight(
                    chat_id=chat_id,
                    message_id=message_id,
                    job_id=job_id,
                    sess=sess,
                ):
                    return
            await edit_message_text(
                chat_id,
                message_id,
                f"Workflow4: vyber {current + 1}. LoRA",
                build_lora_keyboard_ext(job_id, page=0, tag_prefix="n"),
            )
            return

        label = _workflow4_combo_label(sess)
        await prompt_extended_model(chat_id=chat_id, message_id=message_id, label=label, job_id=job_id, tag_prefix="n")
        return

    if tag == "nwh":
        if len(parts) != 3:
            return
        try:
            weight_idx = int(parts[1])
            job_id = parts[2]
        except Exception:
            return
        sess = _wf4_session_get(job_id)
        if not sess:
            return
        current = int(sess.get("current_lora") or 1)
        lora = sess.get(f"lora{current}") or {}
        cfg = lora.get("cfg")
        if not isinstance(cfg, dict):
            return
        if str(lora.get("type") or "single").lower() != "pair":
            return
        options, _ = _weight_options_for(cfg, "high")
        if weight_idx < 0 or weight_idx >= len(options):
            return
        lora["high_weight"] = options[weight_idx]
        sess[f"lora{current}"] = lora
        _wf4_session_touch(sess)

        if chat_id and message_id:
            reply_markup = build_weight_keyboard_ext(job_id, cfg, "low", tag_prefix="n")
            await edit_message_text(chat_id, message_id, f"{lora.get('label')}: vyber LOW vahu", reply_markup)
        return

    if tag == "nwl":
        if len(parts) != 3:
            return
        try:
            weight_idx = int(parts[1])
            job_id = parts[2]
        except Exception:
            return
        sess = _wf4_session_get(job_id)
        if not sess:
            return
        current = int(sess.get("current_lora") or 1)
        lora = sess.get(f"lora{current}") or {}
        cfg = lora.get("cfg")
        if not isinstance(cfg, dict):
            return
        if str(lora.get("type") or "single").lower() != "pair":
            return
        options, _ = _weight_options_for(cfg, "low")
        if weight_idx < 0 or weight_idx >= len(options):
            return
        lora["low_weight"] = options[weight_idx]
        sess[f"lora{current}"] = lora
        _wf4_session_touch(sess)

        if not (chat_id and message_id):
            return
        if current < WORKFLOW4_PARTS:
            sess["current_lora"] = current + 1
            _wf4_session_touch(sess)
            if sess.get("group_key"):
                if await _workflow4_prompt_current_weight(
                    chat_id=chat_id,
                    message_id=message_id,
                    job_id=job_id,
                    sess=sess,
                ):
                    return
            await edit_message_text(
                chat_id,
                message_id,
                f"Workflow4: vyber {current + 1}. LoRA",
                build_lora_keyboard_ext(job_id, page=0, tag_prefix="n"),
            )
            return

        label = _workflow4_combo_label(sess)
        await prompt_extended_model(chat_id=chat_id, message_id=message_id, label=label, job_id=job_id, tag_prefix="n")
        return

    if tag == "nmt":
        if len(parts) != 3:
            return
        job_id = parts[2]
        sess = _wf4_session_get(job_id)
        if not sess:
            return
        try:
            model_type, model_idx, _model_key, _model_cfg = _get_default_model_selection()
        except Exception:
            if chat_id and message_id:
                label = _workflow4_combo_label(sess)
                await edit_message_text(chat_id, message_id, f"{label}: chyba modelu", json.dumps({"inline_keyboard": []}))
            return

        sess["model_type"] = model_type
        sess["model_idx"] = model_idx
        _wf4_session_touch(sess)
        if chat_id and message_id:
            label = _workflow4_combo_label(sess)
            await prompt_extended_resolution(
                chat_id=chat_id,
                message_id=message_id,
                label=label,
                job_id=job_id,
                tag_prefix="n",
            )
        return

    if tag == "npm":
        if len(parts) != 4:
            return
        try:
            model_type = parts[1]
            page = int(parts[2])
            job_id = parts[3]
        except Exception:
            return
        if model_type not in ("wan", "gguf"):
            return
        if not _wf4_session_get(job_id):
            return
        if chat_id and message_id:
            await edit_keyboard(
                chat_id,
                message_id,
                build_unet_keyboard_ext(job_id, model_type, page=page, tag_prefix="n"),
            )
        return

    if tag == "nmm":
        if len(parts) != 4:
            return
        try:
            job_id = parts[3]
        except Exception:
            return
        sess = _wf4_session_get(job_id)
        if not sess:
            return
        try:
            model_type, model_idx, _model_key, _model_cfg = _get_default_model_selection()
        except Exception:
            if chat_id and message_id:
                label = _workflow4_combo_label(sess)
                await edit_message_text(chat_id, message_id, f"{label}: chyba modelu", json.dumps({"inline_keyboard": []}))
            return
        sess["model_type"] = model_type
        sess["model_idx"] = model_idx
        _wf4_session_touch(sess)
        if chat_id and message_id:
            label = _workflow4_combo_label(sess)
            await prompt_extended_resolution(
                chat_id=chat_id,
                message_id=message_id,
                label=label,
                job_id=job_id,
                tag_prefix="n",
            )
        return

    if tag == "nlf":
        if len(parts) != 3:
            return
        use_last_frame = str(parts[1]).strip().lower() in ("1", "true", "yes", "y", "on")
        job_id = parts[2]
        sess = _wf4_session_get(job_id)
        if not sess:
            return
        sess["use_last_frame"] = use_last_frame
        _wf4_session_touch(sess)
        if chat_id and message_id:
            label = _workflow4_combo_label(sess)
            await prompt_workflow4_anchor_mode(
                chat_id=chat_id,
                message_id=message_id,
                label=label,
                job_id=job_id,
            )
        return

    if tag == "nla":
        if len(parts) != 3:
            return
        anchor_mode = str(parts[1]).strip().lower()
        if anchor_mode not in ("off", "anchor", "blend"):
            return
        job_id = parts[2]
        sess = _wf4_session_get(job_id)
        if not sess:
            return
        sess["anchor_mode"] = anchor_mode
        _wf4_session_touch(sess)
        if chat_id and message_id:
            label = _workflow4_combo_label(sess)
            await prompt_extended_resolution(
                chat_id=chat_id,
                message_id=message_id,
                label=label,
                job_id=job_id,
                tag_prefix="n",
            )
        return

    if tag == "nrs":
        if len(parts) != 3:
            return
        try:
            resolution_idx = int(parts[1])
            job_id = parts[2]
        except Exception:
            return
        if _resolution_by_index(resolution_idx) is None:
            return
        sess = _wf4_session_get(job_id)
        if not sess:
            return
        sess["resolution_idx"] = resolution_idx
        _wf4_session_touch(sess)
        if chat_id and message_id:
            label = _workflow4_combo_label(sess)
            await prompt_extended_steps(
                chat_id=chat_id,
                message_id=message_id,
                label=label,
                job_id=job_id,
                tag_prefix="n",
            )
        return

    if tag == "nst":
        if len(parts) != 3:
            return
        try:
            steps_idx = int(parts[1])
            job_id = parts[2]
        except Exception:
            return
        if _steps_by_index(steps_idx) is None:
            return
        sess = _wf4_session_get(job_id)
        if not sess:
            return
        sess["steps_idx"] = steps_idx
        sess["use_prompt"] = True
        sess["use_lora"] = True
        sess["use_last_frame"] = False
        sess["anchor_mode"] = "off"
        sess["drift_key"] = "default"
        sess["rife_multiplier"] = None
        _wf4_session_touch(sess)
        if chat_id and message_id:
            label = _workflow4_combo_label(sess)
            await prompt_workflow4_interpolation(
                chat_id=chat_id,
                message_id=message_id,
                label=label,
                job_id=job_id,
            )
        return

    if tag == "npu":
        if len(parts) != 3:
            return
        prompt_mode = str(parts[1]).strip().lower()
        prompt_only = prompt_mode in ("2", "prompt-only", "prompt_only")
        use_prompt = prompt_only or prompt_mode in ("1", "true", "yes", "y", "on")
        use_lora = not prompt_only
        job_id = parts[2]
        sess = _wf4_session_get(job_id)
        if not sess:
            return
        sess["use_prompt"] = use_prompt
        sess["use_lora"] = use_lora
        sess["drift_key"] = None
        sess["rife_multiplier"] = None
        _wf4_session_touch(sess)
        if chat_id and message_id:
            label = _workflow4_combo_label(sess)
            await prompt_workflow4_drift(
                chat_id=chat_id,
                message_id=message_id,
                label=label,
                job_id=job_id,
            )
        return

    if tag == "nd":
        if len(parts) != 3:
            return
        drift_key = parts[1]
        job_id = parts[2]
        sess = _wf4_session_get(job_id)
        if not sess:
            return
        preset = _get_drift_preset(drift_key)
        if not preset:
            return
        sess["drift_key"] = drift_key
        _wf4_session_touch(sess)
        if chat_id and message_id:
            label = _workflow4_combo_label(sess)
            await prompt_workflow4_interpolation(
                chat_id=chat_id,
                message_id=message_id,
                label=label,
                job_id=job_id,
            )
        return

    if tag == "nri":
        if len(parts) != 3:
            return
        try:
            rife_multiplier = int(parts[1])
        except Exception:
            return
        if rife_multiplier not in INTERPOLATION_OPTIONS:
            return
        job_id = parts[2]
        sess = _wf4_session_get(job_id)
        if not sess:
            return
        sess["rife_multiplier"] = rife_multiplier
        _wf4_session_touch(sess)
        drift_key = str(sess.get("drift_key") or "default")
        preset = _get_drift_preset(drift_key)
        if not preset:
            return
        await _workflow4_submit_from_session(
            job_id=job_id,
            chat_id=chat_id,
            message_id=message_id,
            sess=sess,
            drift_preset=preset,
        )
        return

    if tag == "ep":
        if len(parts) != 3:
            return
        try:
            page = int(parts[1])
            job_id = parts[2]
        except Exception:
            return
        if not _ext_session_get(job_id):
            return
        if chat_id and message_id:
            await edit_keyboard(chat_id, message_id, build_lora_keyboard_ext(job_id, page=page))
        return

    if tag == "el":
        if len(parts) != 3:
            return
        try:
            idx = int(parts[1])
            job_id = parts[2]
        except Exception:
            return
        sess = _ext_session_get(job_id)
        if not sess:
            return
        lora_key, cfg = _get_lora_cfg_by_index(idx)
        if not cfg:
            return
        lora_type = str(cfg.get("type") or "single").lower()
        label = str(cfg.get("label") or lora_key)
        current = int(sess.get("current_lora") or 1)
        sess[f"lora{current}"] = {"key": lora_key, "label": label, "cfg": cfg, "type": lora_type}
        _ext_session_touch(sess)

        if not (chat_id and message_id):
            return
        if lora_type == "pair":
            reply_markup = build_weight_keyboard_ext(job_id, cfg, "high")
            await edit_message_text(chat_id, message_id, f"{label}: vyber HIGH vahu", reply_markup)
        else:
            reply_markup = build_weight_keyboard_ext(job_id, cfg, "single")
            await edit_message_text(chat_id, message_id, f"{label}: vyber vahu", reply_markup)
        return

    if tag == "ews":
        if len(parts) != 3:
            return
        try:
            weight_idx = int(parts[1])
            job_id = parts[2]
        except Exception:
            return
        sess = _ext_session_get(job_id)
        if not sess:
            return
        current = int(sess.get("current_lora") or 1)
        lora = sess.get(f"lora{current}") or {}
        cfg = lora.get("cfg")
        if not isinstance(cfg, dict):
            return
        if str(lora.get("type") or "single").lower() == "pair":
            return
        options, _ = _weight_options_for(cfg, "single")
        if weight_idx < 0 or weight_idx >= len(options):
            return
        lora["weight"] = options[weight_idx]
        sess[f"lora{current}"] = lora
        _ext_session_touch(sess)

        if not (chat_id and message_id):
            return
        if current == 1:
            sess["current_lora"] = 2
            _ext_session_touch(sess)
            await edit_message_text(
                chat_id,
                message_id,
                "Extended10s: vyber 2. LoRA",
                build_lora_keyboard_ext(job_id, page=0),
            )
            return

        label = _extended_combo_label(sess)
        await prompt_extended_model(chat_id=chat_id, message_id=message_id, label=label, job_id=job_id)
        return

    if tag == "ewh":
        if len(parts) != 3:
            return
        try:
            weight_idx = int(parts[1])
            job_id = parts[2]
        except Exception:
            return
        sess = _ext_session_get(job_id)
        if not sess:
            return
        current = int(sess.get("current_lora") or 1)
        lora = sess.get(f"lora{current}") or {}
        cfg = lora.get("cfg")
        if not isinstance(cfg, dict):
            return
        if str(lora.get("type") or "single").lower() != "pair":
            return
        options, _ = _weight_options_for(cfg, "high")
        if weight_idx < 0 or weight_idx >= len(options):
            return
        lora["high_weight"] = options[weight_idx]
        sess[f"lora{current}"] = lora
        _ext_session_touch(sess)

        if chat_id and message_id:
            reply_markup = build_weight_keyboard_ext(job_id, cfg, "low")
            await edit_message_text(chat_id, message_id, f"{lora.get('label')}: vyber LOW vahu", reply_markup)
        return

    if tag == "ewl":
        if len(parts) != 3:
            return
        try:
            weight_idx = int(parts[1])
            job_id = parts[2]
        except Exception:
            return
        sess = _ext_session_get(job_id)
        if not sess:
            return
        current = int(sess.get("current_lora") or 1)
        lora = sess.get(f"lora{current}") or {}
        cfg = lora.get("cfg")
        if not isinstance(cfg, dict):
            return
        if str(lora.get("type") or "single").lower() != "pair":
            return
        options, _ = _weight_options_for(cfg, "low")
        if weight_idx < 0 or weight_idx >= len(options):
            return
        lora["low_weight"] = options[weight_idx]
        sess[f"lora{current}"] = lora
        _ext_session_touch(sess)

        if not (chat_id and message_id):
            return
        if current == 1:
            sess["current_lora"] = 2
            _ext_session_touch(sess)
            await edit_message_text(
                chat_id,
                message_id,
                "Extended10s: vyber 2. LoRA",
                build_lora_keyboard_ext(job_id, page=0),
            )
            return

        label = _extended_combo_label(sess)
        await prompt_extended_model(chat_id=chat_id, message_id=message_id, label=label, job_id=job_id)
        return

    if tag == "emt":
        if len(parts) != 3:
            return
        job_id = parts[2]
        sess = _ext_session_get(job_id)
        if not sess:
            return
        try:
            model_type, model_idx, _model_key, _model_cfg = _get_default_model_selection()
        except Exception:
            if chat_id and message_id:
                label = _extended_combo_label(sess)
                await edit_message_text(chat_id, message_id, f"{label}: chyba modelu", json.dumps({"inline_keyboard": []}))
            return

        sess["model_type"] = model_type
        sess["model_idx"] = model_idx
        _ext_session_touch(sess)
        if chat_id and message_id:
            label = _extended_combo_label(sess)
            await prompt_extended_last_frame(chat_id=chat_id, message_id=message_id, label=label, job_id=job_id)
        return

    if tag == "epm":
        if len(parts) != 4:
            return
        try:
            model_type = parts[1]
            page = int(parts[2])
            job_id = parts[3]
        except Exception:
            return
        if model_type not in ("wan", "gguf"):
            return
        if not _ext_session_get(job_id):
            return
        if chat_id and message_id:
            await edit_keyboard(chat_id, message_id, build_unet_keyboard_ext(job_id, model_type, page=page))
        return

    if tag == "emm":
        if len(parts) != 4:
            return
        try:
            job_id = parts[3]
        except Exception:
            return
        sess = _ext_session_get(job_id)
        if not sess:
            return
        try:
            model_type, model_idx, _model_key, _model_cfg = _get_default_model_selection()
        except Exception:
            if chat_id and message_id:
                label = _extended_combo_label(sess)
                await edit_message_text(chat_id, message_id, f"{label}: chyba modelu", json.dumps({"inline_keyboard": []}))
            return
        sess["model_type"] = model_type
        sess["model_idx"] = model_idx
        _ext_session_touch(sess)
        if chat_id and message_id:
            label = _extended_combo_label(sess)
            await prompt_extended_last_frame(chat_id=chat_id, message_id=message_id, label=label, job_id=job_id)
        return

    if tag == "elf":
        if len(parts) != 3:
            return
        use_last_frame = str(parts[1]).strip().lower() in ("1", "true", "yes", "y", "on")
        job_id = parts[2]
        sess = _ext_session_get(job_id)
        if not sess:
            return
        sess["use_last_frame"] = use_last_frame
        _ext_session_touch(sess)
        if chat_id and message_id:
            label = _extended_combo_label(sess)
            await prompt_extended_resolution(chat_id=chat_id, message_id=message_id, label=label, job_id=job_id)
        return

    if tag == "ers":
        if len(parts) != 3:
            return
        try:
            resolution_idx = int(parts[1])
            job_id = parts[2]
        except Exception:
            return
        if _resolution_by_index(resolution_idx) is None:
            return
        sess = _ext_session_get(job_id)
        if not sess:
            return
        sess["resolution_idx"] = resolution_idx
        _ext_session_touch(sess)
        if chat_id and message_id:
            label = _extended_combo_label(sess)
            await prompt_extended_steps(chat_id=chat_id, message_id=message_id, label=label, job_id=job_id)
        return

    if tag == "est":
        if len(parts) != 3:
            return
        try:
            steps_idx = int(parts[1])
            job_id = parts[2]
        except Exception:
            return
        if _steps_by_index(steps_idx) is None:
            return
        sess = _ext_session_get(job_id)
        if not sess:
            return
        sess["steps_idx"] = steps_idx
        _ext_session_touch(sess)
        if chat_id and message_id:
            label = _extended_combo_label(sess)
            await prompt_extended_prompt(chat_id=chat_id, message_id=message_id, label=label, job_id=job_id)
        return

    if tag == "epu":
        if len(parts) != 3:
            return
        use_prompt = str(parts[1]).strip().lower() in ("1", "true", "yes", "y", "on")
        job_id = parts[2]
        sess = _ext_session_get(job_id)
        if not sess:
            return
        sess["use_prompt"] = use_prompt
        sess["rife_multiplier"] = None
        _ext_session_touch(sess)
        if chat_id and message_id:
            label = _extended_combo_label(sess)
            await prompt_extended_interpolation(
                chat_id=chat_id,
                message_id=message_id,
                label=label,
                job_id=job_id,
                default_value=DEFAULT_INTERPOLATION_CLASSIC,
            )
        return

    if tag == "eri":
        if len(parts) != 3:
            return
        try:
            rife_multiplier = int(parts[1])
        except Exception:
            return
        if rife_multiplier not in INTERPOLATION_OPTIONS:
            return
        job_id = parts[2]
        sess = _ext_session_get(job_id)
        if not sess:
            return
        sess["rife_multiplier"] = rife_multiplier
        _ext_session_touch(sess)

        lora1 = sess.get("lora1") or {}
        lora2 = sess.get("lora2") or {}
        if not lora1 or not lora2:
            return
        cfg1 = lora1.get("cfg")
        cfg2 = lora2.get("cfg")
        if not isinstance(cfg1, dict) or not isinstance(cfg2, dict):
            return

        if sess.get("resolution_idx") is None:
            resolution_idx = DEFAULT_RESOLUTION_IDX
        else:
            resolution_idx = int(sess.get("resolution_idx"))
        default_steps_idx = STEPS_OPTIONS.index(DEFAULT_STEPS) if DEFAULT_STEPS in STEPS_OPTIONS else 0
        if sess.get("steps_idx") is None:
            steps_idx = default_steps_idx
        else:
            steps_idx = int(sess.get("steps_idx"))
        resolution = _resolution_by_index(resolution_idx)
        steps = _steps_by_index(steps_idx)
        if not resolution or steps is None:
            return

        model_type = str(sess.get("model_type") or "")
        model_idx = int(sess.get("model_idx") or -1)
        if model_type not in ("wan", "gguf"):
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

        use_prompt = bool(sess.get("use_prompt"))
        prompt_text_1 = (cfg1.get("positive") or DEFAULT_PROMPT) if use_prompt else DEFAULT_PROMPT
        prompt_text_2 = (cfg2.get("positive") or DEFAULT_PROMPT) if use_prompt else DEFAULT_PROMPT

        weights1 = {
            "weight": lora1.get("weight"),
            "high_weight": lora1.get("high_weight"),
            "low_weight": lora1.get("low_weight"),
        }
        weights2 = {
            "weight": lora2.get("weight"),
            "high_weight": lora2.get("high_weight"),
            "low_weight": lora2.get("low_weight"),
        }

        await _submit_extended_job(
            job_id=job_id,
            lora1_key=str(lora1.get("key")),
            lora1_cfg=cfg1,
            lora1_weights=weights1,
            lora2_key=str(lora2.get("key")),
            lora2_cfg=cfg2,
            lora2_weights=weights2,
            use_gguf=(model_type == "gguf"),
            use_last_frame=bool(sess.get("use_last_frame")),
            video_width=int(resolution["width"]),
            video_height=int(resolution["height"]),
            total_steps=int(steps),
            positive_prompt_1=prompt_text_1,
            positive_prompt_2=prompt_text_2,
            rife_multiplier=rife_multiplier,
            model_label=model_label,
            model_high_filename=model_high_filename,
            model_low_filename=model_low_filename,
            chat_id=chat_id,
            message_id=message_id,
        )
        _ext_session_clear(job_id)
        return

    if tag == "t5p":
        if len(parts) != 3:
            return
        try:
            page = int(parts[1])
            job_id = parts[2]
        except Exception:
            return
        if chat_id and message_id:
            await edit_keyboard(chat_id, message_id, build_lora_keyboard_ext(job_id, page=page, tag_prefix="t5"))
        return

    if tag == "t5l":
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

        label = str(cfg.get("label") or lora_key)
        lora_type = str(cfg.get("type") or "single").lower()
        try:
            model_type, _model_idx, model_key, model_cfg = _get_default_model_selection()
        except Exception:
            if chat_id and message_id:
                await edit_message_text(chat_id, message_id, f"{label}: chyba modelu", json.dumps({"inline_keyboard": []}))
            return

        prompt_text = cfg.get("positive") or DEFAULT_PROMPT

        model_label = str(model_cfg.get("label") or model_key)
        model_high_filename = model_cfg.get("high_filename")
        model_low_filename = model_cfg.get("low_filename")
        use_gguf = model_type == "gguf"

        group_job_ids = _test5s_group_get(job_id)
        target_job_ids = group_job_ids or [job_id]

        if lora_type == "pair":
            for i, target_job_id in enumerate(target_job_ids):
                await _submit_pair_job(
                    job_id=target_job_id,
                    lora_key=lora_key,
                    cfg=cfg,
                    high_weight=TEST5S_WEIGHT,
                    low_weight=TEST5S_WEIGHT,
                    use_gguf=use_gguf,
                    use_last_frame=False,
                    video_width=TEST5S_VIDEO_WIDTH,
                    video_height=TEST5S_VIDEO_HEIGHT,
                    total_steps=TEST5S_STEPS,
                    positive_prompt=prompt_text,
                    rife_multiplier=TEST5S_RIFE_MULTIPLIER,
                    model_label=model_label,
                    model_high_filename=model_high_filename,
                    model_low_filename=model_low_filename,
                    chat_id=chat_id if i == 0 else 0,
                    message_id=message_id if i == 0 else 0,
                )
            if group_job_ids:
                test5s_group_jobs.pop(job_id, None)
            return

        for i, target_job_id in enumerate(target_job_ids):
            await _submit_single_job(
                job_id=target_job_id,
                lora_key=lora_key,
                cfg=cfg,
                weight=TEST5S_WEIGHT,
                use_gguf=use_gguf,
                use_last_frame=False,
                video_width=TEST5S_VIDEO_WIDTH,
                video_height=TEST5S_VIDEO_HEIGHT,
                total_steps=TEST5S_STEPS,
                positive_prompt=prompt_text,
                rife_multiplier=TEST5S_RIFE_MULTIPLIER,
                model_label=model_label,
                model_high_filename=model_high_filename,
                model_low_filename=model_low_filename,
                chat_id=chat_id if i == 0 else 0,
                message_id=message_id if i == 0 else 0,
            )
        if group_job_ids:
            test5s_group_jobs.pop(job_id, None)
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
        try:
            model_type, model_idx, _model_key, _model_cfg = _get_default_model_selection()
        except Exception:
            await edit_message_text(chat_id, message_id, f"{label}: chyba modelu", json.dumps({"inline_keyboard": []}))
            return

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
        try:
            model_type, model_idx, _model_key, _model_cfg = _get_default_model_selection()
        except Exception:
            await edit_message_text(chat_id, message_id, f"{label}: chyba modelu", json.dumps({"inline_keyboard": []}))
            return

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
            try:
                model_type, model_idx, model_key, model_cfg = _get_default_model_selection()
            except Exception:
                if chat_id and message_id:
                    label = str(cfg.get("label") or lora_key)
                    await edit_message_text(chat_id, message_id, f"{label}: chyba modelu", json.dumps({"inline_keyboard": []}))
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
                    rife_multiplier=DEFAULT_INTERPOLATION_CLASSIC,
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
        try:
            model_type, model_idx, model_key, model_cfg = _get_default_model_selection()
        except Exception:
            if chat_id and message_id:
                label = str(cfg.get("label") or lora_key)
                await edit_message_text(chat_id, message_id, f"{label}: chyba modelu", json.dumps({"inline_keyboard": []}))
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
                rife_multiplier=DEFAULT_INTERPOLATION_CLASSIC,
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
        prompt_mode = 1 if use_prompt else 0
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
        label = str(cfg.get("label") or lora_key)

        resolution = _resolution_by_index(resolution_idx)
        if not resolution:
            return

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

            await prompt_interpolation(
                chat_id=chat_id,
                message_id=message_id,
                label=label,
                job_id=job_id,
                prompt_mode=prompt_mode,
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

        await prompt_interpolation(
            chat_id=chat_id,
            message_id=message_id,
            label=label,
            job_id=job_id,
            prompt_mode=prompt_mode,
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

    if tag == "ri":
        if len(parts) not in (11, 12):
            return
        try:
            rife_multiplier = int(parts[1])
        except Exception:
            return
        if rife_multiplier not in INTERPOLATION_OPTIONS:
            return

        use_prompt = str(parts[2]).strip().lower() in ("1", "true", "yes", "y", "on")
        try:
            steps_idx = int(parts[3])
            resolution_idx = int(parts[4])
        except Exception:
            return
        if _resolution_by_index(resolution_idx) is None:
            return
        steps = _steps_by_index(steps_idx)
        if steps is None:
            return

        use_last_frame = str(parts[5]).strip().lower() in ("1", "true", "yes", "y", "on")

        try:
            lora_idx = int(parts[6])
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

        if len(parts) == 11:
            if lora_type == "pair":
                return
            try:
                weight_idx = int(parts[7])
                model_type = parts[8]
                model_idx = int(parts[9])
                job_id = parts[10]
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
                rife_multiplier=rife_multiplier,
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
            high_idx = int(parts[7])
            low_idx = int(parts[8])
            model_type = parts[9]
            model_idx = int(parts[10])
            job_id = parts[11]
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
            rife_multiplier=rife_multiplier,
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
            job_id = parts[4]
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
        try:
            model_type, model_idx, model_key, model_cfg = _get_default_model_selection()
        except Exception:
            if chat_id and message_id:
                label = str(cfg.get("label") or lora_key)
                await edit_message_text(chat_id, message_id, f"{label}: chyba modelu", json.dumps({"inline_keyboard": []}))
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
            return

        model_label = str(model_cfg.get("label") or model_key)
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
            rife_multiplier=DEFAULT_INTERPOLATION_CLASSIC,
            model_label=model_label,
            model_high_filename=model_cfg.get("high_filename"),
            model_low_filename=model_cfg.get("low_filename"),
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
            job_id = parts[5]
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
        try:
            model_type, model_idx, model_key, model_cfg = _get_default_model_selection()
        except Exception:
            if chat_id and message_id:
                label = str(cfg.get("label") or lora_key)
                await edit_message_text(chat_id, message_id, f"{label}: chyba modelu", json.dumps({"inline_keyboard": []}))
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
            return

        model_label = str(model_cfg.get("label") or model_key)
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
            rife_multiplier=DEFAULT_INTERPOLATION_CLASSIC,
            model_label=model_label,
            model_high_filename=model_cfg.get("high_filename"),
            model_low_filename=model_cfg.get("low_filename"),
            chat_id=chat_id,
            message_id=message_id,
        )
        return

    return


async def process_update(update: Dict[str, Any]) -> Dict[str, Any]:
    _dedup(update)
    _flush_test5s_group_jobs()
    await _flush_test5s_album_sessions()

    if "callback_query" in update:
        await process_callback(update)
        return {"type": "callback"}

    chat_id, file_id, media_group_id = extract_photo(update)
    rate_limit(chat_id)

    if media_group_id:
        await _handle_test5s_album_photo(chat_id, file_id, str(media_group_id))
        return {"type": "photo_album", "media_group_id": str(media_group_id)}

    placeholder_id = await send_placeholder(chat_id)
    job_id = uuid.uuid4().hex  # shorter than str(uuid.uuid4())

    try:
        save_job_awaiting_lora(job_id, chat_id, placeholder_id, file_id)
    except Exception as exc:
        await edit_placeholder(chat_id, placeholder_id, f"Chyba DB: {exc}")
        raise

    await edit_placeholder(chat_id, placeholder_id, "Vyber mod…")
    try:
        await send_mode_picker(chat_id, job_id)
    except Exception as exc:
        await edit_placeholder(chat_id, placeholder_id, f"Chyba: neviem zobraziť výber módu ({exc})")
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
