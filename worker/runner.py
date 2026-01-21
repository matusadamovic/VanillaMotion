import json
import logging
import os
import pathlib
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional

import psycopg2
import requests
import runpod
from psycopg2.extras import RealDictCursor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

TELEGRAM_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
DATABASE_URL = os.environ["DATABASE_URL"]

WORKFLOW_PATH = os.environ.get("WORKFLOW_PATH", "/app/workflow.json")
WORKFLOW_PATH_NEW = os.environ.get("WORKFLOW_PATH_NEW", "/app/workflow_new.json")
COMFY_ROOT = os.environ.get("COMFY_ROOT", "/comfyui")
COMFY_PORT = int(os.environ.get("COMFY_PORT", "8188"))
COMFY_START_TIMEOUT = int(os.environ.get("COMFY_START_TIMEOUT", "600"))
COMFY_PROMPT_TIMEOUT = int(os.environ.get("COMFY_PROMPT_TIMEOUT", "1800"))

PERSIST_ROOT = os.environ.get("PERSIST_ROOT", "/runpod-volume/out")

VIDEO_OUTPUT_FORMAT = os.environ.get("VIDEO_OUTPUT_FORMAT")
VIDEO_OUTPUT_PIX_FMT = os.environ.get("VIDEO_OUTPUT_PIX_FMT")

PLACEHOLDER_DONE = {"COMPLETED", "FAILED", "CANCELLED"}
SAME_WOMAN_PROMPT = "The exact same woman"


def db_conn():
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)


def mark_running(job_id: str) -> Optional[Dict[str, Any]]:
    # IMPORTANT: only start from QUEUED to prevent duplicate renders
    with db_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE jobs
            SET state = 'RUNNING', attempts = attempts + 1, updated_at = NOW()
            WHERE id = %s AND state = 'QUEUED'
            RETURNING *;
            """,
            (job_id,),
        )
        return cur.fetchone()


def finalize(job_id: str, state: str, error: Optional[str] = None) -> Optional[Dict[str, Any]]:
    with db_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT * FROM jobs WHERE id = %s FOR UPDATE", (job_id,))
        row = cur.fetchone()
        if not row or row["state"] in PLACEHOLDER_DONE:
            return None
        cur.execute(
            """
            UPDATE jobs
            SET state = %s, error = %s, updated_at = NOW()
            WHERE id = %s
            RETURNING chat_id, placeholder_message_id;
            """,
            (state, error, job_id),
        )
        return cur.fetchone()


def download_telegram_file(file_id: str, dest_path: pathlib.Path, max_size: int = 10 * 1024 * 1024) -> str:
    api = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
    info = requests.get(f"{api}/getFile", params={"file_id": file_id}, timeout=30).json()
    file_path = info["result"]["file_path"]
    file_size = info["result"].get("file_size", max_size)
    if file_size > max_size:
        raise ValueError("Input file too large")

    url = f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{file_path}"
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    return pathlib.Path(file_path).suffix or ".png"


def _stream_comfy_logs(stream):
    for line in stream:
        logging.info("COMFY %s", line.rstrip())


def start_comfy(output_dir: pathlib.Path, temp_dir: pathlib.Path) -> subprocess.Popen:
    comfy_python = os.environ.get("COMFY_PYTHON") or shutil.which("python3") or shutil.which("python") or "python3"
    cmd = [
        comfy_python,
        f"{COMFY_ROOT}/main.py",
        "--disable-auto-launch",
        "--listen",
        "0.0.0.0",
        "--port",
        str(COMFY_PORT),
        "--output-directory",
        str(output_dir),
        "--temp-directory",
        str(temp_dir),
    ]

    logging.info("Starting ComfyUI: %s", " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        cwd=COMFY_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    logging.info("ComfyUI pid=%s", proc.pid)

    if proc.stdout is not None:
        threading.Thread(target=_stream_comfy_logs, args=(proc.stdout,), daemon=True).start()

    wait_until = time.time() + COMFY_START_TIMEOUT
    last_exc: Optional[Exception] = None
    while time.time() < wait_until:
        if proc.poll() is not None:
            raise RuntimeError(f"ComfyUI exited early (code={proc.returncode}). Check COMFY logs above.")
        try:
            resp = requests.get(f"http://127.0.0.1:{COMFY_PORT}/history", timeout=2)
            if resp.status_code == 200:
                return proc
        except Exception as exc:
            last_exc = exc
        time.sleep(2)

    proc.kill()
    raise RuntimeError(f"ComfyUI server did not start (timeout). Last error: {last_exc}")


def stop_comfy(proc: subprocess.Popen):
    try:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=20)
    except Exception:
        proc.kill()


# -----------------------------
# Workflow logging helpers
# -----------------------------
def _get_title(node: Dict[str, Any]) -> str:
    return str(((node.get("_meta") or {}).get("title") or "")).strip()


def _extract_part_index(title: str) -> Optional[int]:
    match = re.search(r"part\s*(\d+)", title, re.IGNORECASE)
    if not match:
        return None
    try:
        value = int(match.group(1))
    except Exception:
        return None
    return value if value > 0 else None


def _inject_same_woman_prompt(text: Optional[str]) -> Optional[str]:
    if not isinstance(text, str):
        return text
    stripped = text.strip()
    if not stripped:
        return text
    if SAME_WOMAN_PROMPT.lower() in stripped.lower():
        return text
    return f"{SAME_WOMAN_PROMPT}, {stripped}"


def _next_numeric_node_id(prompt: Dict[str, Any]) -> int:
    max_id = 0
    for key in prompt.keys():
        s = str(key)
        if s.isdigit():
            max_id = max(max_id, int(s))
    return max_id + 1


def _part_index_from_ref(prompt: Dict[str, Any], ref: Any) -> Optional[int]:
    if not isinstance(ref, list) or not ref:
        return None
    node = prompt.get(str(ref[0]))
    if not isinstance(node, dict):
        return None
    return _extract_part_index(_get_title(node))


def _collect_part_nodes(prompt: Dict[str, Any]) -> Dict[int, str]:
    part_nodes: Dict[int, str] = {}
    for node_id, node in prompt.items():
        if node.get("class_type") != "WanImageToVideoSVIPro":
            continue
        inputs = node.get("inputs") or {}
        part_idx = _part_index_from_ref(prompt, inputs.get("positive"))
        if part_idx:
            part_nodes[part_idx] = str(node_id)
    return part_nodes


def _collect_last_frame_nodes(prompt: Dict[str, Any]) -> Dict[int, str]:
    last_frame_nodes: Dict[int, str] = {}
    for node_id, node in prompt.items():
        if node.get("class_type") != "ImageFromBatch":
            continue
        inputs = node.get("inputs") or {}
        part_idx = _part_index_from_ref(prompt, inputs.get("image"))
        if part_idx:
            last_frame_nodes[part_idx] = str(node_id)
    return last_frame_nodes


def _get_vae_ref(prompt: Dict[str, Any]) -> Optional[List[Any]]:
    for node in prompt.values():
        if node.get("class_type") != "VAEEncode":
            continue
        inputs = node.get("inputs") or {}
        ref = inputs.get("vae")
        if isinstance(ref, list) and ref:
            return ref
    return None


def _apply_workflow4_last_frame_mode(prompt: Dict[str, Any], mode: Optional[str]) -> None:
    mode_value = str(mode or "").strip().lower()
    if mode_value in ("", "off", "none", "0"):
        return
    if mode_value not in ("anchor", "blend"):
        logging.warning("Unknown last-frame mode: %s", mode_value)
        return

    part_nodes = _collect_part_nodes(prompt)
    last_frame_nodes = _collect_last_frame_nodes(prompt)
    vae_ref = _get_vae_ref(prompt)
    if not part_nodes or not last_frame_nodes or not vae_ref:
        logging.warning("Last-frame mode skipped (missing nodes)")
        return

    next_id = _next_numeric_node_id(prompt)
    for part_idx in sorted(part_nodes.keys()):
        if part_idx <= 1:
            continue
        prev_part = part_idx - 1
        last_frame_node_id = last_frame_nodes.get(prev_part)
        if not last_frame_node_id:
            continue
        lf_encode_id = str(next_id)
        next_id += 1
        prompt[lf_encode_id] = {
            "inputs": {
                "pixels": [last_frame_node_id, 0],
                "vae": vae_ref,
            },
            "class_type": "VAEEncode",
            "_meta": {"title": f"Last Frame Encode: Part {part_idx}"},
        }
        target = prompt.get(part_nodes[part_idx])
        if not isinstance(target, dict):
            continue
        inputs = target.setdefault("inputs", {})
        if mode_value == "anchor":
            if "prev_samples" in inputs:
                inputs["prev_samples"] = [lf_encode_id, 0]
        else:
            inputs["anchor_samples"] = [lf_encode_id, 0]


def _guess_model_fields(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Heuristicky vyťahuje "čo sa loaduje" z inputs, lebo rôzne workflow používajú rôzne kľúče.
    """
    out: Dict[str, Any] = {}
    candidate_keys = [
        "ckpt_name",
        "checkpoint",
        "checkpoint_name",
        "model_name",
        "base_model",
        "unet_name",
        "unet",
        "diffusion_model",
        "diffusion_model_name",
        "vae_name",
        "vae",
        "clip_name",
        "clip",
    ]
    for k in candidate_keys:
        v = inputs.get(k)
        if isinstance(v, (str, int, float, bool)) and v not in ("", None):
            out[k] = v
    return out


def _summarize_lora_slots(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    rgthree Power Lora Loader sloty majú tvar:
      "lora_1": {"on": True, "lora": "...safetensors", "strength": 0.6}
    """
    slots: Dict[str, Any] = {}
    for k, v in inputs.items():
        if not isinstance(k, str) or not k.startswith("lora_"):
            continue
        if not isinstance(v, dict):
            continue
        slots[k] = {
            "on": bool(v.get("on")),
            "lora": v.get("lora"),
            "strength": v.get("strength"),
        }
    return slots


def summarize_workflow(prompt: Dict[str, Any]) -> Dict[str, Any]:
    """
    Vráti summary: model loadery + lory + prompty + či sa použil Step1 a/alebo Step3 LoRA.
    """
    models = []
    loras = []
    prompts = []

    found_step1 = False
    found_step3 = False
    step1_any_lora_on = False
    step3_any_lora_on = False

    for node_id, node in prompt.items():
        if not isinstance(node, dict):
            continue

        ct = str(node.get("class_type") or "")
        inputs = node.get("inputs") or {}
        if not isinstance(inputs, dict):
            inputs = {}

        # Prompty
        if ct == "CLIPTextEncode":
            t = _get_title(node)
            text = inputs.get("text")
            if isinstance(text, str):
                prompts.append({"node_id": node_id, "title": t, "text": text})

        # LoRA (rgthree)
        if ct == "Power Lora Loader (rgthree)":
            t = _get_title(node)
            slots = _summarize_lora_slots(inputs)
            loras.append({"node_id": node_id, "title": t, "slots": slots})

            t_l = t.lower()
            is_step1 = "step 1 lora" in t_l
            is_step3 = "step 3 lora" in t_l
            found_step1 = found_step1 or is_step1
            found_step3 = found_step3 or is_step3

            any_on = any(bool(s.get("on")) and s.get("lora") for s in slots.values())
            if is_step1 and any_on:
                step1_any_lora_on = True
            if is_step3 and any_on:
                step3_any_lora_on = True

        # Model loadery (heuristika)
        ct_l = ct.lower()
        if any(x in ct_l for x in ["checkpoint", "loader", "load", "unet", "vae", "clip"]):
            fields = _guess_model_fields(inputs)
            if fields:
                models.append(
                    {
                        "node_id": node_id,
                        "class_type": ct,
                        "title": _get_title(node),
                        "fields": fields,
                    }
                )

    # vyber “positive” / “negative” prompt podľa title
    positive = None
    negative = None
    for p in prompts:
        tl = (p.get("title") or "").lower()
        if "positive" in tl and positive is None:
            positive = p.get("text")
        if "negative" in tl and negative is None:
            negative = p.get("text")

    # Step mode (zmysel: či Step3 LoRA reálne beží)
    if found_step3:
        if step3_any_lora_on:
            step_mode = "STEP1_AND_STEP3"
        else:
            step_mode = "STEP1_ONLY (Step3 exists but all LoRA slots appear OFF)"
    else:
        step_mode = "STEP1_ONLY (no Step3 loader found)"

    return {
        "model_loaders": models,
        "loras": loras,
        "prompts": {
            "positive": positive,
            "negative": negative,
            "all_cliptextencode_nodes": prompts,
        },
        "steps": {
            "found_step1_loader": found_step1,
            "found_step3_loader": found_step3,
            "step1_any_lora_on": step1_any_lora_on,
            "step3_any_lora_on": step3_any_lora_on,
            "mode": step_mode,
        },
    }


def _patch_video_combine_outputs(prompt: Dict[str, Any]) -> None:
    fallback_format = "video/h264-mp4"
    fallback_pix_fmt = "yuv420p"

    for node in prompt.values():
        if node.get("class_type") != "VHS_VideoCombine":
            continue
        inputs = node.setdefault("inputs", {})
        current_format = inputs.get("format")

        # Avoid AV1 NVENC failures on hosts that don't support it.
        if VIDEO_OUTPUT_FORMAT:
            inputs["format"] = VIDEO_OUTPUT_FORMAT
        elif current_format == "video/nvenc_av1-mp4":
            inputs["format"] = fallback_format

        if VIDEO_OUTPUT_PIX_FMT:
            inputs["pix_fmt"] = VIDEO_OUTPUT_PIX_FMT
        elif inputs.get("format") == fallback_format:
            current_pix_fmt = inputs.get("pix_fmt")
            if current_pix_fmt in (None, "", "p010le"):
                inputs["pix_fmt"] = fallback_pix_fmt


def _patch_interpolation_multiplier(prompt: Dict[str, Any], value: int) -> None:
    for node in prompt.values():
        title_l = _get_title(node).lower()
        if title_l != "interpolation multiplier":
            continue
        class_type = node.get("class_type")
        if class_type in {"PrimitiveFloat", "PrimitiveInt", "FLOATConstant", "INTConstant"}:
            node.setdefault("inputs", {})["value"] = int(value)


def _read_numeric_value(node: Optional[Dict[str, Any]]) -> Optional[float]:
    if not isinstance(node, dict):
        return None
    class_type = node.get("class_type")
    if class_type not in {"PrimitiveFloat", "PrimitiveInt", "FLOATConstant", "INTConstant"}:
        return None
    inputs = node.get("inputs") or {}
    value = inputs.get("value")
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _collect_rife_ids(prompt: Dict[str, Any]) -> List[str]:
    return [str(node_id) for node_id, node in prompt.items() if node.get("class_type") == "RIFE VFI"]


def _get_interpolation_defaults(prompt: Dict[str, Any]) -> tuple[Optional[float], Optional[float]]:
    default_multiplier = None
    default_fps = None
    rife_ids = _collect_rife_ids(prompt)

    for rife_id in rife_ids:
        rife_node = prompt.get(rife_id) or {}
        ref = (rife_node.get("inputs") or {}).get("multiplier")
        if isinstance(ref, list) and ref:
            default_multiplier = _read_numeric_value(prompt.get(str(ref[0])))
            if default_multiplier is not None:
                break

    if default_multiplier is None:
        for node in prompt.values():
            if _get_title(node).lower() == "interpolation multiplier":
                default_multiplier = _read_numeric_value(node)
                if default_multiplier is not None:
                    break

    for node in prompt.values():
        if node.get("class_type") != "VHS_VideoCombine":
            continue
        inputs = node.get("inputs") or {}
        images = inputs.get("images")
        if not (isinstance(images, list) and images):
            continue
        if str(images[0]) not in rife_ids:
            continue
        frame_rate = inputs.get("frame_rate")
        if isinstance(frame_rate, list) and frame_rate:
            default_fps = _read_numeric_value(prompt.get(str(frame_rate[0])))
            if default_fps is not None:
                break
        if isinstance(frame_rate, (int, float, str)):
            try:
                default_fps = float(frame_rate)
                break
            except Exception:
                pass

    return default_multiplier, default_fps


def _patch_interpolation_fps(prompt: Dict[str, Any], value: float) -> None:
    rife_ids = _collect_rife_ids(prompt)
    for node in prompt.values():
        if node.get("class_type") != "VHS_VideoCombine":
            continue
        inputs = node.get("inputs") or {}
        images = inputs.get("images")
        if not (isinstance(images, list) and images):
            continue
        if str(images[0]) not in rife_ids:
            continue
        frame_rate = inputs.get("frame_rate")
        if isinstance(frame_rate, list) and frame_rate:
            ref_node = prompt.get(str(frame_rate[0]))
            if _read_numeric_value(ref_node) is not None:
                ref_node.setdefault("inputs", {})["value"] = float(value)
            else:
                inputs["frame_rate"] = float(value)
        else:
            inputs["frame_rate"] = float(value)


def _disable_rife_interpolation(prompt: Dict[str, Any]) -> None:
    for rife_id, rife_node in prompt.items():
        if rife_node.get("class_type") != "RIFE VFI":
            continue
        frames_ref = (rife_node.get("inputs") or {}).get("frames")
        if not (isinstance(frames_ref, list) and frames_ref):
            continue
        for node in prompt.values():
            inputs = node.get("inputs")
            if not isinstance(inputs, dict):
                continue
            for key, value in list(inputs.items()):
                if isinstance(value, list) and value and str(value[0]) == str(rife_id):
                    inputs[key] = list(frames_ref)


def _set_lora_slot(node: Dict[str, Any], slot: str, filename: str, strength: float) -> None:
    inputs = node.setdefault("inputs", {})
    inputs[slot] = {"on": True, "lora": filename, "strength": float(strength)}


def _disable_lora_slot(node: Dict[str, Any], slot: str) -> None:
    inputs = node.setdefault("inputs", {})
    v = inputs.get(slot)
    if isinstance(v, dict):
        v["on"] = False


def _patch_user_loras_rgthree(
    prompt: Dict[str, Any],
    *,
    lora_type: str,
    lora_filename: Optional[str],
    lora_strength: Optional[float],
    lora_high_filename: Optional[str],
    lora_high_strength: Optional[float],
    lora_low_filename: Optional[str],
    lora_low_strength: Optional[float],
    slot: str = "lora_2",
) -> None:
    """
    Patchuje iba user slot (default lora_2), aby sme nezničili autorove default LoRA v lora_1.
    - single: dá loru do Step 1/2/3
    - pair: Step1 vypne, Step2=high, Step3=low
    """
    lt = (lora_type or "single").lower()

    found_step2 = False
    found_step1 = False
    found_step3 = False

    for node in prompt.values():
        if node.get("class_type") != "Power Lora Loader (rgthree)":
            continue

        title = ((node.get("_meta") or {}).get("title") or "").lower()
        is_step1 = "step 1 lora" in title
        is_step2 = "step 2 lora" in title
        is_step3 = "step 3 lora" in title

        if not (is_step1 or is_step2 or is_step3):
            continue

        found_step1 = found_step1 or is_step1
        found_step2 = found_step2 or is_step2
        found_step3 = found_step3 or is_step3

        if lt == "pair":
            if is_step1:
                _disable_lora_slot(node, slot)

            if is_step2:
                if lora_high_filename:
                    _set_lora_slot(node, slot, lora_high_filename, float(lora_high_strength or 1.0))
                else:
                    _disable_lora_slot(node, slot)

            if is_step3:
                if lora_low_filename:
                    _set_lora_slot(node, slot, lora_low_filename, float(lora_low_strength or 1.0))
                else:
                    _disable_lora_slot(node, slot)

        else:
            # single
            if lora_filename:
                _set_lora_slot(node, slot, lora_filename, float(lora_strength or 1.0))
            else:
                _disable_lora_slot(node, slot)

    # Voliteľné: keď máš pair, ale workflow nemá Step2 loader, stojí za to to aspoň zalogovať
    if lt == "pair" and not found_step2:
        logging.warning(
            "PAIR LoRA requested but no 'Step 2 Lora' rgthree loader found in workflow. HighNoise will not be applied."
        )


def load_and_patch_workflow(
    input_filename: str,
    lora_type: str,
    lora_filename: Optional[str],
    lora_strength: Optional[float],
    lora_high_filename: Optional[str],
    lora_high_strength: Optional[float],
    lora_low_filename: Optional[str],
    lora_low_strength: Optional[float],
    model_high_filename: Optional[str],
    model_low_filename: Optional[str],
    positive_prompt: Optional[str],
    use_gguf: Optional[bool],
    use_last_frame: Optional[bool],
    video_width: Optional[int],
    video_height: Optional[int],
    total_steps: Optional[int],
    rife_multiplier: Optional[int],
) -> Dict[str, Any]:
    with open(WORKFLOW_PATH, "r", encoding="utf-8") as f:
        prompt = json.load(f)

    # Must be API prompt format: node_id -> {class_type, inputs}
    if not isinstance(prompt, dict) or "nodes" in prompt:
        raise ValueError("WORKFLOW_PATH must be ComfyUI API prompt JSON (not UI workflow export).")

    default_rife_multiplier, default_rife_fps = _get_interpolation_defaults(prompt)

    # Patch all LoadImage nodes to use downloaded file
    for node in prompt.values():
        if node.get("class_type") == "LoadImage":
            node.setdefault("inputs", {})["image"] = input_filename

    # Patch Positive Prompt if provided
    if positive_prompt:
        for node in prompt.values():
            if node.get("class_type") != "CLIPTextEncode":
                continue
            title = ((node.get("_meta") or {}).get("title") or "").lower()
            if "positive prompt" in title:
                node.setdefault("inputs", {})["text"] = positive_prompt

    # Patch GGUF toggle if provided
    if use_gguf is not None:
        for node in prompt.values():
            if node.get("class_type") != "BOOLConstant":
                continue
            title = ((node.get("_meta") or {}).get("title") or "").lower()
            if "use gguf" in title:
                node.setdefault("inputs", {})["value"] = bool(use_gguf)

    # Patch Last Frame toggle if provided
    if use_last_frame is not None:
        for node in prompt.values():
            if node.get("class_type") != "BOOLConstant":
                continue
            title = ((node.get("_meta") or {}).get("title") or "").lower()
            if "use last frame" in title:
                node.setdefault("inputs", {})["value"] = bool(use_last_frame)

    # Patch Video Width/Height if provided
    if video_width is not None or video_height is not None:
        for node in prompt.values():
            if node.get("class_type") != "Width/Height Literal (Image Saver)":
                continue
            title = _get_title(node).lower()
            if "video width" in title and video_width is not None:
                node.setdefault("inputs", {})["int"] = int(video_width)
            if "video height" in title and video_height is not None:
                node.setdefault("inputs", {})["int"] = int(video_height)

    # Patch Total Steps if providedd
    if total_steps is not None:
        for node in prompt.values():
            if node.get("class_type") != "INTConstant":
                continue
            title = _get_title(node).lower()
            if "total steps" in title:
                node.setdefault("inputs", {})["value"] = int(total_steps)

    # Patch interpolation (RIFE) if provided
    if rife_multiplier is not None:
        effective_rife = int(rife_multiplier)
        target_fps = None
        if default_rife_multiplier and default_rife_fps and default_rife_multiplier != 0:
            if effective_rife <= 0:
                target_fps = default_rife_fps / default_rife_multiplier
            else:
                target_fps = default_rife_fps * (effective_rife / default_rife_multiplier)
        if effective_rife <= 0:
            if target_fps is not None:
                _patch_interpolation_fps(prompt, target_fps)
            _disable_rife_interpolation(prompt)
        else:
            _patch_interpolation_multiplier(prompt, effective_rife)
            if target_fps is not None:
                _patch_interpolation_fps(prompt, target_fps)

    # Patch UNET selection if provided
    if model_high_filename or model_low_filename:
        if use_gguf is None:
            logging.warning("model_* provided but use_gguf is None; skipping UNET patch")
        else:
            def _unet_root() -> pathlib.Path:
                p = pathlib.Path("/runpod-volume/models/unet")
                if p.exists():
                    return p
                return pathlib.Path(COMFY_ROOT) / "models" / "unet"

            def _assert_unet_exists(filename: str) -> None:
                p = _unet_root() / filename
                if not p.exists():
                    raise FileNotFoundError(f"UNET file not found: {p}")

            if model_high_filename:
                _assert_unet_exists(model_high_filename)
            if model_low_filename:
                _assert_unet_exists(model_low_filename)

            target_class = "UnetLoaderGGUF" if use_gguf else "UNETLoader"
            for node in prompt.values():
                if node.get("class_type") != target_class:
                    continue
                title = _get_title(node).lower()
                if "high noise" in title and model_high_filename:
                    node.setdefault("inputs", {})["unet_name"] = model_high_filename
                if "low noise" in title and model_low_filename:
                    node.setdefault("inputs", {})["unet_name"] = model_low_filename

    # ---- LoRA patching (TAILORED to your workflow: Step 1 + Step 3 only) ----
    def _assert_lora_exists(filename: str) -> None:
        p = pathlib.Path("/runpod-volume/models/loras") / filename
        if not p.exists():
            raise FileNotFoundError(f"LoRA file not found: {p}")

    def _set_lora(inputs: Dict[str, Any], slot: str, filename: str, strength: float) -> None:
        inputs[slot] = {"on": True, "lora": filename, "strength": float(strength)}

    def _set_off(inputs: Dict[str, Any], slot: str) -> None:
        v = inputs.get(slot)
        if isinstance(v, dict):
            v["on"] = False

    lt = (lora_type or "single").lower()

    # Fail-fast validate files
    if lt == "pair":
        if not lora_high_filename or not lora_low_filename:
            raise ValueError("PAIR LoRA requires both lora_high_filename and lora_low_filename")
        _assert_lora_exists(lora_high_filename)
        _assert_lora_exists(lora_low_filename)
    else:
        if lora_filename:
            _assert_lora_exists(lora_filename)

    # Apply:
    # - single: Step1.lora_1 = file, Step3.lora_2 = file (keep Step3.lora_1 reserved/off)
    # - pair:   Step1.lora_1 = high, Step3.lora_2 = low  (keep Step3.lora_1 reserved/off)
    for node in prompt.values():
        if node.get("class_type") != "Power Lora Loader (rgthree)":
            continue

        title = ((node.get("_meta") or {}).get("title") or "").lower()
        if "step 1 lora" not in title and "step 3 lora" not in title:
            continue

        inputs = node.setdefault("inputs", {})

        if lt == "pair":
            high_s = float(lora_high_strength) if lora_high_strength is not None else 1.0
            low_s = float(lora_low_strength) if lora_low_strength is not None else 1.0

            if "step 1 lora" in title:
                # Step 1 has only lora_1 in your workflow
                _set_lora(inputs, "lora_1", lora_high_filename, high_s)

            if "step 3 lora" in title:
                # Keep lora_1 reserved/off (in your workflow it holds a lightning low-noise filename)
                _set_off(inputs, "lora_1")
                _set_lora(inputs, "lora_2", lora_low_filename, low_s)

        else:
            # single
            if not lora_filename:
                # nothing selected -> keep off (also avoids any accidental previous state)
                if "step 1 lora" in title:
                    _set_off(inputs, "lora_1")
                if "step 3 lora" in title:
                    _set_off(inputs, "lora_1")
                    _set_off(inputs, "lora_2")
                continue

            s = float(lora_strength) if lora_strength is not None else 1.0

            if "step 1 lora" in title:
                _set_lora(inputs, "lora_1", lora_filename, s)

            if "step 3 lora" in title:
                # Keep lora_1 reserved/off; apply user lora into lora_2
                _set_off(inputs, "lora_1")
                _set_lora(inputs, "lora_2", lora_filename, s)

    _patch_video_combine_outputs(prompt)

    return prompt


def load_and_patch_workflow_new(
    input_filename: str,
    lora_type: str,
    lora_filename: Optional[str],
    lora_strength: Optional[float],
    lora_high_filename: Optional[str],
    lora_high_strength: Optional[float],
    lora_low_filename: Optional[str],
    lora_low_strength: Optional[float],
    lora2_type: Optional[str],
    lora2_filename: Optional[str],
    lora2_strength: Optional[float],
    lora2_high_filename: Optional[str],
    lora2_high_strength: Optional[float],
    lora2_low_filename: Optional[str],
    lora2_low_strength: Optional[float],
    lora3_type: Optional[str],
    lora3_filename: Optional[str],
    lora3_strength: Optional[float],
    lora3_high_filename: Optional[str],
    lora3_high_strength: Optional[float],
    lora3_low_filename: Optional[str],
    lora3_low_strength: Optional[float],
    lora4_type: Optional[str],
    lora4_filename: Optional[str],
    lora4_strength: Optional[float],
    lora4_high_filename: Optional[str],
    lora4_high_strength: Optional[float],
    lora4_low_filename: Optional[str],
    lora4_low_strength: Optional[float],
    model_high_filename: Optional[str],
    model_low_filename: Optional[str],
    positive_prompt: Optional[str],
    positive_prompt_2: Optional[str],
    positive_prompt_3: Optional[str],
    positive_prompt_4: Optional[str],
    use_lora: Optional[bool],
    use_gguf: Optional[bool],
    last_frame_mode: Optional[str],
    video_width: Optional[int],
    video_height: Optional[int],
    total_steps: Optional[int],
    drift_speed_shift: Optional[float],
    drift_denoise: Optional[float],
    drift_overlap: Optional[int],
    drift_rife_multiplier: Optional[int],
    rife_multiplier: Optional[int],
) -> Dict[str, Any]:
    with open(WORKFLOW_PATH_NEW, "r", encoding="utf-8") as f:
        prompt = json.load(f)

    if not isinstance(prompt, dict) or "nodes" in prompt:
        raise ValueError("WORKFLOW_PATH_NEW must be ComfyUI API prompt JSON (not UI workflow export).")

    default_rife_multiplier, default_rife_fps = _get_interpolation_defaults(prompt)

    for node in prompt.values():
        if node.get("class_type") == "LoadImage":
            node.setdefault("inputs", {})["image"] = input_filename

    prompts_by_part = {
        1: positive_prompt,
        2: positive_prompt_2,
        3: positive_prompt_3,
        4: positive_prompt_4,
    }
    for part_idx in (2, 3, 4):
        prompts_by_part[part_idx] = _inject_same_woman_prompt(prompts_by_part.get(part_idx))
    for node in prompt.values():
        if node.get("class_type") != "CLIPTextEncode":
            continue
        title = _get_title(node)
        title_l = title.lower()
        if "positive" not in title_l:
            continue
        part_idx = _extract_part_index(title)
        if not part_idx:
            continue
        text = prompts_by_part.get(part_idx)
        if isinstance(text, str):
            node.setdefault("inputs", {})["text"] = text

    if total_steps is not None:
        for node in prompt.values():
            title_l = _get_title(node).lower()
            if title_l != "steps":
                continue
            class_type = node.get("class_type")
            inputs = node.setdefault("inputs", {})
            if class_type == "mxSlider":
                inputs["Xi"] = int(total_steps)
                inputs["Xf"] = int(total_steps)
            elif class_type in {"PrimitiveInt", "PrimitiveFloat", "INTConstant", "FLOATConstant"}:
                inputs["value"] = int(total_steps)

    if video_width is not None or video_height is not None:
        for node in prompt.values():
            if node.get("class_type") != "Width/Height Literal (Image Saver)":
                continue
            title_l = _get_title(node).lower()
            if "video width" in title_l and video_width is not None:
                node.setdefault("inputs", {})["int"] = int(video_width)
            if "video height" in title_l and video_height is not None:
                node.setdefault("inputs", {})["int"] = int(video_height)

    if video_width is not None and video_height is not None:
        for node in prompt.values():
            if node.get("class_type") != "CustomResolutionI2V":
                continue
            inputs = node.setdefault("inputs", {})
            inputs["manual_override"] = True
            inputs["manual_width"] = int(video_width)
            inputs["manual_height"] = int(video_height)

    if drift_speed_shift is not None:
        for node in prompt.values():
            title_l = _get_title(node).lower()
            if title_l != "speed / shift":
                continue
            class_type = node.get("class_type")
            inputs = node.setdefault("inputs", {})
            if class_type in {"PrimitiveFloat", "PrimitiveInt", "FLOATConstant", "INTConstant"}:
                inputs["value"] = float(drift_speed_shift)

    if drift_denoise is not None:
        for node in prompt.values():
            if node.get("class_type") != "BasicScheduler":
                continue
            node.setdefault("inputs", {})["denoise"] = float(drift_denoise)

    if drift_overlap is not None:
        for node in prompt.values():
            if node.get("class_type") != "ImageBatchExtendWithOverlap":
                continue
            node.setdefault("inputs", {})["overlap"] = int(drift_overlap)

    effective_rife = None
    if rife_multiplier is not None:
        effective_rife = int(rife_multiplier)
    elif drift_rife_multiplier is not None:
        effective_rife = int(drift_rife_multiplier)
    if effective_rife is not None:
        target_fps = None
        if default_rife_multiplier and default_rife_fps and default_rife_multiplier != 0:
            if effective_rife <= 0:
                target_fps = default_rife_fps / default_rife_multiplier
            else:
                target_fps = default_rife_fps * (effective_rife / default_rife_multiplier)
        if effective_rife <= 0:
            if target_fps is not None:
                _patch_interpolation_fps(prompt, target_fps)
            _disable_rife_interpolation(prompt)
        else:
            _patch_interpolation_multiplier(prompt, effective_rife)
            if target_fps is not None:
                _patch_interpolation_fps(prompt, target_fps)

    _apply_workflow4_last_frame_mode(prompt, last_frame_mode)

    apply_loras = True if use_lora is None else bool(use_lora)

    def _assert_lora_exists(filename: str) -> None:
        p = pathlib.Path("/runpod-volume/models/loras") / filename
        if not p.exists():
            raise FileNotFoundError(f"LoRA file not found: {p}")

    def _normalize_lora_part(
        name: str,
        lora_type_value: Optional[str],
        single_filename: Optional[str],
        single_strength: Optional[float],
        high_filename: Optional[str],
        high_strength: Optional[float],
        low_filename: Optional[str],
        low_strength: Optional[float],
    ) -> Dict[str, Any]:
        lt = str(lora_type_value or "single").lower()
        if lt == "pair":
            if not high_filename or not low_filename:
                raise ValueError(f"{name} requires both high and low LoRA filenames")
            _assert_lora_exists(high_filename)
            _assert_lora_exists(low_filename)
            return {
                "type": "pair",
                "high_filename": high_filename,
                "high_strength": float(high_strength) if high_strength is not None else 1.0,
                "low_filename": low_filename,
                "low_strength": float(low_strength) if low_strength is not None else 1.0,
            }

        if not single_filename:
            raise ValueError(f"{name} requires lora_filename")
        _assert_lora_exists(single_filename)
        s = float(single_strength) if single_strength is not None else 1.0
        return {
            "type": "single",
            "high_filename": single_filename,
            "high_strength": s,
            "low_filename": single_filename,
            "low_strength": s,
        }

    part_loras: Dict[int, Dict[str, Any]] = {}
    if apply_loras:
        part_loras = {
            1: _normalize_lora_part(
                "workflow4 lora1",
                lora_type,
                lora_filename,
                lora_strength,
                lora_high_filename,
                lora_high_strength,
                lora_low_filename,
                lora_low_strength,
            ),
            2: _normalize_lora_part(
                "workflow4 lora2",
                lora2_type,
                lora2_filename,
                lora2_strength,
                lora2_high_filename,
                lora2_high_strength,
                lora2_low_filename,
                lora2_low_strength,
            ),
            3: _normalize_lora_part(
                "workflow4 lora3",
                lora3_type,
                lora3_filename,
                lora3_strength,
                lora3_high_filename,
                lora3_high_strength,
                lora3_low_filename,
                lora3_low_strength,
            ),
            4: _normalize_lora_part(
                "workflow4 lora4",
                lora4_type,
                lora4_filename,
                lora4_strength,
                lora4_high_filename,
                lora4_high_strength,
                lora4_low_filename,
                lora4_low_strength,
            ),
        }

    def _get_lora_name(node: Dict[str, Any], slot: str) -> str:
        inputs = node.get("inputs") or {}
        v = inputs.get(slot)
        if isinstance(v, dict):
            return str(v.get("lora") or "")
        return ""

    def _is_lightx2v_node(node: Dict[str, Any]) -> bool:
        return "lightx2v" in _get_lora_name(node, "lora_1").lower()

    def _is_svi_node(node: Dict[str, Any]) -> bool:
        return "svi" in _get_lora_name(node, "lora_1").lower()

    # Disable Lightning (LightX2v) so we only use base model -> SVI + user LoRA.
    for node in prompt.values():
        if node.get("class_type") != "Power Lora Loader (rgthree)":
            continue
        if _is_lightx2v_node(node):
            _disable_lora_slot(node, "lora_1")

    for node in prompt.values():
        if node.get("class_type") != "Power Lora Loader (rgthree)":
            continue
        title = _get_title(node)
        title_l = title.lower()
        if "lora high noise" not in title_l and "lora low noise" not in title_l:
            continue
        if not _is_svi_node(node):
            continue
        if not apply_loras:
            _disable_lora_slot(node, "lora_2")
            continue
        part_idx = _extract_part_index(title)
        if not part_idx:
            continue
        cfg = part_loras.get(part_idx)
        if not cfg:
            continue
        if "lora high noise" in title_l:
            _set_lora_slot(node, "lora_2", cfg["high_filename"], cfg["high_strength"])
        else:
            _set_lora_slot(node, "lora_2", cfg["low_filename"], cfg["low_strength"])

    def _unet_root() -> pathlib.Path:
        p = pathlib.Path("/runpod-volume/models/unet")
        if p.exists():
            return p
        return pathlib.Path(COMFY_ROOT) / "models" / "unet"

    def _assert_unet_exists(filename: str) -> None:
        p = _unet_root() / filename
        if not p.exists():
            raise FileNotFoundError(f"UNET file not found: {p}")

    def _resolve_node(ref: Any) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
        if not isinstance(ref, list) or not ref:
            return None, None
        node_id = str(ref[0])
        node = prompt.get(node_id)
        if not isinstance(node, dict):
            return node_id, None
        return node_id, node

    switches = []
    for switch_id, switch_node in prompt.items():
        if switch_node.get("class_type") != "Any Switch (rgthree)":
            continue
        inputs = switch_node.get("inputs") or {}
        any_01 = inputs.get("any_01")
        any_02 = inputs.get("any_02")
        id1, node1 = _resolve_node(any_01)
        id2, node2 = _resolve_node(any_02)
        if not node1 or not node2:
            continue
        if node1.get("class_type") == "UnetLoaderGGUF":
            gguf_ref = any_01
            gguf_node = node1
            unet_ref = any_02
            unet_node = node2
        elif node2.get("class_type") == "UnetLoaderGGUF":
            gguf_ref = any_02
            gguf_node = node2
            unet_ref = any_01
            unet_node = node1
        else:
            continue
        noise = None
        gguf_title = _get_title(gguf_node).lower()
        if "high" in gguf_title:
            noise = "high"
        elif "low" in gguf_title:
            noise = "low"
        switches.append(
            {
                "switch_id": str(switch_id),
                "gguf_ref": gguf_ref,
                "gguf_node": gguf_node,
                "unet_ref": unet_ref,
                "unet_node": unet_node,
                "noise": noise,
            }
        )

    if model_high_filename:
        _assert_unet_exists(model_high_filename)
    if model_low_filename:
        _assert_unet_exists(model_low_filename)

    if use_gguf is True:
        for info in switches:
            if info["noise"] == "high" and model_high_filename:
                info["gguf_node"].setdefault("inputs", {})["unet_name"] = model_high_filename
            if info["noise"] == "low" and model_low_filename:
                info["gguf_node"].setdefault("inputs", {})["unet_name"] = model_low_filename
    elif use_gguf is False:
        for info in switches:
            if info["noise"] == "high" and model_high_filename:
                info["unet_node"].setdefault("inputs", {})["unet_name"] = model_high_filename
            if info["noise"] == "low" and model_low_filename:
                info["unet_node"].setdefault("inputs", {})["unet_name"] = model_low_filename
    else:
        if model_high_filename or model_low_filename:
            logging.warning("model_* provided but use_gguf is None; skipping model selection patch")

    if use_gguf is not None:
        for info in switches:
            replacement = info["gguf_ref"] if use_gguf else info["unet_ref"]
            if not isinstance(replacement, list):
                continue
            for node in prompt.values():
                inputs = node.get("inputs") or {}
                for k, v in inputs.items():
                    if isinstance(v, list) and v and str(v[0]) == info["switch_id"]:
                        inputs[k] = list(replacement)

    _patch_video_combine_outputs(prompt)

    return prompt


def send_prompt(workflow: Dict[str, Any]) -> str:
    url = f"http://127.0.0.1:{COMFY_PORT}/prompt"
    resp = requests.post(url, json={"prompt": workflow}, timeout=30)
    if resp.status_code != 200:
        logging.error("COMFY prompt rejected: status=%s body=%s", resp.status_code, resp.text)
    resp.raise_for_status()
    return resp.json()["prompt_id"]


def _unwrap_history_payload(data: Dict[str, Any], prompt_id: str) -> Dict[str, Any]:
    if isinstance(data, dict) and prompt_id in data and isinstance(data[prompt_id], dict):
        return data[prompt_id]
    return data


def _queue_snapshot() -> Optional[Dict[str, Any]]:
    try:
        r = requests.get(f"http://127.0.0.1:{COMFY_PORT}/queue", timeout=3)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None


def wait_for_prompt(prompt_id: str) -> Dict[str, Any]:
    url = f"http://127.0.0.1:{COMFY_PORT}/history/{prompt_id}"
    deadline = time.time() + COMFY_PROMPT_TIMEOUT

    last_log = 0.0
    last_seen_outputs_count = -1
    last_seen_mp4: Optional[str] = None

    while time.time() < deadline:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            time.sleep(2)
            continue

        raw = resp.json()
        data = _unwrap_history_payload(raw, prompt_id)

        status_obj = data.get("status") or {}
        outputs_obj = data.get("outputs") or {}

        mp4_found = None
        try:
            if isinstance(outputs_obj, dict):
                for v in outputs_obj.values():
                    items = v if isinstance(v, list) else [v]
                    for item in items:
                        if not isinstance(item, dict):
                            continue
                        fn = (item.get("filename") or "")
                        if fn.endswith(".mp4"):
                            mp4_found = fn
                            break
                    if mp4_found:
                        break
        except Exception:
            pass

        status_str = status_obj.get("status_str") or status_obj.get("status") or status_obj.get("state") or ""
        status_str_l = str(status_str).lower()
        completed_flag = status_obj.get("completed")
        success_flag = status_obj.get("success")

        now = time.time()
        if now - last_log >= 5:
            q = _queue_snapshot()
            q_running = q.get("queue_running") if isinstance(q, dict) else None
            q_pending = q.get("queue_pending") if isinstance(q, dict) else None
            outputs_count = len(outputs_obj) if isinstance(outputs_obj, dict) else 0

            if outputs_count != last_seen_outputs_count or mp4_found != last_seen_mp4:
                logging.info(
                    "COMFY_PROGRESS prompt_id=%s status=%s completed=%s success=%s outputs=%s mp4=%s queue_running=%s queue_pending=%s",
                    prompt_id,
                    status_str,
                    completed_flag,
                    success_flag,
                    outputs_count,
                    mp4_found,
                    q_running,
                    q_pending,
                )
                last_seen_outputs_count = outputs_count
                last_seen_mp4 = mp4_found

            last_log = now

        if (
            mp4_found
            or completed_flag is True
            or success_flag is True
            or status_str_l in {"completed", "complete", "success", "succeeded", "done"}
        ):
            return data

        if status_str_l in {"failed", "error", "cancelled", "canceled"}:
            raise RuntimeError(f"ComfyUI prompt failed: status={status_str}")

        time.sleep(2)

    raise TimeoutError("ComfyUI prompt timeout")


def run_comfy_prompt(
    *,
    label: Optional[str],
    workflow_key: Optional[str] = None,
    input_filename: str,
    lora_type: str,
    lora_filename: Optional[str],
    lora_strength: Optional[float],
    lora_high_filename: Optional[str],
    lora_high_strength: Optional[float],
    lora_low_filename: Optional[str],
    lora_low_strength: Optional[float],
    lora2_type: Optional[str] = None,
    lora2_filename: Optional[str] = None,
    lora2_strength: Optional[float] = None,
    lora2_high_filename: Optional[str] = None,
    lora2_high_strength: Optional[float] = None,
    lora2_low_filename: Optional[str] = None,
    lora2_low_strength: Optional[float] = None,
    lora3_type: Optional[str] = None,
    lora3_filename: Optional[str] = None,
    lora3_strength: Optional[float] = None,
    lora3_high_filename: Optional[str] = None,
    lora3_high_strength: Optional[float] = None,
    lora3_low_filename: Optional[str] = None,
    lora3_low_strength: Optional[float] = None,
    lora4_type: Optional[str] = None,
    lora4_filename: Optional[str] = None,
    lora4_strength: Optional[float] = None,
    lora4_high_filename: Optional[str] = None,
    lora4_high_strength: Optional[float] = None,
    lora4_low_filename: Optional[str] = None,
    lora4_low_strength: Optional[float] = None,
    model_high_filename: Optional[str],
    model_low_filename: Optional[str],
    positive_prompt: Optional[str],
    positive_prompt_2: Optional[str] = None,
    positive_prompt_3: Optional[str] = None,
    positive_prompt_4: Optional[str] = None,
    use_lora: Optional[bool] = None,
    use_gguf: Optional[bool],
    use_last_frame: Optional[bool],
    last_frame_mode: Optional[str] = None,
    video_width: Optional[int],
    video_height: Optional[int],
    total_steps: Optional[int],
    drift_speed_shift: Optional[float] = None,
    drift_denoise: Optional[float] = None,
    drift_overlap: Optional[int] = None,
    drift_rife_multiplier: Optional[int] = None,
    rife_multiplier: Optional[int] = None,
    output_dir: pathlib.Path,
) -> tuple[Dict[str, Any], pathlib.Path]:
    if workflow_key == "new":
        workflow = load_and_patch_workflow_new(
            input_filename=input_filename,
            lora_type=lora_type,
            lora_filename=lora_filename,
            lora_strength=lora_strength,
            lora_high_filename=lora_high_filename,
            lora_high_strength=lora_high_strength,
            lora_low_filename=lora_low_filename,
            lora_low_strength=lora_low_strength,
            lora2_type=lora2_type,
            lora2_filename=lora2_filename,
            lora2_strength=lora2_strength,
            lora2_high_filename=lora2_high_filename,
            lora2_high_strength=lora2_high_strength,
            lora2_low_filename=lora2_low_filename,
            lora2_low_strength=lora2_low_strength,
            lora3_type=lora3_type,
            lora3_filename=lora3_filename,
            lora3_strength=lora3_strength,
            lora3_high_filename=lora3_high_filename,
            lora3_high_strength=lora3_high_strength,
            lora3_low_filename=lora3_low_filename,
            lora3_low_strength=lora3_low_strength,
            lora4_type=lora4_type,
            lora4_filename=lora4_filename,
            lora4_strength=lora4_strength,
            lora4_high_filename=lora4_high_filename,
            lora4_high_strength=lora4_high_strength,
            lora4_low_filename=lora4_low_filename,
            lora4_low_strength=lora4_low_strength,
            model_high_filename=model_high_filename,
            model_low_filename=model_low_filename,
            positive_prompt=positive_prompt,
            positive_prompt_2=positive_prompt_2,
            positive_prompt_3=positive_prompt_3,
            positive_prompt_4=positive_prompt_4,
            use_lora=use_lora,
            use_gguf=use_gguf,
            last_frame_mode=last_frame_mode,
            video_width=video_width,
            video_height=video_height,
            total_steps=total_steps,
            drift_speed_shift=drift_speed_shift,
            drift_denoise=drift_denoise,
            drift_overlap=drift_overlap,
            drift_rife_multiplier=drift_rife_multiplier,
            rife_multiplier=rife_multiplier,
        )
    else:
        workflow = load_and_patch_workflow(
            input_filename=input_filename,
            lora_type=lora_type,
            lora_filename=lora_filename,
            lora_strength=lora_strength,
            lora_high_filename=lora_high_filename,
            lora_high_strength=lora_high_strength,
            lora_low_filename=lora_low_filename,
            lora_low_strength=lora_low_strength,
            model_high_filename=model_high_filename,
            model_low_filename=model_low_filename,
            positive_prompt=positive_prompt,
            use_gguf=use_gguf,
            use_last_frame=use_last_frame,
            video_width=video_width,
            video_height=video_height,
            total_steps=total_steps,
            rife_multiplier=rife_multiplier,
        )

    label_tag = f"segment={label}" if label else "segment=single"
    summary = summarize_workflow(workflow)
    logging.info("WORKFLOW_SUMMARY %s %s", label_tag, json.dumps(summary, ensure_ascii=False))
    logging.info(
        "JOB_PARAMS %s lora_type=%s lora=%s s=%s high=%s hs=%s low=%s ls=%s model_high=%s model_low=%s use_gguf=%s use_last_frame=%s last_frame_mode=%s video=%sx%s steps=%s rife_multiplier=%s positive_prompt=%s",
        label_tag,
        lora_type,
        lora_filename,
        lora_strength,
        lora_high_filename,
        lora_high_strength,
        lora_low_filename,
        lora_low_strength,
        model_high_filename,
        model_low_filename,
        use_gguf,
        use_last_frame,
        last_frame_mode,
        video_width,
        video_height,
        total_steps,
        rife_multiplier,
        (positive_prompt[:120] + "…")
        if isinstance(positive_prompt, str) and len(positive_prompt) > 120
        else positive_prompt,
    )

    workflow_name = "workflow_patched.json" if not label else f"workflow_patched_{label}.json"
    with open(output_dir / workflow_name, "w", encoding="utf-8") as f:
        json.dump(workflow, f, ensure_ascii=False, indent=2)

    prompt_id = send_prompt(workflow)
    logging.info("COMFY_SUBMITTED %s prompt_id=%s", label_tag, prompt_id)

    history = wait_for_prompt(prompt_id)
    history_name = "history.json" if not label else f"history_{label}.json"
    with open(output_dir / history_name, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    logging.info("COMFY_DONE %s prompt_id=%s", label_tag, prompt_id)
    video_path = resolve_output_video(history, output_dir=output_dir)
    logging.info(
        "VIDEO_PATH %s path=%s size_bytes=%s",
        label_tag,
        video_path,
        video_path.stat().st_size if video_path.exists() else None,
    )
    return history, video_path


def resolve_output_video(history: Dict[str, Any], output_dir: pathlib.Path) -> pathlib.Path:
    outputs = history.get("outputs") or {}

    # 1) from history metadata
    try:
        if isinstance(outputs, dict):
            for v in outputs.values():
                items = v if isinstance(v, list) else [v]
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    fn = item.get("filename") or ""
                    if not fn.endswith(".mp4"):
                        continue
                    sub = item.get("subfolder") or ""
                    p = output_dir / sub / fn
                    if p.exists():
                        return p
    except Exception:
        pass

    # 2) scan persisted output dir
    candidates = sorted(output_dir.glob("**/*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]

    # 3) fallback comfy output
    comfy_out = pathlib.Path(COMFY_ROOT) / "output"
    candidates = sorted(comfy_out.glob("**/*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]

    raise RuntimeError(f"No output video produced. output_dir={output_dir}")


def _run_ffmpeg(cmd: list[str]) -> None:
    logging.info("FFMPEG %s", " ".join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        tail = proc.stdout[-4000:] if proc.stdout else ""
        logging.error("FFMPEG failed (code=%s): %s", proc.returncode, tail)
        raise RuntimeError("ffmpeg failed")


def extract_last_frame(video_path: pathlib.Path, output_path: pathlib.Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-sseof",
        "-0.1",
        "-i",
        str(video_path),
        "-vframes",
        "1",
        "-q:v",
        "2",
        str(output_path),
    ]
    _run_ffmpeg(cmd)


def concat_videos(video_paths: list[pathlib.Path], output_path: pathlib.Path) -> None:
    list_path = output_path.with_suffix(".txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for p in video_paths:
            f.write(f"file '{p.as_posix()}'\n")

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_path),
        "-c",
        "copy",
        str(output_path),
    ]
    try:
        _run_ffmpeg(cmd)
    except RuntimeError:
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
        _run_ffmpeg(cmd)


def _format_weight(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        v = float(value)
    except Exception:
        s = str(value).strip()
        return s or None
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return s or "0"


def _weight_suffix(value: Any) -> str:
    s = _format_weight(value)
    return f"@{s}" if s else ""


def _format_model_line(
    *,
    use_gguf: Optional[bool],
    model_label: Optional[str],
    model_high_filename: Optional[str],
    model_low_filename: Optional[str],
) -> str:
    if use_gguf is True:
        model_type_label = "GGUF"
    elif use_gguf is False:
        model_type_label = "WAN"
    else:
        model_type_label = "Model"

    label = str(model_label).strip() if model_label else ""
    line = f"Model: {model_type_label}"
    if label:
        line += f" ({label})"

    parts = []
    if model_high_filename:
        parts.append(f"high={model_high_filename}")
    if model_low_filename:
        parts.append(f"low={model_low_filename}")
    if parts:
        line += ", " + ", ".join(parts)
    elif not label:
        line += " (workflow default)"
    return line


def _format_lora_line(
    *,
    lora_type: str,
    lora_label: Optional[str],
    lora_key: Optional[str],
    lora_filename: Optional[str],
    lora_strength: Optional[float],
    lora_high_filename: Optional[str],
    lora_high_strength: Optional[float],
    lora_low_filename: Optional[str],
    lora_low_strength: Optional[float],
) -> str:
    lt = str(lora_type or "single").lower()
    label = ""
    if lora_label:
        label = str(lora_label).strip()
    elif lora_key:
        label = str(lora_key).strip()

    if lt == "pair":
        parts = []
        if lora_high_filename:
            parts.append(f"high={lora_high_filename}{_weight_suffix(lora_high_strength)}")
        if lora_low_filename:
            parts.append(f"low={lora_low_filename}{_weight_suffix(lora_low_strength)}")
        if not parts:
            return "LoRA: none"
        prefix = f"LoRA: {label} (pair)" if label else "LoRA: pair"
        return f"{prefix}, " + ", ".join(parts)

    weight_suffix = _weight_suffix(lora_strength)
    name = label or (str(lora_filename).strip() if lora_filename else "")
    if lora_filename and name and lora_filename != name:
        return f"LoRA: {name} ({lora_filename}{weight_suffix})"
    if name:
        return f"LoRA: {name}{weight_suffix}"
    if lora_filename:
        return f"LoRA: {lora_filename}{weight_suffix}"
    return "LoRA: none"


def _format_lora_line_named(name: str, **kwargs: Any) -> str:
    line = _format_lora_line(**kwargs)
    if line.startswith("LoRA:"):
        return f"{name}:{line[5:]}"
    return f"{name}: {line}"


def _format_render_line(
    *,
    video_width: Optional[int],
    video_height: Optional[int],
    total_steps: Optional[int],
) -> str:
    parts = []
    if video_width and video_height:
        parts.append(f"{video_width}x{video_height}")
    if total_steps:
        parts.append(f"steps={total_steps}")
    if parts:
        return "Render: " + ", ".join(parts)
    return "Render: workflow default"


def build_caption(
    *,
    use_gguf: Optional[bool],
    model_label: Optional[str],
    model_high_filename: Optional[str],
    model_low_filename: Optional[str],
    lora_type: str,
    lora_label: Optional[str],
    lora_key: Optional[str],
    lora_filename: Optional[str],
    lora_strength: Optional[float],
    lora_high_filename: Optional[str],
    lora_high_strength: Optional[float],
    lora_low_filename: Optional[str],
    lora_low_strength: Optional[float],
    video_width: Optional[int],
    video_height: Optional[int],
    total_steps: Optional[int],
) -> str:
    lines = ["Hotovo"]
    lines.append(
        _format_model_line(
            use_gguf=use_gguf,
            model_label=model_label,
            model_high_filename=model_high_filename,
            model_low_filename=model_low_filename,
        )
    )
    lines.append(
        _format_lora_line(
            lora_type=lora_type,
            lora_label=lora_label,
            lora_key=lora_key,
            lora_filename=lora_filename,
            lora_strength=lora_strength,
            lora_high_filename=lora_high_filename,
            lora_high_strength=lora_high_strength,
            lora_low_filename=lora_low_filename,
            lora_low_strength=lora_low_strength,
        )
    )
    lines.append(
        _format_render_line(
            video_width=video_width,
            video_height=video_height,
            total_steps=total_steps,
        )
    )
    return "\n".join(lines)


def build_caption_extended(
    *,
    use_gguf: Optional[bool],
    model_label: Optional[str],
    model_high_filename: Optional[str],
    model_low_filename: Optional[str],
    lora_type: str,
    lora_label: Optional[str],
    lora_key: Optional[str],
    lora_filename: Optional[str],
    lora_strength: Optional[float],
    lora_high_filename: Optional[str],
    lora_high_strength: Optional[float],
    lora_low_filename: Optional[str],
    lora_low_strength: Optional[float],
    lora2_type: str,
    lora2_label: Optional[str],
    lora2_key: Optional[str],
    lora2_filename: Optional[str],
    lora2_strength: Optional[float],
    lora2_high_filename: Optional[str],
    lora2_high_strength: Optional[float],
    lora2_low_filename: Optional[str],
    lora2_low_strength: Optional[float],
    video_width: Optional[int],
    video_height: Optional[int],
    total_steps: Optional[int],
) -> str:
    lines = ["Hotovo (extended10s)"]
    lines.append(
        _format_model_line(
            use_gguf=use_gguf,
            model_label=model_label,
            model_high_filename=model_high_filename,
            model_low_filename=model_low_filename,
        )
    )
    lines.append(
        _format_lora_line_named(
            "LoRA 1",
            lora_type=lora_type,
            lora_label=lora_label,
            lora_key=lora_key,
            lora_filename=lora_filename,
            lora_strength=lora_strength,
            lora_high_filename=lora_high_filename,
            lora_high_strength=lora_high_strength,
            lora_low_filename=lora_low_filename,
            lora_low_strength=lora_low_strength,
        )
    )
    lines.append(
        _format_lora_line_named(
            "LoRA 2",
            lora_type=lora2_type,
            lora_label=lora2_label,
            lora_key=lora2_key,
            lora_filename=lora2_filename,
            lora_strength=lora2_strength,
            lora_high_filename=lora2_high_filename,
            lora_high_strength=lora2_high_strength,
            lora_low_filename=lora2_low_filename,
            lora_low_strength=lora2_low_strength,
        )
    )
    lines.append(
        _format_render_line(
            video_width=video_width,
            video_height=video_height,
            total_steps=total_steps,
        )
    )
    return "\n".join(lines)


def build_caption_workflow4(
    *,
    use_gguf: Optional[bool],
    model_label: Optional[str],
    model_high_filename: Optional[str],
    model_low_filename: Optional[str],
    lora_type: str,
    lora_label: Optional[str],
    lora_key: Optional[str],
    lora_filename: Optional[str],
    lora_strength: Optional[float],
    lora_high_filename: Optional[str],
    lora_high_strength: Optional[float],
    lora_low_filename: Optional[str],
    lora_low_strength: Optional[float],
    lora2_type: str,
    lora2_label: Optional[str],
    lora2_key: Optional[str],
    lora2_filename: Optional[str],
    lora2_strength: Optional[float],
    lora2_high_filename: Optional[str],
    lora2_high_strength: Optional[float],
    lora2_low_filename: Optional[str],
    lora2_low_strength: Optional[float],
    lora3_type: str,
    lora3_label: Optional[str],
    lora3_key: Optional[str],
    lora3_filename: Optional[str],
    lora3_strength: Optional[float],
    lora3_high_filename: Optional[str],
    lora3_high_strength: Optional[float],
    lora3_low_filename: Optional[str],
    lora3_low_strength: Optional[float],
    lora4_type: str,
    lora4_label: Optional[str],
    lora4_key: Optional[str],
    lora4_filename: Optional[str],
    lora4_strength: Optional[float],
    lora4_high_filename: Optional[str],
    lora4_high_strength: Optional[float],
    lora4_low_filename: Optional[str],
    lora4_low_strength: Optional[float],
    video_width: Optional[int],
    video_height: Optional[int],
    total_steps: Optional[int],
) -> str:
    lines = ["Hotovo (workflow4)"]
    lines.append(
        _format_model_line(
            use_gguf=use_gguf,
            model_label=model_label,
            model_high_filename=model_high_filename,
            model_low_filename=model_low_filename,
        )
    )
    lines.append(
        _format_lora_line_named(
            "LoRA 1",
            lora_type=lora_type,
            lora_label=lora_label,
            lora_key=lora_key,
            lora_filename=lora_filename,
            lora_strength=lora_strength,
            lora_high_filename=lora_high_filename,
            lora_high_strength=lora_high_strength,
            lora_low_filename=lora_low_filename,
            lora_low_strength=lora_low_strength,
        )
    )
    lines.append(
        _format_lora_line_named(
            "LoRA 2",
            lora_type=lora2_type,
            lora_label=lora2_label,
            lora_key=lora2_key,
            lora_filename=lora2_filename,
            lora_strength=lora2_strength,
            lora_high_filename=lora2_high_filename,
            lora_high_strength=lora2_high_strength,
            lora_low_filename=lora2_low_filename,
            lora_low_strength=lora2_low_strength,
        )
    )
    lines.append(
        _format_lora_line_named(
            "LoRA 3",
            lora_type=lora3_type,
            lora_label=lora3_label,
            lora_key=lora3_key,
            lora_filename=lora3_filename,
            lora_strength=lora3_strength,
            lora_high_filename=lora3_high_filename,
            lora_high_strength=lora3_high_strength,
            lora_low_filename=lora3_low_filename,
            lora_low_strength=lora3_low_strength,
        )
    )
    lines.append(
        _format_lora_line_named(
            "LoRA 4",
            lora_type=lora4_type,
            lora_label=lora4_label,
            lora_key=lora4_key,
            lora_filename=lora4_filename,
            lora_strength=lora4_strength,
            lora_high_filename=lora4_high_filename,
            lora_high_strength=lora4_high_strength,
            lora_low_filename=lora4_low_filename,
            lora_low_strength=lora4_low_strength,
        )
    )
    lines.append(
        _format_render_line(
            video_width=video_width,
            video_height=video_height,
            total_steps=total_steps,
        )
    )
    return "\n".join(lines)


def upload_video(chat_id: int, message_id: int, video_path: pathlib.Path, caption: str = "Hotovo") -> bool:
    api = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
    sent_new = False
    with open(video_path, "rb") as f:
        files = {"video": ("video.mp4", f, "video/mp4")}
        data = {"chat_id": chat_id, "caption": caption}
        r = requests.post(f"{api}/sendVideo", data=data, files=files, timeout=120)
        sent_new = r.ok
        if not r.ok and message_id:
            f.seek(0)
            files2 = {"media": ("video.mp4", f, "video/mp4")}
            data2 = {
                "chat_id": chat_id,
                "message_id": message_id,
                "media": json.dumps({"type": "video", "media": "attach://media", "caption": caption}),
            }
            requests.post(f"{api}/editMessageMedia", data=data2, files=files2, timeout=120)
    return sent_new


def update_placeholder_text(chat_id: int, message_id: int, text: str) -> None:
    api = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
    try:
        requests.post(
            f"{api}/editMessageText",
            data={"chat_id": chat_id, "message_id": message_id, "text": text},
            timeout=30,
        )
    except Exception:
        logging.exception("Failed to update placeholder message text")


def handler(event):
    payload = (event.get("input") or {})

    job_id = payload["job_id"]
    chat_id = int(payload["chat_id"])
    file_id = payload["input_file_id"]

    lora_key = payload.get("lora_key")
    lora_label = payload.get("lora_label")

    lora_type = (payload.get("lora_type") or "single")

    lora_filename = payload.get("lora_filename")
    lora_strength = payload.get("lora_strength")

    lora_high_filename = payload.get("lora_high_filename")
    lora_high_strength = payload.get("lora_high_strength")

    lora_low_filename = payload.get("lora_low_filename")
    lora_low_strength = payload.get("lora_low_strength")

    workflow_key = payload.get("workflow_key")
    mode = str(payload.get("mode") or "").strip().lower()
    is_extended = mode == "extended10s"
    is_workflow4 = mode == "workflow4"
    is_batch = mode == "batch5s"
    if is_workflow4 and not workflow_key:
        workflow_key = "new"

    lora2_key = payload.get("lora2_key")
    lora2_label = payload.get("lora2_label")
    lora2_type = (payload.get("lora2_type") or "single")
    lora2_filename = payload.get("lora2_filename")
    lora2_strength = payload.get("lora2_strength")
    lora2_high_filename = payload.get("lora2_high_filename")
    lora2_high_strength = payload.get("lora2_high_strength")
    lora2_low_filename = payload.get("lora2_low_filename")
    lora2_low_strength = payload.get("lora2_low_strength")
    positive_prompt_2 = payload.get("positive_prompt_2")

    lora3_key = payload.get("lora3_key")
    lora3_label = payload.get("lora3_label")
    lora3_type = (payload.get("lora3_type") or "single")
    lora3_filename = payload.get("lora3_filename")
    lora3_strength = payload.get("lora3_strength")
    lora3_high_filename = payload.get("lora3_high_filename")
    lora3_high_strength = payload.get("lora3_high_strength")
    lora3_low_filename = payload.get("lora3_low_filename")
    lora3_low_strength = payload.get("lora3_low_strength")
    positive_prompt_3 = payload.get("positive_prompt_3")

    lora4_key = payload.get("lora4_key")
    lora4_label = payload.get("lora4_label")
    lora4_type = (payload.get("lora4_type") or "single")
    lora4_filename = payload.get("lora4_filename")
    lora4_strength = payload.get("lora4_strength")
    lora4_high_filename = payload.get("lora4_high_filename")
    lora4_high_strength = payload.get("lora4_high_strength")
    lora4_low_filename = payload.get("lora4_low_filename")
    lora4_low_strength = payload.get("lora4_low_strength")
    positive_prompt_4 = payload.get("positive_prompt_4")

    batch_loras = payload.get("batch_loras")
    batch_weights = payload.get("batch_weights")
    batch_default_prompt = payload.get("batch_default_prompt")
    batch_use_prompt = payload.get("batch_use_prompt")

    if is_extended:
        lora2_type_l = str(lora2_type or "single").lower()
        if lora2_type_l == "pair":
            if not lora2_high_filename or not lora2_low_filename:
                raise ValueError("extended10s requires lora2 high/low filenames")
        else:
            if not lora2_filename:
                raise ValueError("extended10s requires lora2 filename")

    def _require_lora_payload(
        name: str,
        lora_type_value: Optional[str],
        single_filename: Optional[str],
        high_filename: Optional[str],
        low_filename: Optional[str],
    ) -> None:
        lt = str(lora_type_value or "single").lower()
        if lt == "pair":
            if not high_filename or not low_filename:
                raise ValueError(f"{name} requires lora high/low filenames")
        else:
            if not single_filename:
                raise ValueError(f"{name} requires lora filename")

    use_lora = payload.get("use_lora")
    if use_lora is None:
        use_lora = True
    elif isinstance(use_lora, str):
        use_lora = use_lora.strip().lower() in ("1", "true", "yes", "y", "on")
    elif isinstance(use_lora, (int, float)) and not isinstance(use_lora, bool):
        use_lora = bool(use_lora)

    if is_workflow4 and use_lora:
        _require_lora_payload(
            "workflow4 lora1",
            lora_type,
            lora_filename,
            lora_high_filename,
            lora_low_filename,
        )
        _require_lora_payload(
            "workflow4 lora2",
            lora2_type,
            lora2_filename,
            lora2_high_filename,
            lora2_low_filename,
        )
        _require_lora_payload(
            "workflow4 lora3",
            lora3_type,
            lora3_filename,
            lora3_high_filename,
            lora3_low_filename,
        )
        _require_lora_payload(
            "workflow4 lora4",
            lora4_type,
            lora4_filename,
            lora4_high_filename,
            lora4_low_filename,
        )

    model_high_filename = payload.get("model_high_filename")
    model_low_filename = payload.get("model_low_filename")
    model_label = payload.get("model_label")

    positive_prompt = payload.get("positive_prompt")
    use_gguf = payload.get("use_gguf")
    if isinstance(use_gguf, str):
        use_gguf = use_gguf.strip().lower() in ("1", "true", "yes", "y", "on")
    elif isinstance(use_gguf, (int, float)) and not isinstance(use_gguf, bool):
        use_gguf = bool(use_gguf)

    use_last_frame = payload.get("use_last_frame")
    if isinstance(use_last_frame, str):
        use_last_frame = use_last_frame.strip().lower() in ("1", "true", "yes", "y", "on")
    elif isinstance(use_last_frame, (int, float)) and not isinstance(use_last_frame, bool):
        use_last_frame = bool(use_last_frame)

    if batch_use_prompt is None:
        batch_use_prompt = True
    elif isinstance(batch_use_prompt, str):
        batch_use_prompt = batch_use_prompt.strip().lower() in ("1", "true", "yes", "y", "on")
    elif isinstance(batch_use_prompt, (int, float)) and not isinstance(batch_use_prompt, bool):
        batch_use_prompt = bool(batch_use_prompt)

    def _to_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, bool):
            return int(value)
        try:
            return int(value)
        except Exception:
            try:
                return int(float(value))
            except Exception:
                return None

    def _to_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, bool):
            return float(int(value))
        try:
            return float(value)
        except Exception:
            return None

    def _to_float_list(values: Any) -> list[float]:
        if values is None:
            return []
        items = values if isinstance(values, list) else [values]
        out: list[float] = []
        for v in items:
            if isinstance(v, bool):
                continue
            try:
                out.append(float(v))
            except Exception:
                continue
        return out

    def _normalize_batch_loras(values: Any) -> list[dict[str, Any]]:
        if not isinstance(values, list):
            return []
        out: list[dict[str, Any]] = []
        for item in values:
            if not isinstance(item, dict):
                continue
            key = str(item.get("key") or "").strip()
            label = str(item.get("label") or key).strip()
            lora_type = str(item.get("type") or "single").lower()
            positive = item.get("positive")
            if lora_type == "pair":
                high_filename = item.get("high_filename")
                low_filename = item.get("low_filename")
                if not high_filename or not low_filename:
                    continue
                out.append(
                    {
                        "key": key,
                        "label": label,
                        "type": "pair",
                        "high_filename": high_filename,
                        "low_filename": low_filename,
                        "positive": positive,
                    }
                )
            else:
                filename = item.get("filename")
                if not filename:
                    continue
                out.append(
                    {
                        "key": key,
                        "label": label,
                        "type": "single",
                        "filename": filename,
                        "positive": positive,
                    }
                )
        return out

    def _should_skip_batch_error(exc: Exception) -> bool:
        msg = str(exc)
        if isinstance(exc, FileNotFoundError) and "LoRA file not found" in msg:
            return True
        if isinstance(exc, ValueError) and "LoRA" in msg and "requires" in msg:
            return True
        return False

    video_width = _to_int(payload.get("video_width"))
    video_height = _to_int(payload.get("video_height"))
    total_steps = _to_int(payload.get("total_steps"))
    drift_speed_shift = _to_float(payload.get("drift_speed_shift"))
    drift_denoise = _to_float(payload.get("drift_denoise"))
    drift_overlap = _to_int(payload.get("drift_overlap"))
    drift_rife_multiplier = _to_int(payload.get("drift_rife_multiplier"))
    rife_multiplier = _to_int(payload.get("rife_multiplier"))
    if rife_multiplier is None:
        rife_multiplier = drift_rife_multiplier
    last_frame_mode = payload.get("anchor_mode")

    job_dir = pathlib.Path(tempfile.mkdtemp(prefix=f"job_{job_id}_", dir="/tmp"))
    temp_dir = job_dir / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    output_dir = pathlib.Path(PERSIST_ROOT) / job_id / time.strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("JOBDIR=%s", job_dir)
    logging.info("OUTPUT_DIR(persist)=%s", output_dir)
    logging.info("TEMP_DIR=%s", temp_dir)

    comfy_input_dir = pathlib.Path(COMFY_ROOT) / "input"
    targets: list[pathlib.Path] = []

    try:
        state_row = mark_running(job_id)
        if not state_row:
            logging.warning("Job %s not in QUEUED (or not found). Skipping.", job_id)
            return {"status": "skipped"}
        placeholder_chat_id = int(state_row.get("chat_id") or chat_id)
        placeholder_message_id = int(state_row.get("placeholder_message_id") or 0)

        input_filename = f"{job_id}.png"
        input_path = job_dir / input_filename
        suffix = download_telegram_file(file_id, input_path)

        if suffix:
            new_name = f"{job_id}{suffix}"
            new_path = job_dir / new_name
            input_path.rename(new_path)
            input_path = new_path
            input_filename = new_name

        target = comfy_input_dir / input_filename
        shutil.copy(input_path, target)
        targets.append(target)

        video_path: Optional[pathlib.Path] = None
        batch_skipped = 0
        comfy_proc = start_comfy(output_dir=output_dir, temp_dir=temp_dir)
        try:
            if is_batch:
                batch_items = _normalize_batch_loras(batch_loras)
                weights = _to_float_list(batch_weights)
                if not batch_items:
                    raise ValueError("batch5s requires batch_loras")
                if not weights:
                    raise ValueError("batch5s requires batch_weights")
                default_prompt_text = batch_default_prompt if isinstance(batch_default_prompt, str) else ""
                total = len(batch_items) * len(weights)
                if placeholder_message_id:
                    update_placeholder_text(
                        placeholder_chat_id,
                        placeholder_message_id,
                        f"Batch 5s: 0/{total}",
                    )

                processed = 0
                last_video_path: Optional[pathlib.Path] = None
                for lora in batch_items:
                    lora_type = str(lora.get("type") or "single").lower()
                    lora_key = str(lora.get("key") or "").strip()
                    lora_label = str(lora.get("label") or lora_key).strip()
                    positive = lora.get("positive")
                    if batch_use_prompt and isinstance(positive, str) and positive.strip():
                        prompt_text = positive
                    else:
                        prompt_text = default_prompt_text

                    for weight in weights:
                        weight_label = f"{float(weight):.2f}".rstrip("0").rstrip(".") or "0"
                        try:
                            history, video_path = run_comfy_prompt(
                                label=None,
                                input_filename=input_filename,
                                lora_type=lora_type,
                                lora_filename=lora.get("filename"),
                                lora_strength=float(weight),
                                lora_high_filename=lora.get("high_filename"),
                                lora_high_strength=float(weight),
                                lora_low_filename=lora.get("low_filename"),
                                lora_low_strength=float(weight),
                                model_high_filename=model_high_filename,
                                model_low_filename=model_low_filename,
                                positive_prompt=prompt_text,
                                use_lora=True,
                                use_gguf=use_gguf,
                                use_last_frame=use_last_frame,
                                video_width=video_width,
                                video_height=video_height,
                                total_steps=total_steps,
                                rife_multiplier=rife_multiplier,
                                output_dir=output_dir,
                            )
                        except Exception as exc:
                            if _should_skip_batch_error(exc):
                                processed += 1
                                batch_skipped += 1
                                logging.warning(
                                    "Batch5s skip lora=%s weight=%s error=%s",
                                    lora_key,
                                    weight_label,
                                    exc,
                                )
                                if placeholder_message_id:
                                    update_placeholder_text(
                                        placeholder_chat_id,
                                        placeholder_message_id,
                                        f"Batch 5s: {processed}/{total} (skip {lora_label}, w={weight_label})",
                                    )
                                continue
                            raise
                        processed += 1
                        last_video_path = video_path
                        caption = build_caption(
                            use_gguf=use_gguf,
                            model_label=model_label,
                            model_high_filename=model_high_filename,
                            model_low_filename=model_low_filename,
                            lora_type=lora_type,
                            lora_label=lora_label,
                            lora_key=lora_key,
                            lora_filename=lora.get("filename") if lora_type != "pair" else None,
                            lora_strength=float(weight) if lora_type != "pair" else None,
                            lora_high_filename=lora.get("high_filename") if lora_type == "pair" else None,
                            lora_high_strength=float(weight) if lora_type == "pair" else None,
                            lora_low_filename=lora.get("low_filename") if lora_type == "pair" else None,
                            lora_low_strength=float(weight) if lora_type == "pair" else None,
                            video_width=video_width,
                            video_height=video_height,
                            total_steps=total_steps,
                        )
                        upload_video(placeholder_chat_id, placeholder_message_id, video_path, caption=caption)
                        if placeholder_message_id:
                            update_placeholder_text(
                                placeholder_chat_id,
                                placeholder_message_id,
                                f"Batch 5s: {processed}/{total} ({lora_label}, w={weight_label})",
                            )
                video_path = last_video_path
            elif is_extended:
                prompt2 = positive_prompt_2 if isinstance(positive_prompt_2, str) else positive_prompt

                history1, video1 = run_comfy_prompt(
                    label="seg1",
                    input_filename=input_filename,
                    lora_type=lora_type,
                    lora_filename=lora_filename,
                    lora_strength=lora_strength,
                    lora_high_filename=lora_high_filename,
                    lora_high_strength=lora_high_strength,
                    lora_low_filename=lora_low_filename,
                    lora_low_strength=lora_low_strength,
                    model_high_filename=model_high_filename,
                    model_low_filename=model_low_filename,
                    positive_prompt=positive_prompt,
                    use_lora=use_lora,
                    use_gguf=use_gguf,
                    use_last_frame=use_last_frame,
                    video_width=video_width,
                    video_height=video_height,
                    total_steps=total_steps,
                    rife_multiplier=rife_multiplier,
                    output_dir=output_dir,
                )

                last_frame_path = job_dir / f"{job_id}_last.png"
                extract_last_frame(video1, last_frame_path)
                input2_filename = last_frame_path.name
                target2 = comfy_input_dir / input2_filename
                shutil.copy(last_frame_path, target2)
                targets.append(target2)

                history2, video2 = run_comfy_prompt(
                    label="seg2",
                    input_filename=input2_filename,
                    lora_type=lora2_type,
                    lora_filename=lora2_filename,
                    lora_strength=lora2_strength,
                    lora_high_filename=lora2_high_filename,
                    lora_high_strength=lora2_high_strength,
                    lora_low_filename=lora2_low_filename,
                    lora_low_strength=lora2_low_strength,
                    model_high_filename=model_high_filename,
                    model_low_filename=model_low_filename,
                    positive_prompt=prompt2,
                    use_lora=use_lora,
                    use_gguf=use_gguf,
                    use_last_frame=use_last_frame,
                    video_width=video_width,
                    video_height=video_height,
                    total_steps=total_steps,
                    rife_multiplier=rife_multiplier,
                    output_dir=output_dir,
                )

                final_video_path = output_dir / "extended10s.mp4"
                concat_videos([video1, video2], final_video_path)
                video_path = final_video_path
            elif is_workflow4:
                history, video_path = run_comfy_prompt(
                    label=None,
                    workflow_key=workflow_key,
                    input_filename=input_filename,
                    lora_type=lora_type,
                    lora_filename=lora_filename,
                    lora_strength=lora_strength,
                    lora_high_filename=lora_high_filename,
                    lora_high_strength=lora_high_strength,
                    lora_low_filename=lora_low_filename,
                    lora_low_strength=lora_low_strength,
                    lora2_type=lora2_type,
                    lora2_filename=lora2_filename,
                    lora2_strength=lora2_strength,
                    lora2_high_filename=lora2_high_filename,
                    lora2_high_strength=lora2_high_strength,
                    lora2_low_filename=lora2_low_filename,
                    lora2_low_strength=lora2_low_strength,
                    lora3_type=lora3_type,
                    lora3_filename=lora3_filename,
                    lora3_strength=lora3_strength,
                    lora3_high_filename=lora3_high_filename,
                    lora3_high_strength=lora3_high_strength,
                    lora3_low_filename=lora3_low_filename,
                    lora3_low_strength=lora3_low_strength,
                    lora4_type=lora4_type,
                    lora4_filename=lora4_filename,
                    lora4_strength=lora4_strength,
                    lora4_high_filename=lora4_high_filename,
                    lora4_high_strength=lora4_high_strength,
                    lora4_low_filename=lora4_low_filename,
                    lora4_low_strength=lora4_low_strength,
                    model_high_filename=model_high_filename,
                    model_low_filename=model_low_filename,
                    positive_prompt=positive_prompt,
                    positive_prompt_2=positive_prompt_2,
                    positive_prompt_3=positive_prompt_3,
                    positive_prompt_4=positive_prompt_4,
                    use_lora=use_lora,
                    use_gguf=use_gguf,
                    use_last_frame=use_last_frame,
                    last_frame_mode=last_frame_mode,
                    video_width=video_width,
                    video_height=video_height,
                    total_steps=total_steps,
                    drift_speed_shift=drift_speed_shift,
                    drift_denoise=drift_denoise,
                    drift_overlap=drift_overlap,
                    drift_rife_multiplier=drift_rife_multiplier,
                    rife_multiplier=rife_multiplier,
                    output_dir=output_dir,
                )
            else:
                history, video_path = run_comfy_prompt(
                    label=None,
                    input_filename=input_filename,
                    lora_type=lora_type,
                    lora_filename=lora_filename,
                    lora_strength=lora_strength,
                    lora_high_filename=lora_high_filename,
                    lora_high_strength=lora_high_strength,
                    lora_low_filename=lora_low_filename,
                    lora_low_strength=lora_low_strength,
                    model_high_filename=model_high_filename,
                    model_low_filename=model_low_filename,
                    positive_prompt=positive_prompt,
                    use_lora=use_lora,
                    use_gguf=use_gguf,
                    use_last_frame=use_last_frame,
                    video_width=video_width,
                    video_height=video_height,
                    total_steps=total_steps,
                    rife_multiplier=rife_multiplier,
                    output_dir=output_dir,
                )
        finally:
            stop_comfy(comfy_proc)
        size_bytes = None
        if video_path:
            size_bytes = video_path.stat().st_size if video_path.exists() else None
        logging.info("FINAL_VIDEO_PATH=%s size_bytes=%s", video_path, size_bytes)

        if is_batch:
            finalize_info = finalize(job_id, "COMPLETED")
            if finalize_info:
                suffix = ""
                if batch_skipped:
                    suffix = f" Preskocene: {batch_skipped}."
                update_placeholder_text(
                    int(finalize_info["chat_id"]),
                    int(finalize_info["placeholder_message_id"]),
                    f"Batch 5s hotovo.{suffix} Videa su v novych spravach.",
                )
            return {"status": "completed", "video": str(video_path) if video_path else ""}

        finalize_info = finalize(job_id, "COMPLETED")
        if finalize_info:
            if is_extended:
                caption = build_caption_extended(
                    use_gguf=use_gguf,
                    model_label=model_label,
                    model_high_filename=model_high_filename,
                    model_low_filename=model_low_filename,
                    lora_type=lora_type,
                    lora_label=lora_label,
                    lora_key=lora_key,
                    lora_filename=lora_filename,
                    lora_strength=lora_strength,
                    lora_high_filename=lora_high_filename,
                    lora_high_strength=lora_high_strength,
                    lora_low_filename=lora_low_filename,
                    lora_low_strength=lora_low_strength,
                    lora2_type=lora2_type,
                    lora2_label=lora2_label,
                    lora2_key=lora2_key,
                    lora2_filename=lora2_filename,
                    lora2_strength=lora2_strength,
                    lora2_high_filename=lora2_high_filename,
                    lora2_high_strength=lora2_high_strength,
                    lora2_low_filename=lora2_low_filename,
                    lora2_low_strength=lora2_low_strength,
                    video_width=video_width,
                    video_height=video_height,
                    total_steps=total_steps,
                )
            elif is_workflow4:
                caption = build_caption_workflow4(
                    use_gguf=use_gguf,
                    model_label=model_label,
                    model_high_filename=model_high_filename,
                    model_low_filename=model_low_filename,
                    lora_type=lora_type,
                    lora_label=lora_label,
                    lora_key=lora_key,
                    lora_filename=lora_filename,
                    lora_strength=lora_strength,
                    lora_high_filename=lora_high_filename,
                    lora_high_strength=lora_high_strength,
                    lora_low_filename=lora_low_filename,
                    lora_low_strength=lora_low_strength,
                    lora2_type=lora2_type,
                    lora2_label=lora2_label,
                    lora2_key=lora2_key,
                    lora2_filename=lora2_filename,
                    lora2_strength=lora2_strength,
                    lora2_high_filename=lora2_high_filename,
                    lora2_high_strength=lora2_high_strength,
                    lora2_low_filename=lora2_low_filename,
                    lora2_low_strength=lora2_low_strength,
                    lora3_type=lora3_type,
                    lora3_label=lora3_label,
                    lora3_key=lora3_key,
                    lora3_filename=lora3_filename,
                    lora3_strength=lora3_strength,
                    lora3_high_filename=lora3_high_filename,
                    lora3_high_strength=lora3_high_strength,
                    lora3_low_filename=lora3_low_filename,
                    lora3_low_strength=lora3_low_strength,
                    lora4_type=lora4_type,
                    lora4_label=lora4_label,
                    lora4_key=lora4_key,
                    lora4_filename=lora4_filename,
                    lora4_strength=lora4_strength,
                    lora4_high_filename=lora4_high_filename,
                    lora4_high_strength=lora4_high_strength,
                    lora4_low_filename=lora4_low_filename,
                    lora4_low_strength=lora4_low_strength,
                    video_width=video_width,
                    video_height=video_height,
                    total_steps=total_steps,
                )
            else:
                caption = build_caption(
                    use_gguf=use_gguf,
                    model_label=model_label,
                    model_high_filename=model_high_filename,
                    model_low_filename=model_low_filename,
                    lora_type=lora_type,
                    lora_label=lora_label,
                    lora_key=lora_key,
                    lora_filename=lora_filename,
                    lora_strength=lora_strength,
                    lora_high_filename=lora_high_filename,
                    lora_high_strength=lora_high_strength,
                    lora_low_filename=lora_low_filename,
                    lora_low_strength=lora_low_strength,
                    video_width=video_width,
                    video_height=video_height,
                    total_steps=total_steps,
                )
            sent_new = upload_video(
                int(finalize_info["chat_id"]),
                int(finalize_info["placeholder_message_id"]),
                video_path,
                caption=caption,
            )
            if sent_new:
                update_placeholder_text(
                    int(finalize_info["chat_id"]),
                    int(finalize_info["placeholder_message_id"]),
                    "Hotovo. Video je v novej sprave.",
                )

        return {"status": "completed", "video": str(video_path)}
    except Exception as exc:
        logging.exception("Job %s failed", job_id)
        finalize(job_id, "FAILED", error=str(exc))
        return {"status": "failed", "error": str(exc)}
    finally:
        for target in targets:
            try:
                if target.exists():
                    target.unlink()
            except Exception:
                logging.exception("Failed to cleanup comfy input file: %s", target)

        shutil.rmtree(job_dir, ignore_errors=True)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
