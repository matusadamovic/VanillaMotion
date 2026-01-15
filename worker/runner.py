import json
import logging
import os
import pathlib
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from typing import Any, Dict, Optional

import psycopg2
import requests
import runpod
from psycopg2.extras import RealDictCursor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

TELEGRAM_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
DATABASE_URL = os.environ["DATABASE_URL"]

WORKFLOW_PATH = os.environ.get("WORKFLOW_PATH", "/app/workflow.json")
COMFY_ROOT = os.environ.get("COMFY_ROOT", "/comfyui")
COMFY_PORT = int(os.environ.get("COMFY_PORT", "8188"))
COMFY_START_TIMEOUT = int(os.environ.get("COMFY_START_TIMEOUT", "600"))
COMFY_PROMPT_TIMEOUT = int(os.environ.get("COMFY_PROMPT_TIMEOUT", "1800"))

PERSIST_ROOT = os.environ.get("PERSIST_ROOT", "/runpod-volume/out")

PLACEHOLDER_DONE = {"COMPLETED", "FAILED", "CANCELLED"}


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
    positive_prompt: Optional[str],
    use_gguf: Optional[bool],
) -> Dict[str, Any]:
    with open(WORKFLOW_PATH, "r", encoding="utf-8") as f:
        prompt = json.load(f)

    # Must be API prompt format: node_id -> {class_type, inputs}
    if not isinstance(prompt, dict) or "nodes" in prompt:
        raise ValueError("WORKFLOW_PATH must be ComfyUI API prompt JSON (not UI workflow export).")

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


def upload_video(chat_id: int, message_id: int, video_path: pathlib.Path, caption: str = "Hotovo"):
    api = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
    with open(video_path, "rb") as f:
        files = {"media": ("video.mp4", f, "video/mp4")}
        data = {
            "chat_id": chat_id,
            "message_id": message_id,
            "media": json.dumps({"type": "video", "media": "attach://media", "caption": caption}),
        }
        r = requests.post(f"{api}/editMessageMedia", data=data, files=files, timeout=120)
        if not r.ok:
            f.seek(0)
            files2 = {"video": ("video.mp4", f, "video/mp4")}
            data2 = {"chat_id": chat_id, "caption": caption}
            requests.post(f"{api}/sendVideo", data=data2, files=files2, timeout=120)


def handler(event):
    payload = (event.get("input") or {})

    job_id = payload["job_id"]
    chat_id = int(payload["chat_id"])
    file_id = payload["input_file_id"]

    lora_type = (payload.get("lora_type") or "single")

    lora_filename = payload.get("lora_filename")
    lora_strength = payload.get("lora_strength")

    lora_high_filename = payload.get("lora_high_filename")
    lora_high_strength = payload.get("lora_high_strength")

    lora_low_filename = payload.get("lora_low_filename")
    lora_low_strength = payload.get("lora_low_strength")

    positive_prompt = payload.get("positive_prompt")
    use_gguf = payload.get("use_gguf")
    if isinstance(use_gguf, str):
        use_gguf = use_gguf.strip().lower() in ("1", "true", "yes", "y", "on")
    elif isinstance(use_gguf, (int, float)) and not isinstance(use_gguf, bool):
        use_gguf = bool(use_gguf)

    job_dir = pathlib.Path(tempfile.mkdtemp(prefix=f"job_{job_id}_", dir="/tmp"))
    temp_dir = job_dir / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    output_dir = pathlib.Path(PERSIST_ROOT) / job_id / time.strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("JOBDIR=%s", job_dir)
    logging.info("OUTPUT_DIR(persist)=%s", output_dir)
    logging.info("TEMP_DIR=%s", temp_dir)

    comfy_input_dir = pathlib.Path(COMFY_ROOT) / "input"
    target: Optional[pathlib.Path] = None

    try:
        state_row = mark_running(job_id)
        if not state_row:
            logging.warning("Job %s not in QUEUED (or not found). Skipping.", job_id)
            return {"status": "skipped"}

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

        comfy_proc = start_comfy(output_dir=output_dir, temp_dir=temp_dir)
        try:
            workflow = load_and_patch_workflow(
                input_filename=input_filename,
                lora_type=lora_type,
                lora_filename=lora_filename,
                lora_strength=lora_strength,
                lora_high_filename=lora_high_filename,
                lora_high_strength=lora_high_strength,
                lora_low_filename=lora_low_filename,
                lora_low_strength=lora_low_strength,
                positive_prompt=positive_prompt,
                use_gguf=use_gguf,
            )

            # Log: čo sme reálne poslali do Comfy (modely/loras/prompt/step1 vs step3)
            summary = summarize_workflow(workflow)
            logging.info("WORKFLOW_SUMMARY %s", json.dumps(summary, ensure_ascii=False))

            # Log: čo prišlo z payloadu (aby si porovnal s patched workflow)
            logging.info(
                "JOB_PARAMS lora_type=%s lora=%s s=%s high=%s hs=%s low=%s ls=%s use_gguf=%s positive_prompt=%s",
                lora_type,
                lora_filename,
                lora_strength,
                lora_high_filename,
                lora_high_strength,
                lora_low_filename,
                lora_low_strength,
                use_gguf,
                (positive_prompt[:120] + "…")
                if isinstance(positive_prompt, str) and len(positive_prompt) > 120
                else positive_prompt,
            )

            with open(output_dir / "workflow_patched.json", "w", encoding="utf-8") as f:
                json.dump(workflow, f, ensure_ascii=False, indent=2)

            prompt_id = send_prompt(workflow)
            logging.infoU = logging.info  # avoid accidental shadowing in some edits
            logging.info("COMFY_SUBMITTED prompt_id=%s", prompt_id)

            history = wait_for_prompt(prompt_id)

            with open(output_dir / "history.json", "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)

            logging.info("COMFY_DONE prompt_id=%s", prompt_id)
        finally:
            stop_comfy(comfy_proc)

        video_path = resolve_output_video(history, output_dir=output_dir)
        logging.info(
            "VIDEO_PATH=%s size_bytes=%s",
            video_path,
            video_path.stat().st_size if video_path.exists() else None,
        )

        finalize_info = finalize(job_id, "COMPLETED")
        if finalize_info:
            upload_video(int(finalize_info["chat_id"]), int(finalize_info["placeholder_message_id"]), video_path)

        return {"status": "completed", "video": str(video_path)}
    except Exception as exc:
        logging.exception("Job %s failed", job_id)
        finalize(job_id, "FAILED", error=str(exc))
        return {"status": "failed", "error": str(exc)}
    finally:
        try:
            if target is not None and target.exists():
                target.unlink()
        except Exception:
            logging.exception("Failed to cleanup comfy input file: %s", target)

        shutil.rmtree(job_dir, ignore_errors=True)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
