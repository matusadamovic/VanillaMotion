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


def load_and_patch_workflow(
    input_filename: str,
    lora_filename: Optional[str],
    lora_strength: Optional[float],
    positive_prompt: Optional[str],
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

    # Patch LoRA loaders if provided
    if lora_filename:
        lora_path = pathlib.Path("/runpod-volume/models/loras") / lora_filename
        if not lora_path.exists():
            raise FileNotFoundError(f"LoRA file not found: {lora_path}")

        strength_val = float(lora_strength) if lora_strength is not None else 1.0

        for node in prompt.values():
            if node.get("class_type") != "Power Lora Loader (rgthree)":
                continue

            title = ((node.get("_meta") or {}).get("title") or "").lower()
            # Only patch known nodes in this workflow
            if "step 1 lora" not in title and "step 3 lora" not in title:
                continue

            inputs = node.setdefault("inputs", {})

            # Disable all existing lora slots
            for k, v in list(inputs.items()):
                if k.startswith("lora_") and isinstance(v, dict):
                    v["on"] = False

            inputs["lora_1"] = {
                "on": True,
                "lora": lora_filename,
                "strength": strength_val,
            }

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

        if mp4_found or completed_flag is True or success_flag is True or status_str_l in {"completed","complete","success","succeeded","done"}:
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

    lora_filename = payload.get("lora_filename")
    lora_strength = payload.get("lora_strength")
    positive_prompt = payload.get("positive_prompt")

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
                lora_filename=lora_filename,
                lora_strength=lora_strength,
                positive_prompt=positive_prompt,
            )

            with open(output_dir / "workflow_patched.json", "w", encoding="utf-8") as f:
                json.dump(workflow, f, ensure_ascii=False, indent=2)

            prompt_id = send_prompt(workflow)
            logging.info("COMFY_SUBMITTED prompt_id=%s", prompt_id)

            history = wait_for_prompt(prompt_id)

            with open(output_dir / "history.json", "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)

            logging.info("COMFY_DONE prompt_id=%s", prompt_id)
        finally:
            stop_comfy(comfy_proc)

        video_path = resolve_output_video(history, output_dir=output_dir)
        logging.info("VIDEO_PATH=%s size_bytes=%s", video_path, video_path.stat().st_size if video_path.exists() else None)

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