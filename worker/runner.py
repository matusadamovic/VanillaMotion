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
import uuid
from typing import Any, Dict, Optional

import psycopg2
import requests
import runpod
from psycopg2.extras import RealDictCursor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

TELEGRAM_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
DATABASE_URL = os.environ["DATABASE_URL"]
WORKFLOW_PATH = "/app/workflow.json"
COMFY_ROOT = "/app/ComfyUI"
COMFY_PORT = int(os.environ.get("COMFY_PORT", "8188"))
COMFY_START_TIMEOUT = int(os.environ.get("COMFY_START_TIMEOUT", "600"))
PLACEHOLDER_DONE = {"COMPLETED", "FAILED", "CANCELLED"}


def db_conn():
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)


def mark_running(job_id: str) -> Optional[Dict[str, Any]]:
    with db_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE jobs
            SET state = 'RUNNING', attempts = attempts + 1, updated_at = NOW()
            WHERE id = %s AND state IN ('QUEUED','RUNNING','FAILED')
            RETURNING *;
            """,
            (job_id,),
        )
        row = cur.fetchone()
        return row


def fetch_for_update(job_id: str) -> Optional[Dict[str, Any]]:
    with db_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT * FROM jobs WHERE id = %s FOR UPDATE", (job_id,))
        return cur.fetchone()


def finalize(job_id: str, state: str, error: Optional[str] = None) -> Optional[Dict[str, Any]]:
    with db_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT * FROM jobs WHERE id = %s FOR UPDATE", (job_id,))
        row = cur.fetchone()
        if not row:
            return None
        if row["state"] in PLACEHOLDER_DONE:
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


def download_telegram_file(file_id: str, dest_path: pathlib.Path, max_size: int = 10 * 1024 * 1024):
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


def start_comfy(output_dir: pathlib.Path, temp_dir: pathlib.Path) -> subprocess.Popen:
    cmd = [
        "/venv/bin/python",
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
    proc = subprocess.Popen(
        cmd,
        cwd=COMFY_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if proc.stdout is not None:
        threading.Thread(target=_stream_comfy_logs, args=(proc.stdout,), daemon=True).start()
    wait_until = time.time() + COMFY_START_TIMEOUT
    last_exc: Optional[Exception] = None
    while time.time() < wait_until:
        if proc.poll() is not None:
            raise RuntimeError(
                f"ComfyUI exited early (code={proc.returncode}). Check COMFY logs above."
            )
        try:
            resp = requests.get(f"http://127.0.0.1:{COMFY_PORT}/history", timeout=2)
            if resp.status_code == 200:
                return proc
        except Exception as exc:
            last_exc = exc
        time.sleep(2)
    proc.kill()
    raise RuntimeError(f"ComfyUI server did not start (timeout). Last error: {last_exc}")


def _stream_comfy_logs(stream):
    for line in stream:
        logging.info("COMFY %s", line.rstrip())


def stop_comfy(proc: subprocess.Popen):
    try:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=20)
    except Exception:
        proc.kill()


def load_and_patch_workflow(input_filename: str) -> Dict[str, Any]:
    with open(WORKFLOW_PATH, "r", encoding="utf-8") as f:
        workflow = json.load(f)
    for node in workflow.get("nodes", []):
        if node.get("type") == "LoadImage":
            widgets = node.get("widgets_values", [])
            if widgets:
                widgets[0] = input_filename
            if len(widgets) > 1:
                widgets[1] = "image"
            node["widgets_values"] = widgets
    return workflow


def send_prompt(workflow: Dict[str, Any]) -> str:
    url = f"http://127.0.0.1:{COMFY_PORT}/prompt"
    resp = requests.post(url, json={"prompt": workflow}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data["prompt_id"]


def wait_for_prompt(prompt_id: str) -> Dict[str, Any]:
    url = f"http://127.0.0.1:{COMFY_PORT}/history/{prompt_id}"
    deadline = time.time() + 900
    while time.time() < deadline:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            status = data.get("status", {}).get("status")
            if status == "completed":
                return data
            if status == "failed":
                raise RuntimeError("ComfyUI prompt failed")
        time.sleep(3)
    raise TimeoutError("ComfyUI prompt timeout")


def resolve_output_video(history: Dict[str, Any], output_dir: pathlib.Path) -> pathlib.Path:
    outputs = history.get("outputs", {})
    videos = []
    for value in outputs.values():
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and item.get("type") == "output" and item.get("filename", "").endswith(".mp4"):
                    videos.append(item)
    if videos:
        item = videos[0]
        subfolder = item.get("subfolder", "")
        filename = item["filename"]
        return output_dir.joinpath(subfolder, filename)
    candidates = sorted(output_dir.glob("**/*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        comfy_out = pathlib.Path(COMFY_ROOT) / "output"
        candidates = sorted(comfy_out.glob("**/*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise RuntimeError("No output video produced")
    return candidates[0]


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
            files = {"video": ("video.mp4", f, "video/mp4")}
            data = {"chat_id": chat_id, "caption": caption}
            requests.post(f"{api}/sendVideo", data=data, files=files, timeout=120)


def handler(event):
    payload = event.get("input", {})
    job_id = payload["job_id"]
    chat_id = int(payload["chat_id"])
    file_id = payload["input_file_id"]

    job_dir = pathlib.Path(tempfile.mkdtemp(prefix=f"job_{job_id}_", dir="/tmp"))
    output_dir = job_dir / "output"
    temp_dir = job_dir / "temp"
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    comfy_input_dir = pathlib.Path(COMFY_ROOT) / "input" / "image"

    try:
        state_row = mark_running(job_id)
        if not state_row:
            logging.warning("Job %s not found or already completed", job_id)
            return {"status": "skipped"}
        if state_row["state"] in PLACEHOLDER_DONE:
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
            workflow = load_and_patch_workflow(input_filename=input_filename)
            prompt_id = send_prompt(workflow)
            history = wait_for_prompt(prompt_id)
        finally:
            stop_comfy(comfy_proc)

        video_path = resolve_output_video(history, output_dir=output_dir)

        finalize_info = finalize(job_id, "COMPLETED")
        if finalize_info:
            upload_video(finalize_info["chat_id"], finalize_info["placeholder_message_id"], video_path)
        else:
            logging.info("Job already finalized, skipping upload")

        return {"status": "completed", "video": str(video_path)}
    except Exception as exc:
        logging.exception("Job %s failed", job_id)
        finalize(job_id, "FAILED", error=str(exc))
        return {"status": "failed", "error": str(exc)}
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
