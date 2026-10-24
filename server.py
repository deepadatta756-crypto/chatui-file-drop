#!/usr/bin/env python3
"""Mobile chat bridge: chat UI -> Claude Code (headless) -> local Ollama model."""
import os
import queue
import subprocess
import threading
import uuid
import zipfile
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

HOME = Path.home()
PROJECT = HOME / "myproject"
UPLOADS = PROJECT / "uploads"
UPLOADS.mkdir(parents=True, exist_ok=True)
APP_DIR = Path(__file__).resolve().parent
PROJECT_R = PROJECT.resolve()

MODEL = "qwen2.5-coder:7b"
# NOTE: --dangerously-skip-permissions is added by the setup task only if the
# headless permission test shows tool calls get blocked without it.
CLAUDE_FLAGS = ["-p", "--model", MODEL]

BASE_ENV = {
    "ANTHROPIC_BASE_URL": "http://localhost:11434",
    "ANTHROPIC_AUTH_TOKEN": "ollama",
    "ANTHROPIC_API_KEY": "",
}

app = Flask(__name__)
jobs = {}
job_q = queue.Queue()
state = {"connected": True}
state_lock = threading.Lock()


def is_connected():
    with state_lock:
        return state["connected"]


def run_claude(prompt):
    cmd = ["claude"] + CLAUDE_FLAGS + [prompt]
    env = dict(os.environ)
    env.update(BASE_ENV)
    try:
        p = subprocess.run(
            cmd, cwd=str(PROJECT), env=env,
            capture_output=True, text=True, timeout=1500,
        )
    except subprocess.TimeoutExpired:
        return None, "Time out ho gaya (25 min). Dobara try karo."
    out = (p.stdout or "").strip()
    err = (p.stderr or "").strip()
    if p.returncode != 0 and not out:
        return None, err or "Claude error code %d" % p.returncode
    return out, None


def worker():
    while True:
        job_id, prompt = job_q.get()
        jobs[job_id]["status"] = "running"
        try:
            reply, error = run_claude(prompt)
            jobs[job_id].update(status="done", reply=reply, error=error)
        except Exception as e:  # never leave a job hanging
            jobs[job_id].update(status="done", reply=None, error=str(e))
        finally:
            job_q.task_done()


threading.Thread(target=worker, daemon=True).start()


@app.get("/")
def index():
    return send_from_directory(str(APP_DIR), "index.html")


@app.get("/api/state")
def get_state():
    return jsonify(
        {
            "connected": is_connected(),
            "model": MODEL,
            "project": "~/myproject",
            "uploads": "~/myproject/uploads",
        }
    )


@app.post("/api/toggle")
def toggle():
    with state_lock:
        state["connected"] = not state["connected"]
        return jsonify({"connected": state["connected"]})


@app.post("/api/chat")
def chat():
    if not is_connected():
        return jsonify({"error": "Disconnected hai — pehle Connect dabao."}), 503
    data = request.get_json(force=True, silent=True) or {}
    msg = (data.get("message") or "").strip()
    if not msg:
        return jsonify({"error": "khali message"}), 400
    jid = uuid.uuid4().hex[:12]
    jobs[jid] = {"status": "queued", "reply": None, "error": None}
    job_q.put((jid, msg))
    return jsonify({"job_id": jid})


@app.get("/api/status/<jid>")
def status(jid):
    j = jobs.get(jid)
    if not j:
        return jsonify({"error": "unknown job"}), 404
    return jsonify(j)


@app.get("/api/files")
def files():
    rel = (request.args.get("path") or "").strip()
    p = (PROJECT / rel).resolve()
    try:
        p.relative_to(PROJECT_R)
    except ValueError:
        return jsonify({"error": "not allowed"}), 403
    if not p.is_dir():
        return jsonify({"error": "folder nahi hai"}), 400
    items = []
    for c in sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
        items.append({"name": c.name, "is_dir": c.is_dir()})
    disp = str(p.relative_to(PROJECT_R))
    return jsonify({"path": "/" if disp == "." else "/" + disp, "items": items[:200]})


@app.post("/api/upload")
def upload():
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"error": "koi file nahi mili"}), 400
    name = Path(f.filename).name
    dest = UPLOADS / name
    f.save(str(dest))
    files, target_rel = [], "uploads/"
    if name.lower().endswith(".zip"):
        target = UPLOADS / Path(name).stem
        try:
            target.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(str(dest)) as z:
                z.extractall(str(target))
            files = [
                str(p.relative_to(target))
                for p in sorted(target.rglob("*"))
                if p.is_file()
            ][:60]
            target_rel = "uploads/%s/" % Path(name).stem
        except Exception as e:
            return jsonify({"error": "unzip fail: %s" % e}), 400
    else:
        files = [name]
    return jsonify(
        {"ok": True, "saved": name, "extracted_to": target_rel, "files": files}
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, threaded=True)
