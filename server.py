#!/usr/bin/env python3
"""Mobile chat bridge: chat UI -> Ollama (qwen2.5-coder:7b) directly.

NOTE (2026-10-23): Claude Code's headless mode (`claude -p --model ...`) fails
on this setup with `[claude-code:unrecognized_model]` — the model isn't
recognized by Claude Code 2.1.310's SDK layer, so no agentic loop runs.
The interactive terminal path (`ollama launch claude`) is unaffected.
This server therefore talks to Ollama's chat API directly: reliable
question-answer + code. The Claude Code agent remains available in the
codespace terminal for real agentic work.
"""
import json as _json
import os
import queue
import threading
import urllib.request
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
OLLAMA_URL = "http://localhost:11434"
SYSTEM_PROMPT = (
    "Tum ek helpful coding assistant ho. User Hindi, Hinglish ya English me "
    "baat karega — usi bhasha me jawab do. Code hamesha ```language fences "
    "me do. Jawab saaf-suthra aur to-the-point rakho."
)

app = Flask(__name__)
jobs = {}
job_q = queue.Queue()
state = {"connected": True}
state_lock = threading.Lock()
history = []  # last 20 messages, single-user
history_lock = threading.Lock()


def is_connected():
    with state_lock:
        return state["connected"]


def ollama_chat(messages):
    body = _json.dumps(
        {
            "model": MODEL,
            "messages": messages,
            "stream": False,
            "options": {"num_ctx": 8192},
        }
    ).encode()
    req = urllib.request.Request(
        OLLAMA_URL + "/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=1500) as r:
        data = _json.loads(r.read().decode())
    return ((data.get("message") or {}).get("content") or "").strip()


def run_chat(prompt):
    with history_lock:
        history.append({"role": "user", "content": prompt})
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + history[-20:]
    try:
        reply = ollama_chat(msgs)
    except Exception as e:
        return None, "Ollama error: %s" % e
    if not reply:
        return None, "Khali jawab aya."
    with history_lock:
        history.append({"role": "assistant", "content": reply})
        del history[:-20]
    return reply, None


def worker():
    while True:
        job_id, prompt = job_q.get()
        jobs[job_id]["status"] = "running"
        try:
            reply, error = run_chat(prompt)
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


@app.get("/api/file")
def read_file():
    rel = (request.args.get("path") or "").strip()
    p = (PROJECT / rel).resolve()
    try:
        p.relative_to(PROJECT_R)
    except ValueError:
        return jsonify({"error": "not allowed"}), 403
    if not p.is_file():
        return jsonify({"error": "file nahi hai"}), 400
    if p.stat().st_size > 102400:
        return jsonify({"error": "file bahut badi hai (100KB se zyada)"}), 400
    try:
        text = p.read_text(encoding="utf-8", errors="strict")
    except Exception:
        return jsonify({"error": "ye text file nahi hai"}), 400
    return jsonify({"name": p.name, "content": text})


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
