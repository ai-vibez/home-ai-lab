#!/usr/bin/env python3
import json
import os
import subprocess
import time
from pathlib import Path
import requests
from flask import Flask, Response, jsonify, request, stream_with_context
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

MANIFESTS_DIR = Path("/usr/share/ollama/.ollama/models/manifests/registry.ollama.ai/library")
BLOBS_DIR     = Path("/usr/share/ollama/.ollama/models/blobs")
LLAMA_BIN     = Path("/home/vibez/llama-cpp/build/bin/llama-server")
LLAMA_PORT    = 8081
DEFAULT_NGL   = 99


# --- VRAM ---

def get_total_vram_mb():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        return int(r.stdout.strip())
    except Exception:
        return None


def ctx_options_for_remaining(remaining_gb):
    if remaining_gb >= 12:
        return [
            {"value": 4096,  "label": "4096 — recommended"},
            {"value": 8192,  "label": "8192"},
            {"value": 16384, "label": "16384"},
            {"value": 32768, "label": "32768 — experimental, may be slower"},
        ]
    elif remaining_gb >= 6:
        return [
            {"value": 4096,  "label": "4096 — recommended"},
            {"value": 8192,  "label": "8192"},
            {"value": 16384, "label": "16384 — experimental, may be slower"},
        ]
    elif remaining_gb >= 3:
        return [
            {"value": 4096, "label": "4096 — recommended"},
            {"value": 6144, "label": "6144"},
            {"value": 8192, "label": "8192 — experimental, may be slower"},
        ]
    elif remaining_gb >= 1.5:
        return [
            {"value": 4096, "label": "4096 — recommended"},
            {"value": 6144, "label": "6144 — experimental, may be slower"},
        ]
    else:
        return [{"value": 4096, "label": "4096 — recommended"}]


def get_ctx_options(blob_size_bytes):
    total_vram_mb = get_total_vram_mb()
    if total_vram_mb is None:
        return [{"value": 4096, "label": "4096 — recommended"}]
    remaining_gb = (total_vram_mb - blob_size_bytes / (1024 ** 2)) / 1024
    return ctx_options_for_remaining(remaining_gb)


# --- MODEL DISCOVERY ---

def discover_models():
    models = []
    if not MANIFESTS_DIR.exists():
        return models

    for model_dir in sorted(MANIFESTS_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        if "embed" in model_dir.name.lower():
            continue

        for tag_file in sorted(model_dir.iterdir()):
            if not tag_file.is_file():
                continue
            full_name = f"{model_dir.name}:{tag_file.name}"
            try:
                manifest   = json.loads(tag_file.read_text())
                gguf_layer = next(
                    (l for l in manifest.get("layers", [])
                     if l.get("mediaType") == "application/vnd.ollama.image.model"),
                    None
                )
                if not gguf_layer:
                    continue

                blob_path = BLOBS_DIR / gguf_layer["digest"].replace("sha256:", "sha256-")
                if not blob_path.exists():
                    continue

                blob_size = gguf_layer.get("size", blob_path.stat().st_size)
                models.append({
                    "name":            full_name,
                    "blob_path":       str(blob_path),
                    "blob_size_bytes": blob_size,
                    "size_gb":         round(blob_size / (1024 ** 3), 2),
                    "ngl":             DEFAULT_NGL,
                })
            except Exception:
                continue

    return models


# --- ACTIVE MODEL ---

def get_active_model():
    try:
        r = subprocess.run(["pgrep", "-a", "llama-server"], capture_output=True, text=True, timeout=5)
        if r.returncode != 0:
            return None, None
        for line in r.stdout.splitlines():
            parts = line.split()
            model = parts[parts.index("--alias") + 1] if "--alias" in parts else None
            ctx   = int(parts[parts.index("-c") + 1]) if "-c" in parts else None
            if model:
                return model, ctx
        return None, None
    except Exception:
        return None, None


# --- ENDPOINTS ---

@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/models")
def models():
    result = []
    for m in discover_models():
        result.append({
            "name":        m["name"],
            "size_gb":     m["size_gb"],
            "ngl":         m["ngl"],
            "ctx_options": get_ctx_options(m["blob_size_bytes"]),
        })
    return jsonify(result)


@app.route("/model/active")
def active_model():
    model, ctx = get_active_model()
    return jsonify({"model": model, "ctx": ctx, "running": model is not None})


@app.route("/ctx-options")
def ctx_options():
    model_name = request.args.get("model")
    if not model_name:
        return jsonify({"error": "model param required"}), 400
    for m in discover_models():
        if m["name"] == model_name:
            return jsonify({"model": model_name, "options": get_ctx_options(m["blob_size_bytes"])})
    return jsonify({"error": f"model {model_name} not found"}), 404


@app.route("/model/switch", methods=["POST"])
def switch_model():
    data = request.get_json()
    if not data or "model" not in data:
        return jsonify({"error": "model required"}), 400

    model_name = data["model"]
    ctx        = int(data.get("ctx", 4096))

    target = next((m for m in discover_models() if m["name"] == model_name), None)
    if not target:
        return jsonify({"error": f"model {model_name} not found"}), 404

    valid_ctx = [o["value"] for o in get_ctx_options(target["blob_size_bytes"])]
    if ctx not in valid_ctx:
        return jsonify({"error": f"ctx {ctx} invalid. valid: {valid_ctx}"}), 400

    # Stop cleanly via systemd, then launch outside systemd
    subprocess.run(["pkill", "-f", "llama-server"], capture_output=True)
    time.sleep(5)

    subprocess.Popen(
        [
            str(LLAMA_BIN),
            "--model", target["blob_path"],
            "-ngl",   str(target["ngl"]),
            "-c",     str(ctx),
            "--port", str(LLAMA_PORT),
            "--host", "0.0.0.0",
            "--alias", model_name,
        ],
        stdout=open("/tmp/llama-server.log", "w"),
        stderr=subprocess.STDOUT,
        start_new_session=True
    )

    return jsonify({"status": "switching", "model": model_name, "ctx": ctx, "poll": "/model/active"})


@app.route("/rag/query", methods=["POST"])
def rag_query():
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "question required"}), 400

    question = data["question"].strip()
    if not question:
        return jsonify({"error": "question cannot be empty"}), 400

    model_active, _ = get_active_model()
    if not model_active:
        return jsonify({"error": "no model loaded, try again shortly"}), 503

    limit = str(data.get("limit", 10))
    model = data.get("model")

    warnings = []
    if model and model != model_active:
        warnings.append(f"requested model {model} ignored — active model is {model_active}")
        model = None

    cmd = [
        "/home/vibez/lab/rag-env/bin/python3",
        "/home/vibez/lab/query.py",
        question,
        "--output", "json",
        "--limit", limit,
    ]
    if model:
        cmd += ["--model", model]

    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"}
        )

        stdout = r.stdout.strip()
        if not stdout:
            return jsonify({"error": "empty output from query.py", "stderr": r.stderr.strip()}), 500

        start = stdout.find("{")
        if start == -1:
            return jsonify({"error": "no JSON in output", "raw": stdout, "stderr": r.stderr.strip()}), 500

        result = json.loads(stdout[start:])
        if warnings:
            result["warnings"] = warnings
        return jsonify(result)

    except json.JSONDecodeError as e:
        return jsonify({"error": f"JSON parse failed: {e}", "raw": stdout}), 500
    except subprocess.TimeoutExpired:
        return jsonify({"error": "query timed out after 300s"}), 504
    except Exception as e:
        return jsonify({"error": str(e), "stderr": r.stderr.strip() if "r" in locals() else ""}), 500


# --- OPENAI-COMPATIBLE ---

@app.route("/v1/models", methods=["GET"])
def openai_models():
    return jsonify({
        "object": "list",
        "data": [
            {"id": m["name"], "object": "model", "created": int(time.time()), "owned_by": "local"}
            for m in discover_models()
        ]
    })


@app.route("/v1/chat/completions", methods=["POST"])
def openai_chat():
    data = request.get_json() or {}
    model_name = data.get("model")

    # Auto-switch model if needed
    if model_name:
        active_model, _ = get_active_model()
        if model_name != active_model:
            target = next((m for m in discover_models() if m["name"] == model_name), None)
            if not target:
                return jsonify({"error": {"message": f"model {model_name} not found", "type": "invalid_request_error"}}), 404

            ctx = get_ctx_options(target["blob_size_bytes"])[0]["value"]
            subprocess.run(["pkill", "-f", "llama-server"], capture_output=True)
            time.sleep(5)

            subprocess.Popen(
                [
                    str(LLAMA_BIN),
                    "--model", target["blob_path"],
                    "-ngl",   str(target["ngl"]),
                    "-c",     str(ctx),
                    "--port", str(LLAMA_PORT),
                    "--host", "0.0.0.0",
                    "--alias", model_name,
                ],
                stdout=open("/tmp/llama-server.log", "w"),
                stderr=subprocess.STDOUT,
                start_new_session=True
            )

            # Wait for llama-server ready (max 60s)
            for _ in range(30):
                time.sleep(2)
                active_model, _ = get_active_model()
                if active_model == model_name:
                    break
            time.sleep(3)

    # Proxy to llama-server with streaming
    is_stream = data.get("stream", False)
    try:
        upstream = requests.post(
            f"http://127.0.0.1:{LLAMA_PORT}/v1/chat/completions",
            json=data,
            stream=is_stream,
            timeout=300
        )
        if is_stream:
            def generate():
                for chunk in upstream.iter_content(chunk_size=None):
                    yield chunk
            return Response(
                stream_with_context(generate()),
                status=upstream.status_code,
                content_type=upstream.headers.get("Content-Type", "text/event-stream"),
                headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"}
            )
        return Response(
            upstream.content,
            status=upstream.status_code,
            content_type=upstream.headers.get("Content-Type", "application/json")
        )
    except requests.exceptions.Timeout:
        return jsonify({"error": {"message": "llama-server timeout", "type": "server_error"}}), 504
    except Exception as e:
        return jsonify({"error": {"message": str(e), "type": "server_error"}}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8082, debug=False)
