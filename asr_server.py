#!/usr/bin/env python3
"""Local ASR server for vtype.

Nemotron 3.5 ASR handles most languages; Swedish is routed to KBLab's KB-Whisper,
which is far more accurate on Swedish. Models load once onto the GPU and serve
transcriptions over a Unix socket, and optionally over HTTP (--http) so vtype on
other machines can use it as a remote backend.

Unix socket protocol (one request per connection):
  client -> server: a JSON header line, then the raw audio bytes
      {"lang": "en-US", "rate": 16000, "nbytes": 12345}\n<int16 mono PCM>
  server -> client: a JSON line
      {"text": "...", "seconds": 0.21}   or   {"error": "..."}

HTTP (--http HOST:PORT): OpenAI-compatible POST /v1/audio/transcriptions,
multipart form with `file` (16-bit PCM WAV) and optional `language`; `model` is
ignored. Answers {"text": "..."}. With --api-key (or $VTYPE_API_KEY) requests
must send `Authorization: Bearer <key>`.
"""
import argparse
import email.parser
import email.policy
import hmac
import io
import json
import os
import socket
import sys
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
from transformers import AutoModelForRNNT, AutoModelForSpeechSeq2Seq, AutoProcessor

NEMOTRON_ID = "nvidia/nemotron-3.5-asr-streaming-0.6b"
SWEDISH_ID = "KBLab/kb-whisper-large"
RATE = 16000
DEFAULT_SOCKET = os.path.expanduser("~/.cache/vtype/asr.sock")
RECORDINGS = os.path.expanduser("~/.cache/vtype/recordings")
KEEP_RECORDINGS = 50

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32


def log(msg):
    print(f"[asr] {msg}", file=sys.stderr, flush=True)


class Nemotron:
    def __init__(self, model_id=NEMOTRON_ID):
        t0 = time.time()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForRNNT.from_pretrained(model_id, device_map=DEVICE, dtype=DTYPE).eval()
        log(f"loaded {model_id} in {time.time() - t0:.1f}s")

    def transcribe(self, audio, lang):
        inputs = self.processor(audio, sampling_rate=RATE, language=lang or "auto")
        inputs.to(self.model.device, dtype=self.model.dtype)
        # the default max_length (50) truncates longer dictations
        max_new = int(len(audio) / RATE * 25) + 64
        out = self.model.generate(**inputs, return_dict_in_generate=True, max_new_tokens=max_new)
        text = self.processor.decode(out.sequences, skip_special_tokens=True)
        return " ".join(text) if isinstance(text, list) else text


class KBWhisper:
    def __init__(self, model_id=SWEDISH_ID):
        t0 = time.time()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, dtype=DTYPE, use_safetensors=True)
        self.model.to(DEVICE).eval()
        log(f"loaded {model_id} in {time.time() - t0:.1f}s")

    def transcribe(self, audio, lang):
        long_form = len(audio) > 30 * RATE  # past 30 s whisper needs sequential long-form decoding
        extra = {"truncation": False, "padding": "longest"} if long_form else {}
        inputs = self.processor(audio, sampling_rate=RATE, return_tensors="pt", return_attention_mask=True, **extra)
        inputs = inputs.to(DEVICE, dtype=DTYPE)
        out = self.model.generate(**inputs, language="sv", task="transcribe",
                                  condition_on_prev_tokens=False, return_timestamps=long_form)
        return self.processor.batch_decode(out, skip_special_tokens=True)[0]


class Router:
    def __init__(self):
        self.models = {}
        self.lock = threading.Lock()

    def backend(self, lang):
        name = "kb-whisper" if (lang or "").lower().startswith("sv") else "nemotron"
        if name not in self.models:
            self.models[name] = KBWhisper() if name == "kb-whisper" else Nemotron()
        return name, self.models[name]

    def transcribe(self, audio, rate, lang):
        if rate != RATE:
            n = int(len(audio) * RATE / rate)
            audio = np.interp(np.linspace(0, len(audio), n, endpoint=False), np.arange(len(audio)), audio)
        audio = audio.astype(np.float32)
        # quiet mics hurt accuracy: bring the peak up to ~0.9 (at most 20x gain)
        peak = float(np.abs(audio).max()) if len(audio) else 0.0
        if peak > 1e-4:
            audio = audio * min(0.9 / peak, 20.0)
        with self.lock, torch.inference_mode():
            name, model = self.backend(lang)
            return name, model.transcribe(audio, lang).strip()


def save_recording(pcm, rate, lang, backend, text):
    os.makedirs(RECORDINGS, exist_ok=True)
    stem = os.path.join(RECORDINGS, time.strftime("%Y%m%d-%H%M%S"))
    with wave.open(stem + ".wav", "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    with open(stem + ".json", "w") as f:
        json.dump({"lang": lang, "backend": backend, "text": text}, f, ensure_ascii=False)
    for old in sorted(os.listdir(RECORDINGS))[:-2 * KEEP_RECORDINGS]:
        os.unlink(os.path.join(RECORDINGS, old))


def run(router, pcm, rate, lang):
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    t0 = time.time()
    backend, text = router.transcribe(audio, rate, lang)
    dt = time.time() - t0
    peak = float(np.abs(audio).max()) if len(audio) else 0.0
    log(f"{len(audio) / rate:.1f}s audio (peak {peak:.3f}) {lang}/{backend} -> {dt:.2f}s: {text!r}")
    save_recording(pcm, rate, lang, backend, text)
    return text, dt


def handle(conn, router):
    with conn:
        try:
            f = conn.makefile("rb")
            header = json.loads(f.readline())
            pcm = f.read(header["nbytes"])
            text, dt = run(router, pcm, header.get("rate", RATE), header.get("lang"))
            resp = {"text": text, "seconds": dt}
        except Exception as e:  # report errors to the client instead of dying
            log(f"error: {e!r}")
            resp = {"error": repr(e)}
        conn.sendall((json.dumps(resp) + "\n").encode())


def read_wav(data):
    with wave.open(io.BytesIO(data)) as w:
        if w.getsampwidth() != 2:
            raise ValueError("only 16-bit PCM WAV is supported")
        rate, channels, pcm = w.getframerate(), w.getnchannels(), w.readframes(w.getnframes())
    if channels > 1:
        pcm = np.frombuffer(pcm, dtype=np.int16).reshape(-1, channels).mean(axis=1).astype(np.int16).tobytes()
    return pcm, rate


def http_handler(router, api_key):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if not self.path.split("?")[0].rstrip("/").endswith("/audio/transcriptions"):
                return self.reply(404, {"error": "not found"})
            if api_key and not hmac.compare_digest(self.headers.get("Authorization", ""), f"Bearer {api_key}"):
                return self.reply(401, {"error": "invalid API key"})
            try:
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                msg = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(
                    f"Content-Type: {self.headers.get('Content-Type', '')}\r\n\r\n".encode() + body)
                fields = {part.get_param("name", header="content-disposition"): part.get_payload(decode=True)
                          for part in msg.iter_parts()}
                pcm, rate = read_wav(fields["file"])
            except Exception as e:
                return self.reply(400, {"error": f"bad request: {e!r}"})
            try:
                text, _ = run(router, pcm, rate, (fields.get("language") or b"auto").decode())
            except Exception as e:
                log(f"error: {e!r}")
                return self.reply(500, {"error": repr(e)})
            self.reply(200, {"text": text})

        def log_message(self, fmt, *args):
            log(f"http {self.address_string()} {fmt % args}")

    return Handler


def serve_http(addr, router, api_key):
    host, _, port = addr.rpartition(":")
    host = host or "127.0.0.1"
    srv = ThreadingHTTPServer((host, int(port)), http_handler(router, api_key))
    if host not in ("127.0.0.1", "localhost", "::1") and not api_key:
        log("warning: HTTP is reachable from the network without an API key (set --api-key)")
    log(f"listening on http://{host}:{port}/v1/audio/transcriptions")
    threading.Thread(target=srv.serve_forever, daemon=True).start()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--socket", default=DEFAULT_SOCKET)
    ap.add_argument("--warmup", action="store_true", help="load models and run a dummy transcription at startup")
    ap.add_argument("--http", metavar="[HOST:]PORT",
                    help="also serve an OpenAI-compatible HTTP API, e.g. 0.0.0.0:8000 (default host 127.0.0.1)")
    ap.add_argument("--api-key", default=os.environ.get("VTYPE_API_KEY"),
                    help="require this Bearer token on HTTP requests (default $VTYPE_API_KEY)")
    args = ap.parse_args()

    router = Router()
    if args.warmup:
        for lang in ("en-US", "sv-SE"):
            router.transcribe(np.zeros(RATE, dtype=np.float32), RATE, lang)
        log("warmup done")

    os.makedirs(os.path.dirname(args.socket), exist_ok=True)
    if os.path.exists(args.socket):
        os.unlink(args.socket)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(args.socket)
    os.chmod(args.socket, 0o600)
    srv.listen(8)
    log(f"listening on {args.socket}")
    if args.http:
        serve_http(args.http, router, args.api_key)
    try:
        while True:
            conn, _ = srv.accept()
            threading.Thread(target=handle, args=(conn, router), daemon=True).start()
    except KeyboardInterrupt:
        pass
    finally:
        srv.close()
        os.unlink(args.socket)


if __name__ == "__main__":
    main()
