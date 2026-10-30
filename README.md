# vtype

Push-to-talk dictation for [Claude Code](https://claude.com/claude-code) (or any terminal program),
running **fully locally** on your GPU, or against a speech model served at an **external URL**.
Built for WSL2 on Windows.

Press **F9**, speak, press **F9** again, and the transcribed text is typed into Claude's prompt.

- **English and 40 other languages:** [NVIDIA Nemotron 3.5 ASR](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) (0.6B)
- **Swedish:** [KBLab KB-Whisper large](https://huggingface.co/KBLab/kb-whisper-large), trained on Swedish speech by the National Library of Sweden
- **Remote backend:** use any OpenAI-compatible `/v1/audio/transcriptions` server instead, including
  `asr_server.py` itself running on another machine
- Recording indicator in Claude Code's status line
- With the local backend, no audio leaves your machine

```
 ● REC  sv-SE          ← Claude Code status line while recording
```

## How it works

```
 Windows mic ──► WSLg PulseAudio ──► vtype ──(Unix socket)──► asr_server.py ──► GPU
                                       │                         ├─ Nemotron 3.5 ASR  (en-US, auto, …)
                                       │                         └─ KB-Whisper large  (sv-*)
                                       ▼
                            types text into Claude Code

 or, with backend = "remote":

 vtype ──(HTTPS, OpenAI-compatible /v1/audio/transcriptions)──► external server
```

- **`vtype`** runs your command (default `claude`) inside a pseudo-terminal and passes all keystrokes
  through, except the hotkeys. On F9 it records from the microphone; on the second F9 it sends the audio
  to the server and types the returned text into the program, as if you had typed it.
  It uses only the Python standard library: audio is captured from WSLg's PulseAudio server through
  `libpulse-simple` via `ctypes`, so no PortAudio or other system packages are needed.
- **`asr_server.py`** loads the models once and serves transcriptions over
  `~/.cache/vtype/asr.sock`. With the local backend it's started automatically by the first `vtype`
  and shared by all sessions. It's a plain Python server on Hugging Face Transformers + PyTorch.
  With `--http` it also serves an OpenAI-compatible HTTP API, so it can be the remote backend for
  other machines.
- **`statusline.sh`** is a Claude Code status line command that shows vtype's state.

## Requirements

- Windows 11 with WSL2 and WSLg (default on current WSL; provides the microphone to Linux)
- For the local backend: NVIDIA GPU with CUDA support in WSL. Both models use about 5 GB of VRAM; English only is about 1.5 GB.
  It runs on CPU too, just slower.
- Python 3.11+ and `libpulse0` (present on standard Ubuntu WSL images)
- About 5 GB of disk for the models (downloaded on first run to `~/.cache/huggingface`)

With the remote backend, the GPU, the models and `requirements.txt` aren't needed on the client:
`vtype` itself uses only the Python standard library.

## Installation

```bash
git clone https://github.com/jmyhr/vtype.git ~/projects/vtype
cd ~/projects/vtype
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
ln -s "$PWD/vtype" ~/.local/bin/vtype     # any directory on your PATH
```

Choose where the model runs:

```bash
vtype --configure
```

```
Where is the speech recognition model served?
  1) local   on this machine (asr_server.py, started automatically)
  2) remote  an external URL with an OpenAI-compatible /v1/audio/transcriptions endpoint
```

For a remote server you're asked for its URL, an optional API key and the model name. Without a
config file, vtype runs locally. See [Configuration](#configuration) for all settings.

Check that the microphone reaches WSL:

```bash
vtype --test-mic
```

Speak during the 3 seconds. You should see a peak level above roughly 5% and your words in the
transcript. With the local backend, the first run takes a minute while the models download and load.

### Status line indicator (optional)

Add this to `~/.claude/settings.json` (adjust the path to where you cloned the repo):

```json
{
  "statusLine": {
    "type": "command",
    "command": "sh /home/YOU/projects/vtype/statusline.sh",
    "refreshInterval": 1
  }
}
```

The script prints nothing when Claude isn't running under vtype, so it's harmless in normal sessions.
If you already have a status line, call `statusline.sh` from it and append its output.

## Usage

```bash
vtype                       # runs `claude`
vtype --lang sv-SE          # start in Swedish
vtype -- claude --resume    # pass arguments to claude
vtype -- bash               # works with any terminal program
```

| Key | Action |
|-----|--------|
| **F9** | Start recording / stop and transcribe |
| **F10** | Cycle language: en-US → sv-SE → auto |

The text is inserted followed by a space, without pressing Enter, so you can review it or keep
dictating. Use `--submit` to press Enter automatically.

Status line states:

| Shown | Meaning |
|-------|---------|
| `🎙 F9 to talk  en-US` | Idle, current language |
| `● REC` (red) | Recording |
| `⏳ transcribing…` | Waiting for the model |
| red message | Error, e.g. `⚠ mic is silent` |

The same state is also written to the terminal tab title.

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--lang` | `en-US` or `$VTYPE_LANG` | `en-US`, `sv-SE`, `de-DE`, …, or `auto` for language detection |
| `--key` | `f9` | Record key: `f5`–`f12` or `ctrl-space` |
| `--lang-key` | `f10` | Language cycle key |
| `--submit` | off | Press Enter after inserting text |
| `--backend` | `local` | `local` or `remote` |
| `--url` | | Remote server URL; implies `--backend remote` |
| `--model` | `whisper-1` | Model name sent to the remote server |
| `--test-mic` | | Record 3 s, print the backend, level and transcript, exit |
| `--configure` | | Choose local or remote interactively and write the config file |

## Configuration

Settings live in `~/.config/vtype/config.toml` (or `$XDG_CONFIG_HOME/vtype/config.toml`), written by
`vtype --configure` or by hand. Command-line options override environment variables, which override
the file.

```toml
backend = "remote"        # "local" or "remote"
lang = "sv-SE"            # default language
key = "f9"                # record key
lang_key = "f10"          # language cycle key
submit = false            # press Enter after inserting text

[remote]                  # used when backend = "remote"
url = "https://asr.example.com/v1"
api_key = "..."           # optional, sent as "Authorization: Bearer ..."
model = "whisper-1"       # whatever the server calls its model
```

| Setting | Environment variable |
|---------|----------------------|
| `backend` | `VTYPE_BACKEND` |
| `lang` | `VTYPE_LANG` |
| `remote.url` | `VTYPE_URL` |
| `remote.api_key` | `VTYPE_API_KEY` |
| `remote.model` | `VTYPE_MODEL` |

The file is created readable only by you, since it may hold an API key. Switching back to
`backend = "local"` keeps the `[remote]` section, so you can change between them freely.

### Remote servers

The remote backend sends a 16 kHz mono WAV to `<url>/audio/transcriptions` as a multipart form with
`file`, `model` and `language` (`sv-SE` is sent as `sv`; `auto` sends no language), and reads `text`
from the JSON reply. That's the OpenAI transcription API, which many servers implement: vLLM,
speaches (faster-whisper), whisper.cpp's server, OpenAI and Groq. The URL may be the server root
(`http://host:8000`), the API base (`https://host/v1`) or the full endpoint.

The server decides which model is used, so the per-language routing to KB-Whisper only applies
when the remote server is `asr_server.py`.

To serve vtype's own models (Nemotron + KB-Whisper) from a GPU machine:

```bash
VTYPE_API_KEY=some-long-secret .venv/bin/python asr_server.py --warmup --http 0.0.0.0:8000
```

and on the client set `url = "http://gpu-host:8000/v1"` and the same `api_key`. The server speaks
plain HTTP; put it behind a TLS reverse proxy, a VPN such as Tailscale, or an SSH tunnel
(`ssh -L 8000:localhost:8000 gpu-host`, then use `http://localhost:8000/v1`) when it crosses an
untrusted network. Without `--http`'s host part it binds only to `127.0.0.1`.

## The local server

With the local backend, `vtype` starts the server automatically if it isn't running. To manage it yourself:

```bash
.venv/bin/python asr_server.py --warmup     # run in the foreground
tail -f ~/.cache/vtype/server.log           # log of the auto-started server
pkill -f 'bin/python asr_server.py'         # stop it, freeing the GPU memory
```

Language routing: any language starting with `sv` goes to KB-Whisper; everything else goes to
Nemotron. Before transcription, quiet audio is boosted to a peak of about 90% (at most 20×).

The last 50 recordings are kept as `.wav` + `.json` (language, model, transcript) in
`~/.cache/vtype/recordings/` for checking accuracy. Delete the folder whenever you like.

## Troubleshooting

**Nothing is typed / status shows `⚠ mic is silent`.**
Windows isn't passing the microphone to WSL. In Windows Settings → Privacy & security → Microphone,
turn on *Microphone access*, *Let apps access your microphone*, and
*Let desktop apps access your microphone*. Check the default input device under
Settings → System → Sound. Then run `wsl --shutdown` in PowerShell (this closes all WSL windows)
and reopen your terminal.

**Agent doesn't show up in herdr.** herdr identifies agents by the pane's foreground process, which
is vtype. vtype handles this by restarting itself with `HERDR_AGENT=<command>` when it runs in a
herdr pane, so herdr knows `claude` is inside. Sessions started before this change need restarting.

**F9 does nothing.** Make sure Claude was started with `vtype`, not `claude`. Some terminals bind
F-keys; pick another with `--key`.

**`⚠ vtype: remote ASR not reachable` / `HTTP 401`.** Check `url` and `api_key` in
`~/.config/vtype/config.toml`, and run `vtype --test-mic` to see the full error.

**`⚠ vtype: ASR server not reachable`.** Look at `~/.cache/vtype/server.log`. The first start
downloads several GB, which can exceed the 3-minute wait; run `vtype --test-mic` again afterwards.

**Words missing or wrong.** Make sure the right language is selected (F10). Speak a moment after
pressing F9 and pause briefly before stopping. Compare with the saved recordings in
`~/.cache/vtype/recordings/`.

## Limitations

- Text is transcribed when you stop recording, not live while you speak.
- The status line refreshes once a second, so the indicator can lag by up to a second.
- Hotkeys are detected as standard xterm escape sequences, which is what Windows Terminal sends.

## Models and licenses

| Model | Used for | License |
|-------|----------|---------|
| [nvidia/nemotron-3.5-asr-streaming-0.6b](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) | Non-Swedish languages | OpenMDW-1.1 |
| [KBLab/kb-whisper-large](https://huggingface.co/KBLab/kb-whisper-large) | Swedish | Apache-2.0 |

The models are downloaded from Hugging Face on first use and aren't part of this repository.
