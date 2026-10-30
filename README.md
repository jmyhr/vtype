# vtype

Push-to-talk dictation for [Claude Code](https://claude.com/claude-code) (or any terminal program),
running **fully locally** on your GPU. Built for WSL2 on Windows.

Press **F9**, speak, press **F9** again, and the transcribed text is typed into Claude's prompt.

- **English and 40 other languages:** [NVIDIA Nemotron 3.5 ASR](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) (0.6B)
- **Swedish:** [KBLab KB-Whisper large](https://huggingface.co/KBLab/kb-whisper-large), trained on Swedish speech by the National Library of Sweden
- Recording indicator in Claude Code's status line
- No audio leaves your machine

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
```

- **`vtype`** runs your command (default `claude`) inside a pseudo-terminal and passes all keystrokes
  through, except the hotkeys. On F9 it records from the microphone; on the second F9 it sends the audio
  to the server and types the returned text into the program, as if you had typed it.
  It uses only the Python standard library: audio is captured from WSLg's PulseAudio server through
  `libpulse-simple` via `ctypes`, so no PortAudio or other system packages are needed.
- **`asr_server.py`** loads the models once and serves transcriptions over
  `~/.cache/vtype/asr.sock`. It's started automatically by the first `vtype` and shared by all
  sessions. It's a plain Python server on Hugging Face Transformers + PyTorch.
- **`statusline.sh`** is a Claude Code status line command that shows vtype's state.

## Requirements

- Windows 11 with WSL2 and WSLg (default on current WSL; provides the microphone to Linux)
- NVIDIA GPU with CUDA support in WSL. Both models use about 5 GB of VRAM; English only is about 1.5 GB.
  It runs on CPU too, just slower.
- Python 3.11+ and `libpulse0` (present on standard Ubuntu WSL images)
- About 5 GB of disk for the models (downloaded on first run to `~/.cache/huggingface`)

## Installation

```bash
git clone https://github.com/jmyhr/vtype.git ~/projects/vtype
cd ~/projects/vtype
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
ln -s "$PWD/vtype" ~/.local/bin/vtype     # any directory on your PATH
```

Check that the microphone reaches WSL:

```bash
vtype --test-mic
```

Speak during the 3 seconds. You should see a peak level above roughly 5% and your words in the
transcript. The first run takes a minute while the models download and load.

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
| `--test-mic` | | Record 3 s, print the level and transcript, exit |

Set `VTYPE_LANG=sv-SE` in your shell profile to make Swedish the default.

## The server

`vtype` starts the server automatically if it isn't running. To manage it yourself:

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

**F9 does nothing.** Make sure Claude was started with `vtype`, not `claude`. Some terminals bind
F-keys; pick another with `--key`.

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
