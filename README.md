# Down And Under Tech — Duplex Voice Help Desk

Real‑time, duplex (barge‑in) voice assistant for phone‑style interaction:

Speech In (mic) → VAD (webrtcvad) → Whisper STT → LLM (Ollama local model) → TTS (Coqui or Edge fallback) → Audio Out.

## Features
- Continuous microphone capture with frame‑level VAD (adjustable aggressiveness)
- Utterance segmentation with silence + max duration limits
- Local Whisper (official) transcription
- Local Ollama LLM (default: `llama3.1:8b-instruct-q4_K_M`) with streaming token handling
- Sentiment → simple tone classification stub for future prosody adaptation
- Coqui TTS offline voice synthesis (primary)
- Automatic fallback to Edge TTS if Coqui missing / model corrupt (no key required)
- Barge‑in: user speech interrupts playback immediately
- Phone-number formatting for clearer speech
- Text smoothing (normalization, number spacing, mild punctuation shaping)

## Architecture
```
┌─────────┐   PCM 16k   ┌────────────┐   text   ┌──────────┐   prompt   ┌──────────┐   sentences  ┌────────┐   audio  ┌──────────┐
│ Microph │────────────▶│  VAD + Buf │─────────▶│ Whisper  │──────────▶│  LLM     │────────────▶│  TTS   │──────────▶│ Playback │
└─────────┘              │  (frames)  │          │ (STT)    │           │ (Ollama) │   stream     │(Coqui/ │   WAV/    │ (sounddev│
                          └────────────┘          └──────────┘           └──────────┘              │ Edge)  │  MP3→WAV  │  + barge )
                                                                                                    └────────┘          └──────────┘
```

## Requirements
Python 3.11+ recommended.

Core runtime dependencies (some pinned in `requirements.txt`):
- faster-whisper (if you switch from official whisper; currently using official `whisper` import)
- whisper (openAI official)
- webrtcvad
- sounddevice, soundfile
- numpy
- num2words
- vaderSentiment
- ollama (Python client)
- TTS (Coqui) plus its backend libs
- edge-tts, pydub (fallback online TTS)
- librosa (optional: higher quality resampling)

### Install (clean virtualenv)
```powershell
python -m venv .venv
. .venv/Scripts/Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
# extras used by code but not yet in requirements.txt:
pip install whisper TTS edge-tts pydub soundfile num2words librosa
```
> If you only want offline: skip `edge-tts pydub`.

### Ollama Setup
1. Install Ollama: https://ollama.com/download
2. Start service: it usually runs automatically (`ollama serve`)
3. Pull model:
```powershell
ollama pull llama3.1:8b-instruct-q4_K_M
```
4. (Optional faster model) Try `phi3.5:3.8b-instruct-q4_K_M` for lower latency.

### Environment Variables
| Name | Default | Purpose |
|------|---------|---------|
| `SAMPLE_RATE` | 16000 | Input audio sample rate |
| `VAD_AGGRESSIVE` | 2 | webrtcvad mode 0–3 |
| `MAX_UTTER_MS` | 7000 | Max single utterance length |
| `SILENCE_END_MS` | 600 | Required trailing silence to close utterance |
| `MIN_UTTER_MS` | 280 | Minimum duration to accept |
| `RMS_GATE` | 0.007 | Ignore very quiet frames |
| `WHISPER_SIZE` | base | Whisper model size (tiny/base/small/...) |
| `WHISPER_DEVICE` | cuda | Set to `cpu` if no GPU |
| `LLM_MODEL` | llama3.1:8b-instruct-q4_K_M | Ollama model name |
| `OLLAMA_HOST` | http://localhost:11434 | Ollama server endpoint |
| `TTS_BACKEND` | auto | `auto`/`coqui`/`edge` |
| `EDGE_TTS_VOICE` | en-US-AriaNeural | Fallback voice |

### Running
```powershell
python voice_faq_agent.py
```
You should see:
```
=== Down And Under Tech — Duplex Voice Help Desk (TTS auto-backend) ===
[VAD] speaking
```
Speak after the greeting.

### Common Issues
| Symptom | Cause | Fix |
|---------|-------|-----|
| Coqui ValueError model not found | Corrupted cache | Delete `%LOCALAPPDATA%/tts` or `~/.local/share/tts` and rerun |
| ImportError: TTS.api | `TTS` not installed in venv | `pip install TTS` |
| Edge fallback 403 | Microsoft endpoint temp block | Retry / switch to Coqui |
| High latency LLM | Large model or CPU only | Use quantized 4-bit; smaller model; ensure GPU layers |
| Whisper slow | Using large size | Switch to `base` or `small` |

### Performance Tuning
- Reduce `num_predict` in Ollama options for shorter replies.
- Use smaller quantized models (`q4_K_M` or 3B family).
- Lower VAD `SILENCE_END_MS` to make turns snappier (e.g. 350–450 ms).
- Stream partial tokens directly to TTS (future enhancement).

### Roadmap Ideas
- True streaming TTS (incremental playback)
- Dynamic prosody via sentiment → SSML / style tokens
- Conversation summarization to trim LLM context
- Browser/WebRTC client
- Call transfer logic integration

### License
Add your license choice here (e.g., MIT) before publishing.

### Disclaimer
Edge TTS fallback depends on an undocumented public endpoint and may break or rate limit; prefer Coqui for reliability or replace with Piper / XTTS for fully local operation.

---
Contributions welcome. Open issues or PRs with improvements.
