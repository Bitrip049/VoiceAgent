import os
import soundfile as sf
import sounddevice as sd
import re  # added: used in chunk_sentences before later import block
# --------------------- Sentence Chunker for Streaming LLM Output ---------------------
SENTENCE_MIN_CHARS = 80
def chunk_sentences(pieces_iter, min_chars=SENTENCE_MIN_CHARS):
    buf = ""
    for piece in pieces_iter:
        if not piece:
            continue
        # Add a space if needed between buffer and new piece
        if buf and not buf.endswith(" ") and not piece.startswith(" "):
            buf += " "
        buf += piece
        while True:
            # Prefer breaking at sentence-ending punctuation, or at semicolons/colons if no full stop found
            m = re.search(r"(.+?[.?!…])(?=\s|$)", buf)
            if not m:
                m = re.search(r"(.+?[;:])(?=\s|$)", buf)
            if not m:
                break
            yield m.group(1).strip()
            buf = buf[m.end():]
        # If buffer is long, break at a natural phrase boundary (comma or long whitespace)
        if len(buf) >= min_chars and not re.search(r"[.?!…;:]\s$", buf):
            # Try to break at last comma before min_chars
            cut = buf.rfind(",")
            if cut == -1 or cut < 30:
                # Fallback: break at last space
                cut = buf.rfind(" ")
            if cut == -1 or cut < 30:
                cut = len(buf)
            yield buf[:cut].strip()
            buf = buf[cut:].lstrip()
    if buf.strip():
        yield buf.strip()

# Helper to play a WAV file
def play_wav(path):
    data, samplerate = sf.read(path, dtype='float32')
    target_rate = 44100
    if samplerate != target_rate:
        import librosa
        if data.ndim == 1:
            data = librosa.resample(data, orig_sr=samplerate, target_sr=target_rate)
        else:
            # For multi-channel, resample each channel
            data = np.stack([librosa.resample(data[:,i], orig_sr=samplerate, target_sr=target_rate) for i in range(data.shape[1])], axis=1)
        samplerate = target_rate
    sd.play(data, samplerate)
    sd.wait()
# --------------------- Helper: float32 to PCM16 ---------------------
import numpy as np
def float_to_pcm16(audio_float32_mono: np.ndarray) -> bytes:
    pcm16 = (np.clip(audio_float32_mono, -1.0, 1.0) * 32767).astype(np.int16)
    return pcm16.tobytes()
# --------------------- Company / System Prompt ---------------------
COMPANY_PROFILE = """
You are the AI Help Desk Assistant for Down And Under Tech Pty Ltd, an Australian IT solutions company.
Speak in a friendly, professional, confident tone. Solve client queries efficiently and clearly.

COMPANY OVERVIEW:
- Name: Down And Under Tech Pty Ltd
- Founder & Director: Bitrip Pathak
- Contact Number: 0406 336 719
- Location: Australia
- Services: Web development, app development, digital marketing, video editing, graphic design,
    social media management, and digital planning/visuals for structural & interior design.
- Structural design work is limited to digital plans/visuals; physical engineering is outsourced.
- Target clients: Businesses in Australia (any size), startups, and individuals needing IT/creative solutions.
- USPs:
    1) End-to-end digital services under one roof
    2) Customised solutions
    3) Professional quality with a personal touch

COMMUNICATION:
- Use clear, simple language for non-technical clients.
- Be proactive: ask clarifying questions when needed.
- Keep responses concise (2–5 sentences).
- If outside scope, politely suggest alternatives.

FAQ HINTS:
- Services → web/app dev, marketing, video, graphics, social, digital planning/visuals.
- Founder → Bitrip Pathak.
- Contact → 0406 → digital only; physical engineering via partners.
- Ongoing support → yes (maintenance & updates).
- Quotes → ask goals, rough scope, timeline, and budget.

TRANSFER PROTOCOL:
If you cannot answer or confirm the client’s request, offer to connect them to the human team and provide the phone number.
Keep answers under ~120 words unless asked for more detail.
"""
try:
    from TTS.api import TTS as CoquiTTS  # Coqui (offline) if available
except Exception as _coqui_imp_err:
    CoquiTTS = None
    print(f"[TTS] Coqui import skipped: {_coqui_imp_err}")

try:
    import edge_tts  # Edge (online, no key) optional fallback
except Exception:
    edge_tts = None

# --------------------- Coqui TTS Worker ---------------------  (clean header)
class TTS_Coqui_Worker:
    """
    Uses Coqui TTS for local, realistic speech synthesis. Saves WAV and plays via sounddevice.
    """
    def __init__(self, model_name="tts_models/en/ljspeech/vits"):
        if CoquiTTS is None:
            raise RuntimeError("Coqui TTS library not available")
        # Try to use GPU if available, else fallback to CPU
        try:
            import torch
            gpu_available = torch.cuda.is_available()
        except ImportError:
            gpu_available = False
        try:
            self.tts = CoquiTTS(model_name=model_name, progress_bar=False, gpu=gpu_available)
        except Exception as e:
            # Common corruption fix hint
            raise RuntimeError(
                "Coqui model load failed. Try deleting the cached folder under %LOCALAPPDATA%/tts or ~/.local/share/tts and rerun. Original error: " + str(e)
            ) from e
        self.is_speaking = threading.Event()
        self._stop = threading.Event()
        self._q = queue.Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def set_tone(self, tone: str):
        pass  # Placeholder for future tone mapping

    def speak(self, text: str):
        t = (text or "").strip()
        if not t:
            return
        print(f"🤖 Agent: {t}")
        self._q.put(t)

    def barge_in(self):
        self.is_speaking.clear()
        try:
            while not self._q.empty():
                self._q.get_nowait()
        except: pass
        sd.stop()

    def shutdown(self):
        self._stop.set()
        self.barge_in()
        try: self._thread.join(timeout=1.5)
        except: pass

    def _worker(self):
        while not self._stop.is_set():
            try:
                text = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            self.is_speaking.set()
            wav_path = os.path.join(tempfile.gettempdir(), f"coqui_tts_{int(time.time()*1000)}.wav")
            try:
                self.tts.tts_to_file(text=text, file_path=wav_path, speed=1.5)
                data, samplerate = sf.read(wav_path, dtype='float32')
                sd.play(data, samplerate)
                sd.wait()
            except Exception as e:
                print("[Coqui TTS] error:", e)
            finally:
                self.is_speaking.clear()
                try: os.remove(wav_path)
                except: pass



# --------------------- Imports ---------------------
import os, sys, time, wave, queue, threading, tempfile, asyncio, re
import numpy as np
import sounddevice as sd
import webrtcvad
import whisper
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --------------------- Config (override via env) ---------------------
SAMPLE_RATE     = int(os.getenv("SAMPLE_RATE", "16000"))
CHANNELS        = 1
FRAME_MS        = 20
VAD_AGGRESSIVE  = int(os.getenv("VAD_AGGRESSIVE", "2"))  # 0..3
MAX_UTTER_MS    = int(os.getenv("MAX_UTTER_MS", "7000"))
SILENCE_END_MS  = int(os.getenv("SILENCE_END_MS", "600"))
MIN_UTTER_MS    = int(os.getenv("MIN_UTTER_MS", "280"))
RMS_GATE        = float(os.getenv("RMS_GATE", "0.007"))  # ignore very quiet noises
WHISPER_SIZE    = os.getenv("WHISPER_SIZE", "base")
WHISPER_DEVICE  = os.getenv("WHISPER_DEVICE", "cuda")      # "cpu" or "cuda"
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "float16" if WHISPER_DEVICE=="cuda" else "int8")
LLM_MODEL       = os.getenv("LLM_MODEL", "llama3.1:8b-instruct-q4_K_M")
OLLAMA_HOST     = os.getenv("OLLAMA_HOST", "http://localhost:11434")


# Down And Under Tech — Duplex Voice Help Desk (Coqui TTS, single-shot, barge-in on speech)
# STT: faster-whisper   LLM: Ollama (Llama 3.1)   VAD: webrtcvad   TTS: Coqui TTS (WAV) + sounddevice

# Requirements:
#   pip install faster-whisper webrtcvad TTS sounddevice vaderSentiment ollama

import sys, time, wave, queue, threading, tempfile, asyncio, re
import numpy as np
import sounddevice as sd
import webrtcvad
# Removed Edge TTS and pygame (no longer needed)
import whisper
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# Removed TTS_Edge_Worker class (now using TTS_Coqui_Worker only)
def save_wav(path, sr, audio_float32_mono):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(float_to_pcm16(audio_float32_mono))

def samples_per_vad_frame() -> int:
    return int(SAMPLE_RATE * (FRAME_MS/1000.0))

# --------------------- Mic with VAD (continuous) ---------------------
class DuplexMic:
    def __init__(self, sr=SAMPLE_RATE, ch=CHANNELS, vad_level=VAD_AGGRESSIVE,
                 is_agent_speaking=lambda: False, on_user_barge=lambda: None, on_vad_status=lambda status: None):
        self.sr = sr; self.ch = ch
        self.vad = webrtcvad.Vad(vad_level)
        self.stream = None
        self._q = queue.Queue()
        self._stop = threading.Event()
        self._barge = threading.Event()
        self.is_agent_speaking = is_agent_speaking
        self.on_user_barge = on_user_barge
        self.on_vad_status = on_vad_status

    def _cb(self, indata, frames, time_info, status):
        if status: print("Audio status:", status, file=sys.stderr)
        self._q.put(indata[:,0].copy())

    def start(self):
        if self.stream: return
        self.stream = sd.InputStream(samplerate=self.sr, channels=self.ch, dtype="float32", callback=self._cb)
        self.stream.start()

    def stop(self):
        self._stop.set()
        try:
            if self.stream:
                self.stream.stop()
                self.stream.close()
        finally:
            self.stream = None

    def signal_barge_in(self):
        self._barge.set()

    def read_utterance(self) -> np.ndarray:
        spf = samples_per_vad_frame()
        voiced = []
        silence = 0
        min_frames = max(1, int(MIN_UTTER_MS/FRAME_MS))
        max_frames = max(1, int(MAX_UTTER_MS/FRAME_MS))
        end_silence = max(1, int(SILENCE_END_MS/FRAME_MS))
        stash = np.zeros(0, dtype=np.float32)
        last_vad_status = None

        while not self._stop.is_set():
            if self._barge.is_set():
                voiced.clear(); silence = 0
                self._barge.clear()

            try:
                chunk = self._q.get(timeout=0.05)
            except queue.Empty:
                if len(voiced) >= min_frames and silence >= end_silence:
                    break
                continue

            stash = np.concatenate([stash, chunk])
            while len(stash) >= spf:
                frame = stash[:spf]; stash = stash[spf:]
                rms = float(np.sqrt(np.mean(np.square(frame))) + 1e-9)
                is_speech = self.vad.is_speech(float_to_pcm16(frame), SAMPLE_RATE)
                vad_status = 'speaking' if is_speech else ('noise' if rms > RMS_GATE else 'silence')
                if vad_status != last_vad_status:
                    self.on_vad_status(vad_status)
                    last_vad_status = vad_status
                if is_speech:
                    # If agent is talking and user starts, trigger barge-in immediately
                    if self.is_agent_speaking():
                        self.on_user_barge()
                        return frame  # quick return to let caller handle
                    voiced.append(frame); silence = 0
                else:
                    if voiced: silence += 1
                if len(voiced) >= max_frames or (len(voiced) >= min_frames and silence >= end_silence):
                    break

            if len(voiced) >= max_frames or (len(voiced) >= min_frames and silence >= end_silence):
                break

        if not voiced: return np.zeros((0,), dtype=np.float32)
        audio = np.concatenate(voiced, axis=0)
        rms = float(np.sqrt(np.mean(np.square(audio))) + 1e-9)
        if rms < RMS_GATE:
            return np.zeros((0,), dtype=np.float32)
        return audio



# --------------------- LLM (Ollama) ---------------------
try:
    from ollama import Client as OllamaClient
except Exception:
    OllamaClient = None
    print("⚠️  pip install ollama (Python client) to connect to your local server.", file=sys.stderr)

class ChatLLM:
    def stream(self, user_text: str):
        if not self.client:
            yield "Local AI not connected. Please start Ollama."
            return
        self.history.append({"role": "user", "content": user_text})
        try:
            stream = self.client.chat(
                model=self.model,
                messages=self.history,
                stream=True,
                options={
                    "temperature": 0.1,
                    "num_ctx": 2048,
                    "num_predict": 120,
                    "top_p": 0.85,
                    "num_gpus": 999,
                }
            )
            buf = ""
            for chunk in stream:
                part = (chunk.get("message") or {}).get("content", "")
                if part:
                    # Add a space if needed between buffer and new part
                    if buf and not buf.endswith(" ") and not part.startswith(" "):
                        buf += " "
                    buf += part
                    yield self._sanitize(part)
            final = self._sanitize(buf)
            self.history.append({"role": "assistant", "content": final})
        except Exception as e:
            yield f"Local model error: {e}"
    def __init__(self, model=LLM_MODEL, system=COMPANY_PROFILE, host=OLLAMA_HOST):
        self.model = model
        self.history = [{"role": "system", "content": system}]
        self.client = None
        if OllamaClient:
            try:
                self.client = OllamaClient(host=host)
                _ = self.client.list()
            except Exception as e:
                print(f"⚠️  Ollama not reachable at {host}. Run 'ollama serve'. Details: {e}", file=sys.stderr)

    def _sanitize(self, text: str) -> str:
        # Remove stray code fences/backticks that sound weird when spoken
        # Also, fix missing spaces between words
        clean = text.replace("```", "").replace("`", "").strip()
        # Insert a space after punctuation if missing (e.g., "today?Are" -> "today? Are")
        clean = re.sub(r"([.?!])([A-Za-z])", r"\1 \2", clean)
        # Collapse multiple spaces
        clean = re.sub(r"\s+", " ", clean)
        return clean

    def stream_response(self, user_text: str):
        """
        Generator that yields each chunk of the LLM response as soon as it arrives.
        """
        if not self.client:
            yield "Local AI not connected. Please start Ollama."
            return
        self.history.append({"role": "user", "content": user_text})
        buf = []
        try:
            stream = self.client.chat(
                model=self.model,
                messages=self.history,
                stream=True,
                options={
                    "temperature": 0.0,
                    "num_ctx": 2048,
                    "num_predict": 512,
                    "top_p": 0.9,
                    "num_gpus": 999,
                }
            )
            for chunk in stream:
                part = (chunk.get("message") or {}).get("content", "")
                if part:
                    sanitized = self._sanitize(part)
                    buf.append(sanitized)
                    yield sanitized
        except Exception as e:
            yield f"Local model error: {e}"
        final = " ".join(buf).strip()
        self.history.append({"role": "assistant", "content": final})

# --------------------- Whisper STT ---------------------
class Transcriber:
    def __init__(self, size=WHISPER_SIZE, device=WHISPER_DEVICE, compute=WHISPER_COMPUTE):
        print(f"[Whisper] (official) Loading {size} on {device}…")
        self.model = whisper.load_model(size, device=device)

    def transcribe(self, wav_path: str) -> str:
        # Official whisper returns a dict with 'text' key
        result = self.model.transcribe(wav_path, language="en")
        return result.get('text', '').strip()

# --------------------- Sentiment → tone ---------------------
def classify_tone(text: str, analyzer: SentimentIntensityAnalyzer):
    if not text.strip(): return "neutral", 0.0
    s = analyzer.polarity_scores(text)["compound"]
    if s <= -0.3: return "negative", s
    if s >=  0.5: return "positive", s
    return "neutral", s

# --------------------- Text Normalization for TTS ---------------------
import re
import num2words  # pip install num2words

CONTRACTIONS = {
    r"\bdo not\b": "don't",
    r"\bcan not\b": "cannot",
    r"\bI am\b": "I'm",
    r"\bwe are\b": "we're",
    r"\bit is\b": "it's",
}

def normalize_numbers(s: str):
    def repl(m):
        n = m.group(0)
        if len(n) >= 4:  # long numbers sound better spoken out
            return " ".join(list(n))
        try:
            return num2words.num2words(int(n))
        except:
            return n
    return re.sub(r"\b\d+\b", repl, s)

def soften_punctuation(s: str):
    # break very long sentences, add soft pauses
    s = re.sub(r"\s*&\s*", " and ", s)
    s = re.sub(r"([,:;])(\S)", r"\1 \2", s)
    s = re.sub(r"([a-z]) — ([a-z])", r"\1 — \2", s, flags=re.I)
    # Removed aggressive split on capitalized words to avoid unnatural pauses in names
    return s

def apply_contractions(s: str):
    for pat, rep in CONTRACTIONS.items():
        s = re.sub(pat, rep, s, flags=re.I)
    return s

def smooth_for_tts(text: str) -> str:
    s = text.strip()
    s = apply_contractions(s)
    s = normalize_numbers(s)
    s = soften_punctuation(s)
    # keep sentences shortish
    s = re.sub(r"\s{2,}", " ", s)
    return s

###########################################################
# Edge TTS Fallback Worker (simple, blocking per utterance)
###########################################################
class TTS_Edge_Fallback:
    """Lightweight fallback if Coqui fails. Requires edge_tts & pydub & soundfile.
    Not streaming: synthesizes whole utterance then plays.
    """
    def __init__(self, voice=os.getenv("EDGE_TTS_VOICE", "en-US-AriaNeural")):
        if edge_tts is None:
            raise RuntimeError("edge-tts not installed; cannot use fallback")
        self.voice = voice
        self.is_speaking = threading.Event()

    def set_tone(self, tone: str):
        pass

    def barge_in(self):
        if self.is_speaking.is_set():
            try: sd.stop()
            except: pass
        self.is_speaking.clear()

    def shutdown(self):
        self.barge_in()

    def speak(self, text: str):
        t = (text or "").strip()
        if not t:
            return
        import tempfile, asyncio
        self.is_speaking.set()
        try:
            mp3_path = os.path.join(tempfile.gettempdir(), f"edge_tts_{int(time.time()*1000)}.mp3")
            async def run():
                comm = edge_tts.Communicate(t, self.voice)
                await comm.save(mp3_path)
            asyncio.run(run())
            try:
                from pydub import AudioSegment
                wav_path = mp3_path + ".wav"
                AudioSegment.from_mp3(mp3_path).export(wav_path, format="wav")
                data, sr = sf.read(wav_path, dtype='float32')
                sd.play(data, sr); sd.wait()
            finally:
                for p in (mp3_path, mp3_path+".wav"):
                    try: os.remove(p)
                    except: pass
        except Exception as e:
            print("[Edge TTS Fallback] error:", e)
        finally:
            self.is_speaking.clear()

# --------------------- Main loop ---------------------
def main():
    backend_pref = os.getenv("TTS_BACKEND", "auto").lower()  # auto|coqui|edge
    active_backend = None
    print("=== Down And Under Tech — Duplex Voice Help Desk (TTS auto-backend) ===")
    print("Talk naturally; the agent will speak a full answer and stop if you start speaking. Ctrl+C to exit.\n")

    # If you need to force output device:
    # print(sd.query_devices())
    # sd.default.device = (sd.default.device[0], <YOUR_OUTPUT_DEVICE_INDEX>)

    # Select backend
    tts = None
    if backend_pref in ("coqui", "auto") and tts is None:
        try:
            tts = TTS_Coqui_Worker()
            active_backend = "coqui"
        except Exception as e:
            print(f"[TTS] Coqui unavailable: {e}")
            if backend_pref == "coqui":
                print("Set TTS_BACKEND=edge to force Edge fallback, or install/fix Coqui.")
    if tts is None and backend_pref in ("edge", "auto"):
        try:
            tts = TTS_Edge_Fallback()
            active_backend = "edge"
        except Exception as e:
            print(f"[TTS] Edge fallback unavailable: {e}")
    if tts is None:
        print("FATAL: No TTS backend available. Install 'TTS' for Coqui or 'edge-tts pydub'. Exiting.")
        return
    print(f"[TTS] Active backend: {active_backend}")
    def vad_status_cb(status):
        print(f"[VAD] {status}")
    mic = DuplexMic(
        is_agent_speaking=lambda: tts.is_speaking.is_set(),
        on_user_barge=lambda: tts.barge_in(),
        on_vad_status=vad_status_cb
    )
    mic.start()

    stt = Transcriber()
    llm = ChatLLM()
    sentiment = SentimentIntensityAnalyzer()

    # Greeting (single-shot)
    tts.barge_in()
    tts.speak("Down And Under Tech voice assistant ready.")
    print("🎙️  Speak… (auto-stop on pause)")

    try:
        while True:
            # 1) capture one utterance
            audio = mic.read_utterance()
            if audio.size == 0:
                continue

            # No barge-in logic for Coqui streaming TTS

            # 2) transcribe
            tmp = os.path.join(tempfile.gettempdir(), f"daut_{int(time.time()*1000)}.wav")
            save_wav(tmp, SAMPLE_RATE, audio)
            user_text = stt.transcribe(tmp)
            try:
                os.remove(tmp)
            except:
                pass
            if not user_text.strip():
                continue

            print(f"🗣️  You: {user_text}")

            # 3) tone (placeholder for future voice mapping)
            tone, _ = classify_tone(user_text, sentiment)
            tts.set_tone(tone)
            tone_hint = {
                "positive": "User sounds positive; keep it friendly and efficient.",
                "neutral":  "Use a clear, professional tone; be concise.",
                "negative": "Acknowledge once, stay calm and solution-focused; offer a human handoff if needed."
            }[tone]

            # Add phone call context to LLM prompt
            phone_context = "This is a phone call. Speak clearly and at a natural pace. If you mention phone numbers, say them digit by digit, e.g., 'zero four zero six ...'."
            prompt = f"[Tone: {tone_hint}] [Context: {phone_context}] User: {user_text}\nAssistant:"


            def format_phone_numbers(text):
                def repl(m):
                    num = m.group(0)
                    return ' '.join(num)
                return re.sub(r'\b\d{8,}\b', repl, text)

            tts.barge_in()   # ensure clean start
            print("🤖 Agent: ", end="", flush=True)
            # Collect the full LLM response at once, joining with spaces to preserve word boundaries
            response = " ".join(chunk for chunk in llm.stream(prompt))
            if response.strip():
                answer_tts = format_phone_numbers(response)
                spoken = smooth_for_tts(answer_tts)
                print(spoken)
                tts.speak(spoken)
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\nExiting…")
    finally:
        try:
            tts.shutdown()
        except:
            pass
        mic.stop()

if __name__ == "__main__":
    main()
