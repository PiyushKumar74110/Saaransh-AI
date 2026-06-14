import whisper
import os
import requests
import soundfile as sf
import numpy as np

SARVAM_PIECE_SECONDS = 25

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_STT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text-translate"
SARVAM_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v2.5")

_model = None



# Whisper loader (SAFE)
def load_model():
    global _model

    if _model is None:
        print(f"Loading Whisper model: {WHISPER_MODEL} ...")
        _model = whisper.load_model(WHISPER_MODEL)
        print("Whisper model loaded.")

    return _model



# FIXED Whisper transcription (ROOT FIX HERE)
def transcribe_chunk_whisper(chunk_path: str) -> str:
    model = load_model()

    print(f"Whisper processing: {chunk_path}")

    # IMPORTANT: prevent CPU fp16 warning + instability
    result = model.transcribe(
    chunk_path,
    task="transcribe",
    fp16=False,
    language=None,
    condition_on_previous_text=False,
    temperature=0,
    best_of=1,
    beam_size=1,
    no_speech_threshold=0.6,
)

    return result["text"]



# Sarvam API
def _send_to_sarvam(piece_path: str) -> str:
    if not SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY is not set")

    headers = {"api-subscription-key": SARVAM_API_KEY}

    with open(piece_path, "rb") as f:
        files = {"file": (os.path.basename(piece_path), f, "audio/wav")}
        data = {"model": SARVAM_MODEL, "with_diarization": "false"}

        response = requests.post(
            SARVAM_STT_TRANSLATE_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=120,
        )

    if not response.ok:
        print("Sarvam error:", response.text)
        response.raise_for_status()

    data = response.json()
    return data.get("transcript") or data.get("text", "")


# FIXED Sarvam chunk processing
def transcribe_chunk_sarvam(chunk_path: str) -> str:
    audio, sr = sf.read(chunk_path, dtype="float32")

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    piece_size = int(SARVAM_PIECE_SECONDS * sr)

    full_text = ""

    for i, start in enumerate(range(0, len(audio), piece_size)):
        piece = audio[start:start + piece_size]

        # IMPORTANT FIX: PCM_16 WAV (prevents API/decoder issues)
        piece = np.clip(piece, -1.0, 1.0)
        piece = (piece * 32767).astype(np.int16)

        piece_path = f"{chunk_path}_sv_{i}.wav"
        sf.write(piece_path, piece, sr, subtype="PCM_16")

        try:
            print(f"  → Sarvam piece {i + 1}")
            full_text += _send_to_sarvam(piece_path) + " "
        finally:
            if os.path.exists(piece_path):
                os.remove(piece_path)

    return full_text.strip()



# Router
def transcribe_chunk(chunk_path: str, language: str = "english") -> str:
    if language.lower() == "hinglish":
        return transcribe_chunk_sarvam(chunk_path)
    return transcribe_chunk_whisper(chunk_path)



# Full pipeline
def transcribe_all(chunks: list, language: str = "english") -> str:
    full_transcript = ""

    engine = "Sarvam AI" if language.lower() == "hinglish" else "Whisper"
    print(f"Using {engine} for transcription.")

    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk {i + 1}/{len(chunks)}...")

        text = transcribe_chunk(chunk, language=language)
        full_transcript += text + " "

    print("Transcription complete.")
    return full_transcript.strip()