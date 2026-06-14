import yt_dlp
import os
import subprocess
import soundfile as sf
import numpy as np

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


# ---------------------------
# 0. sanity check (optional)
# ---------------------------
print(subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout[:150])


# ---------------------------
# 1. Download YouTube audio → WAV (FFmpeg used internally)
# ---------------------------
def download_youtube_audio(url: str) -> str:
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "noplaylist": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)

    wav_path = filename.replace(".webm", ".wav").replace(".m4a", ".wav")
    return wav_path


# ---------------------------
# 2. Convert local file → WAV (ONLY for supported formats)
# ---------------------------
def convert_to_wav(input_path: str) -> str:
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"

    audio, sr = sf.read(input_path, dtype="float32")

    # stereo → mono
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    sf.write(output_path, audio, sr)
    return output_path


# ---------------------------
# 3. Chunk audio using soundfile
# ---------------------------
def chunk_audio(wav_path: str, chunk_minutes: int = 10) -> list:
    audio, sr = sf.read(wav_path, dtype="float32")

    # mono fix
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    chunk_size = chunk_minutes * 60 * sr

    chunks = []

    for i, start in enumerate(range(0, len(audio), chunk_size)):
        chunk = audio[start:start + chunk_size]

        chunk_path = f"{wav_path}_chunk_{i}.wav"
        sf.write(chunk_path, chunk, sr)

        chunks.append(chunk_path)

    return chunks


# ---------------------------
# 4. Main pipeline
# ---------------------------
def process_input(source: str) -> list:
    if source.startswith("http"):
        print("Downloading YouTube audio...")
        wav_path = download_youtube_audio(source)
    else:
        print("Processing local file...")
        wav_path = convert_to_wav(source)

    print("Final WAV path:", wav_path)

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)

    print(f"Done — {len(chunks)} chunks created.")
    return chunks


# ---------------------------
# 5. Run test
# ---------------------------
