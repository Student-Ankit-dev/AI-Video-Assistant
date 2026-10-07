import yt_dlp
from pydub import AudioSegment
import os

DOWNLOAD_DIR = "downloads"

os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def download_youtube_audio(url: str) -> str:
    output_path = os.path.join(
        DOWNLOAD_DIR,
        "%(title)s.%(ext)s"
    )

    ydl_opts = {
    "format": "bestaudio/best",
    "outtmpl": output_path,

    "noplaylist": True,

    "extractor_args": {
        "youtube": {
            "player_client": ["android"]
        }
    },

    "http_headers": {
        "User-Agent": (
            "Mozilla/5.0 "
            "(Linux; Android 10; K) "
            "AppleWebKit/537.36 "
            "Chrome/131.0 Mobile Safari/537.36"
        )
    },

    "postprocessors": [
        {
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "192"
        }
    ],

    "quiet": False,
}

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

        print("\n===== INFO =====")
        print("prepare_filename:", ydl.prepare_filename(info))
        print("requested_downloads:", info.get("requested_downloads"))
        print("filepath:", info.get("filepath"))
        print("================\n")
 
        # filename = (
        #     ydl.prepare_filename(info)
        #     .replace(".webm", ".wav")
        #     .replace(".m4a", ".wav")
        # )

        filename = os.path.splitext(ydl.prepare_filename(info))[0] + ".wav"

        print("Returning:", filename)

    return filename


def convert_to_wav(input_path: str) -> str:
    """
    Convert any audio/video file to WAV format
    """

    output_path = os.path.splitext(input_path)[0] + "_converted.wav"

    audio = AudioSegment.from_file(input_path)

    # Whisper works better with mono 16kHz audio
    audio = (
        audio
        .set_channels(1)
        .set_frame_rate(16000)
    )

    audio.export(output_path, format="wav")

    return output_path


def chunk_audio(wav_path: str, chunk_minutes: int = 5) -> list:
    """
    Split audio into smaller chunks for Whisper
    """

    audio = AudioSegment.from_wav(wav_path)

    chunk_ms = chunk_minutes * 60 * 1000

    chunks = []

    for i, start in enumerate(range(0, len(audio), chunk_ms)):

        chunk = audio[start:start + chunk_ms]

        # Skip empty audio chunks
        if len(chunk) == 0:
            print(f"Skipping empty chunk {i}")
            continue

        chunk_path = f"{wav_path}_chunk_{i}.wav"

        chunk.export(chunk_path, format="wav")

        chunks.append(chunk_path)

    return chunks


def process_input(source: str) -> list:

    if source.startswith("http://") or source.startswith("https://"):

        print("Detected Youtube URL, Downloading audio...")

        wav_path = download_youtube_audio(source)

    else:

        print("Detected local file, Converting to WAV...")

        wav_path = convert_to_wav(source)


    print("Chunking audio...")

    chunks = chunk_audio(wav_path)


    print(f"Audio ready - {len(chunks)} chunk(s) created")


    return chunks