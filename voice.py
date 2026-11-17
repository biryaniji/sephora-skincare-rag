"""Voice input: turn a recording from the chat box's mic button into text.

Uses a Whisper model that runs on this laptop (no audio is sent anywhere). Models are downloaded from
Hugging Face the first time they're needed, then cached: English-only base (~290 MB) for English, and the
multilingual small model (~970 MB) when the answer language involves Hindi.
"""
import io
import wave

import numpy as np

ENGLISH_MODEL = "openai/whisper-base.en"   # ~290 MB, English only, fastest
MULTILINGUAL_MODEL = "openai/whisper-small"  # ~970 MB, understands Hindi and 90+ languages
SAMPLE_RATE = 16000

_pipes: dict[str, object] = {}


def wav_to_array(data: bytes) -> np.ndarray:
    """Decode WAV bytes to mono float32 at 16 kHz (the format Whisper expects)."""
    with wave.open(io.BytesIO(data), "rb") as w:
        rate, channels, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        frames = w.readframes(w.getnframes())
    if width == 2:
        audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 1:
        audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 4:
        audio = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported audio format ({width * 8}-bit)")
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if rate != SAMPLE_RATE and len(audio):
        n = int(len(audio) * SAMPLE_RATE / rate)
        audio = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32)
    return audio


def _load(model: str):
    if model not in _pipes:
        from transformers import pipeline
        _pipes[model] = pipeline("automatic-speech-recognition", model=model)
    return _pipes[model]


def transcribe(data: bytes, language: str | None = "en") -> str:
    """Return the spoken words as plain text ('' if nothing was heard).
    language: "en" (English model), "hi" (Hindi, written in Devanagari) or None (the multilingual model
    detects the language itself, so mixed Hindi/English speech works)."""
    audio = wav_to_array(data)
    if len(audio) < SAMPLE_RATE // 4 or float(np.abs(audio).max()) < 0.003:  # under 0.25 s, or silence
        return ""
    if language == "en":
        out = _load(ENGLISH_MODEL)({"raw": audio, "sampling_rate": SAMPLE_RATE})
    else:
        kwargs = {"task": "transcribe"} | ({"language": language} if language else {})
        out = _load(MULTILINGUAL_MODEL)({"raw": audio, "sampling_rate": SAMPLE_RATE}, generate_kwargs=kwargs)
    return " ".join(out["text"].split()).strip()
