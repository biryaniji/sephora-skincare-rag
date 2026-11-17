"""Quick voice check (Windows): Windows reads a sentence aloud into a WAV, then voice.py transcribes it.

Run:  python check_voice.py
The first run downloads the Whisper model (~290 MB). Expect it to print roughly the sentence below.
"""
import os
import subprocess
import tempfile

import voice

SENTENCE = "I have dry skin, recommend a face wash under twenty dollars"

wav = os.path.join(tempfile.gettempdir(), "voice_check.wav")
ps = (
    "Add-Type -AssemblyName System.Speech; "
    "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
    "$f = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, 'Sixteen', 'Mono'); "
    f"$s.SetOutputToWaveFile('{wav}', $f); $s.Speak('{SENTENCE}'); $s.Dispose()"
)
subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)

with open(wav, "rb") as fh:
    heard = voice.transcribe(fh.read())
print("SAID :", SENTENCE)
print("HEARD:", heard)
print("OK" if "dry skin" in heard.lower() and "face wash" in heard.lower() else "CHECK THE TRANSCRIPT ABOVE")
