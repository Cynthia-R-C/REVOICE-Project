# Quick test script for CoquiTTS transcription using basic code given on Coqui Github

import torch
from TTS.api import TTS
import os

# Print working directory
print("Working dir:", os.getcwd())

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# List available 🐸TTS models
print(TTS().list_models())

TEXT_PATH = "../whisper_streaming/mrs_dalloway.txt"
OUTPUT_PATH = "output/out.wav"

# Initialize TTS with the target model name
tts = TTS("tts_models/en/ljspeech/fast_pitch").to(device)

# Extract transcription text from file
with open(TEXT_PATH, 'r') as f:
    txt = f.read()

# Run TTS
tts.tts_to_file(text=txt, file_path=OUTPUT_PATH)

print("\ncoqui_test.py executed properly.")