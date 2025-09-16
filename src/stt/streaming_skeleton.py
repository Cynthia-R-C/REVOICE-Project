# Basic audio streaming skeleton to work out errors.

import sounddevice as sd
import numpy as np

def audio_callback(indata, frames, time_info, status):
    print("Callback called!", flush=True)

print("Starting test...", flush=True)

with sd.InputStream(channels=1, 
                    samplerate=16000, 
                    callback=audio_callback):
    print("Stream opened!", flush=True)
    input("Press Enter to stop...")  # Keeps main thread alive