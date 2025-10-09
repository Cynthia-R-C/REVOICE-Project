import sounddevice as sd
print(sd.query_devices())
try:
    sd.check_output_settings(device=None, channels=1, dtype='int16', samplerate=16000)
    print("16000 Hz supported.")
except Exception as e:
    print(f"16000 Hz not supported: {e}")