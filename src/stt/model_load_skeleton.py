import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from faster_whisper import WhisperModel

try:
    model = WhisperModel("small.en", device="cpu", compute_type="int8")
    print("Model loaded successfully")
except Exception as e:
    print("Error:", e)
