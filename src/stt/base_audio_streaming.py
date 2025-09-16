import torch
import queue
import threading
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
FRAME_MS = 160
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000
MODEL_NAME = "small.en"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

audio_q = queue.Queue()

# ---- Audio callback ---- #
def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    # convert to mono float32
    audio_q.put(indata.mean(axis=1).astype(np.float32))

# ---- Mic thread ---- #
def mic_stream():
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        blocksize=FRAME_SAMPLES,
        channels=1,
        dtype="float32",
        callback=audio_callback,
        latency="low"
    ):
        while True:
            sd.sleep(100)  # keep stream alive

# ---- Main ---- #
if __name__ == "__main__":
    model = WhisperModel(MODEL_NAME, device=DEVICE)

    threading.Thread(target=mic_stream, daemon=True).start()

    ring = np.zeros(0, dtype=np.float32)

    committed_words = []          # full transcript (words)
    last_committed_time = 0.0     # ABSOLUTE seconds of last printed word end
    total_samples = 0             # ABSOLUTE sample counter since start
    EPS = 1e-3                    # small tolerance for float comparisons

    print("[Streaming... speak into your mic.]")
    try:
        while True:
            # Drain audio queue and update absolute sample counter
            while not audio_q.empty():
                chunk = audio_q.get_nowait()
                ring = np.concatenate((ring, chunk))
                total_samples += len(chunk)

            # Keep only last 1s in the ring (for inference speed)
            if len(ring) > SAMPLE_RATE:
                ring = ring[-SAMPLE_RATE:]

            if len(ring) > SAMPLE_RATE // 2:  # at least ~0.5s audio
                # Absolute start time of the current ring in seconds
                window_start_time = (total_samples - len(ring)) / SAMPLE_RATE

                segments, _ = model.transcribe(
                    ring,
                    language="en",
                    beam_size=1,
                    vad_filter=True,       # try False if you still miss words
                    word_timestamps=True
                )

                delta_words = []
                for seg in segments:
                    if seg.words:
                        for w in seg.words:
                            # Convert word end time to ABSOLUTE seconds
                            abs_end = window_start_time + float(w.end)
                            if abs_end > last_committed_time + EPS:
                                word = w.word.strip()
                                if word:
                                    committed_words.append(word)
                                    delta_words.append(word)
                                    last_committed_time = abs_end

                if delta_words:
                    # Print only the new words
                    print(" ".join(delta_words), end=" ", flush=True)

    except KeyboardInterrupt:
        print("\n[Stopped.]")
