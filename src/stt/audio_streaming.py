import torch
import queue, threading, time, sys
from collections import deque

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel


# ---------- CONSTANTS ---------- #
SAMPLE_RATE = 16000  # in Hz; per second
FRAME_MS = 160  # how far in terms of samples to shift each frame forward by along the audio signal; aka how long each frame lasts in ms
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # how many individual audio samples are in each frame of audio; aka samples/second * seconds
CONTEXT_SECONDS = 1.4  # rolling context length fed to the model each pass; this is the rolling buffer maintained
STEP_MS = 200   # how often transcription is re-run
STABILIZATION_DELAY = 0.4  # commit words that ended at least this long ago; MUST BE SMALLER THAN CONTEXT_SECONDS

MODEL_NAME = "small.en"  # faster-whisper model selection
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"  # defaults for GPU vs CPU, both faster than float32

LANGUAGE = "en"  # not presetting the language adds latency
BEAM_SIZE = 1  # 1 is fastest, increase for accuracay but adds latency


# ---------- FUNCTIONS ---------- #
audio_q = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    '''Callback function for audio chunks in the stream.'''
    # Short because callback needs to finish quickly or audio stream will glitch; just dump into queue and unblock mic

    if status:  # if there are errors
        print(status, file=sys.stderr)  # write the error in the error stream
    
    # Take the mean across channels for each frame; flatten
    # faster-whisper expects type np.float32
    data = indata.copy().mean(axis=1).astype(np.float32)  # indata.shape = (frames, channels)
    # frames = how many samples in this chunk (e.g. FRAME_SAMPLES = 2560)
    # channels = # of audio channels (1 for mono, 2 for stereo etc.)

    audio_q.put(data)  # put the chunk of audio at the end of audio_q

def mic_stream():
    '''Function defining the audio stream'''

    # Open audio stream
    try:
        with sd.InputStream(  # use with instead of stream.start() and stream.stop() to guarentee proper cleanup even with exceptions or errors; also we don't need manual start/stop yet here so this is fine
            channels=1,
            samplerate=SAMPLE_RATE,
            blocksize=FRAME_SAMPLES,
            dtype="float32",
            callback=audio_callback,
            latency="low"
        ):
            while True:  # keep the stream alive
                time.sleep(0.1)
    
    except Exception as e:
        print("Mic stream error:", e, flush=True)

def get_latencies(next_words, emission_time):
    '''Given the list of newly committed words and their emission times, return a list of their latencies'''
    latencies = []

    for (start, end, word) in next_words:
        latencies.append(emission_time - start)
    
    return latencies

def calc_avg(latencies):
    '''Given a list of latencies in seconds, calculates the average latency.'''
    return sum(latencies) / len(latencies) if latencies else 0  # in case of an empty list


if __name__ == '__main__':


# ---------- SETUP ---------- #
    # Transcription model
    try:
        model = WhisperModel(MODEL_NAME,
                            device=DEVICE,
                            compute_type=COMPUTE_TYPE)  # compute type is the type it computes internally, not the type it takes in
    except Exception as e:
        print("Error loading model:", e, flush=True)

    # Start separate microphone thread
    t = threading.Thread(target=mic_stream, 
                        daemon=True)  # daemon = True -> will not keep the program alive if everything else is done; killed when main thread finishes
    # This is fine because even if it is killed "with" will clean stuff up
    t.start()

    # Rolling buffer of the last CONTEXT_SECONDS
    ctx_samples = int(CONTEXT_SECONDS * SAMPLE_RATE)  # number of context audio samples
    ring = deque(maxlen=ctx_samples)  # created double-ended queue that can hold at most ctx_samples; make a ring buffer (oldest items drop off automatically); use deque because faster than queue.Queue

    # Tracking already committed words so we don't repeat them
    committed = []  # list of (start, end, text); start and end are timestamps
    committed_text = ""  # string we've already printed

    # Tracking latencies for calculation at the end
    latencies = []

    last_infer_ts = 0  # timestamp of last inference


# ---------- MAIN ---------- #
    print("\n[ Streaming... speak into your mic. ]\n", flush=True)
    try:
        while True:
            # drain audio queue quickly
            drained = False  # flag
            while not audio_q.empty():
                ring.extend(audio_q.get_nowait())  # gets the queue contents without waiting; check for emptiness prevents exception; good practice in case of edge cases
                drained=True

            now = time.time()  # time since last time inference was run; helps with STEP_MS to run inference only at certain intervals

            # Run inference every STEP_MS if enough audio
            # Ensure we have at least 0.5 s of audio before inferring or bad performance
            if drained and (now - last_infer_ts) * 1000 >= STEP_MS and len(ring) > 0.5 * SAMPLE_RATE:
                last_infer_ts = now

                # Prepare context audio
                ctx_audio = np.frombuffer(np.array(ring, dtype=np.float32).tobytes(), dtype=np.float32)
                # the whole tobytes() and frombuffer() thing is to ensure the array is contiguous in memory, gets rid of gaps
                # overall flattens audio buffer into contiguous float32 array for whisper to safely process

                # See default values for parameters here: https://whisper-api.com/docs/transcription-options
                segments, info = model.transcribe(
                    ctx_audio,
                    language=LANGUAGE,
                    beam_size=BEAM_SIZE,
                    vad_filter=True,   # might introduce latency - experiment with turning this on/off
                    word_timestamps=True,
                    temperature=0.0,  # fastest, reduces hallucination; always picks the most likely
                    compression_ratio_threshold=2.4,  # if the audio segment is too "repetitive" or low quality it skips this chunk; if it exceeds compressed_size/uncompressed_size then skip 
                    log_prob_threshold=-1.0,  # threshold for determining when a word is too unlikely to keep
                    no_speech_threshold=0.6  # if this much probability of no speech, chunk is ignored
                    # probably modify this based on stuttering detection
                )

                # List of words with timestamps - word-level timeline
                curr_words = []
                for seg in segments:
                    if seg.words:  # words have (start, end, word, prob), retrieve the word part of it
                        for w in seg.words:
                            # strip whitespace from each word and convert timestamps to consistent datatype
                            curr_words.append((float(w.start), float(w.end), w.word.strip()))

                # Decide which words are stable enough to commit
                # Commit words that ended earlier than (ctx_duration - STABILIZATION_DELAY)
                ctx_dur = len(ctx_audio) / SAMPLE_RATE  # numsamples / sample rate - gives duration of the audio
                cutoff = ctx_dur - STABILIZATION_DELAY

                # Find index in curr_word where end <= cutoff
                stable_idx = 0
                while stable_idx < len(curr_words) and curr_words[stable_idx][1] <= cutoff:  # curr_words[stable_idx][1] gets the end time
                    stable_idx += 1

                stable_words = curr_words[:stable_idx]

                # Avoid printing the same words twice in streaming transcription
                # Join the words in the list to get a string because not every element of the list might be a full word, e.g. Hello and Hel, lo
                new_text = "".join(w[2] for w in stable_words)
                old_text = "".join(w[2] for w in committed)

                if len(new_text) > len(old_text):  # if there are new stable words ready to be printed
                    next_text = new_text[len(old_text):]

                    if next_text.strip() != "":  # if the next text isn't blank or whitespace
                        # Update lists
                        next_words = stable_words[len(committed):]
                        committed.extend(next_words)
                        committed_text += next_text

                        # Latency tracking
                        emission_time = time.perf_counter()  # in seconds
                        latencies.extend(get_latencies(next_words, emission_time))

                        # Compute latest average latency
                        avg_latency = calc_avg(latencies)

                        # Print to console
                        print(new_text, end="", flush=True)  # print transcription
                        # print(f"\r[ Current latency: {avg_latency:.3f}s ]   ", end="", flush=True)  # print latency real time; overwrites previous latency

            time.sleep(0.01)  # could reduce more for lower latency

    except KeyboardInterrupt:
        print("\n\n[ Stopped. ]")
        print(f"Average latency: {calc_avg(latencies)}s")