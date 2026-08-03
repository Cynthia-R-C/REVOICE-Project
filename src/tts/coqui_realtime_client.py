# real-time TTS client with a playout (jitter) buffer

# class PlayoutBuffer does:
# 1. Hysteresis: playback doesn't resume until a rebuffer threshold is met, so output returns in coherent groups instead of spikes
# 2. Catchup: when buffered backlog exceeds MAX_BACKLOG_SEC, low-energy blocks (silences) are dropped from the buffer until it shrinks to TARGET_BACKLOG_SEC - reclaims latency
# 3. Partial-underrun handling: if buffer holds < one callback's worth, what exists is played (padded with silence) + buffer rearms instead of leaving fragments

# Note: current final_received_audio.wav = untrimmed, catchup trimming in live playback only rn

import socket
import threading
import sys
import soundfile as sf
import numpy as np
import queue
import time

from threading import Lock

# Server connection settings (match those used for the server)
HOST = 'localhost'
PORT = 9000
SAMPLING_RATE = 16000  # same as in whisper_online_server.py
CHANNELS = 1
DTYPE = np.int16
BYTES_PER_SAMPLE = np.dtype(DTYPE).itemsize  # 2 for int16

# Playout buffer tuning
PREBUFFER_SECS = 3.5    # initial fill (deep). MUST match CLIENT_PREBUFFER_SECS in whisper_online_server.py (server estimates buffer depth off this number)
REBUFFER_SECS = 2.0    # for deep refill after underrun
MAX_BACKLOG_SECS = 8.0   # reserve limit for fluent stretches
TARGET_BACKLOG_SECS = 6.0   # trim only when backlog is really really excessive
TRIM_BLOCK_MS = 20
TRIM_RMS_THRESHOLD = 120
MIN_TRIM_RUN_MS = 200
GUARD_BLOCKS = 2

# Saving a final audio file
GROUP = 'convlstm'
FINAL_AUDIO_PATH = f'C:\\Users\\crc24\\Documents\\VS_Code_Python_Folder\\ScienceFair2025\\src\\whisper_streaming\\test_results\\{GROUP}\\final_received_audio.wav'
REALTIME_OUTPUT_PATH = f'C:\\Users\\crc24\\Documents\\VS_Code_Python_Folder\\ScienceFair2025\\src\\whisper_streaming\\test_results\\{GROUP}\\realtime_output.wav'


class PlayoutBuffer:
    '''Playout buffer.

    1. Hysteresis: playback doesn't resume until a rebuffer threshold is met, so output returns in coherent groups instead of spikes
    2. Catchup: when buffered backlog exceeds MAX_BACKLOG_SEC, low-energy blocks (silences) are dropped from the buffer until it shrinks to TARGET_BACKLOG_SEC - reclaims latency
    3. Partial-underrun handling: if buffer holds < one callback's worth, what exists is played (padded with silence) + buffer rearms instead of leaving fragments'''

    BUFFERING = 'buffering'
    PLAYING = 'playing'

    def __init__(self, sr=SAMPLING_RATE, channels=CHANNELS, bytes_per_sample=BYTES_PER_SAMPLE,
                 prebuffer_secs=PREBUFFER_SECS, rebuffer_secs=REBUFFER_SECS,
                 max_backlog_secs=MAX_BACKLOG_SECS, target_backlog_secs=TARGET_BACKLOG_SECS,
                 trim_block_ms=TRIM_BLOCK_MS, trim_rms_threshold=TRIM_RMS_THRESHOLD,
                 verbose=True):
        self._buf = bytearray()
        self._lock = Lock()
        self.state = self.BUFFERING
        self.verbose = verbose

        self._bytes_per_sec = sr * channels * bytes_per_sample
        frame_bytes = channels * bytes_per_sample
        self._prebuffer_bytes = int(prebuffer_secs * self._bytes_per_sec)
        self._rebuffer_bytes = int(rebuffer_secs * self._bytes_per_sec)
        self._max_backlog_bytes = int(max_backlog_secs * self._bytes_per_sec)
        self._target_backlog_bytes = int(target_backlog_secs * self._bytes_per_sec)
        self._trim_block_bytes = max(frame_bytes,
                                     int(trim_block_ms / 1000 * self._bytes_per_sec)
                                     // frame_bytes * frame_bytes)  # frame-aligned
        self._trim_rms_threshold = trim_rms_threshold

        # First start uses the (larger) prebuf thresh, later uses (smaller) rebuff thresh
        self._resume_threshold = self._prebuffer_bytes

        self.total_trimmed_secs = 0.0  # running tot

    def push(self, data: bytes):
        '''Called by the network receiver thread with each received chunk.'''
        with self._lock:
            self._buf += data
            if len(self._buf) > self._max_backlog_bytes:
                self._catch_up_locked()

    def _catch_up_locked(self):
        '''Drop silence from the buffered audio until backlog shrinks to target, but only for sustained silences not isolated ones to prevent cutting consonants'''
        needed_drop = len(self._buf) - self._target_backlog_bytes
        if needed_drop <= 0:
            return

        # Classify blocks as silent/not silent
        blocks = []
        i = 0
        n = len(self._buf)
        while i < n:
            block = self._buf[i:i + self._trim_block_bytes]
            i += len(block)
            silent = False
            if len(block) == self._trim_block_bytes:
                arr = np.frombuffer(bytes(block), dtype=DTYPE)
                rms = float(np.sqrt(np.mean(arr.astype(np.float64) ** 2)))
                silent = rms < self._trim_rms_threshold
            blocks.append((block, silent))

        # Find consecutive silent blocks + mark as droppable (add guard blocks at ends)
        min_run_blocks = max(1, int(MIN_TRIM_RUN_MS / TRIM_BLOCK_MS))
        droppable = [False] * len(blocks)
        run_start = None
        for idx in range(len(blocks) + 1):
            is_silent = idx < len(blocks) and blocks[idx][1]
            if is_silent and run_start is None:
                run_start = idx
            elif not is_silent and run_start is not None:
                run_len = idx - run_start
                if run_len >= min_run_blocks:
                    for j in range(run_start + GUARD_BLOCKS, idx - GUARD_BLOCKS):
                        droppable[j] = True
                run_start = None

        # Drop droppable blocks until target met
        out = bytearray()
        dropped = 0
        for idx, (block, _) in enumerate(blocks):
            if dropped < needed_drop and droppable[idx]:
                dropped += len(block)
                continue
            out += block
        self._buf = out
        if dropped:
            trimmed_secs = dropped / self._bytes_per_sec
            self.total_trimmed_secs += trimmed_secs
            if self.verbose:
                print(f'[PLAYOUT] Catch-up: trimmed {trimmed_secs:.2f}s of sustained silence (total reclaimed this session: {self.total_trimmed_secs:.2f}s)', flush=True)
        elif self.verbose:
            # exceeded target but nothing safe to trim
            print(f'[PLAYOUT] Catch-up wanted to reclaim {needed_drop / self._bytes_per_sec:.2f}s but found no sustained silence to trim, keeping all content.', flush=True)

    def pull(self, needed_bytes: int) -> bytes:
        '''Called by the audio output callback. Plain 1.0x read, returns exactly needed_bytes (pad if necessary)'''
        with self._lock:
            if self.state == self.BUFFERING:
                if len(self._buf) < self._resume_threshold:
                    return b'\x00' * needed_bytes
                self.state = self.PLAYING
                if self.verbose:
                    print(f'[PLAYOUT] Buffer filled ({len(self._buf) / self._bytes_per_sec:.2f}s) - starting playback.', flush=True)

            if len(self._buf) >= needed_bytes:
                data = bytes(self._buf[:needed_bytes])
                del self._buf[:needed_bytes]
                return data

            # Partial underrun
            data = bytes(self._buf)
            self._buf.clear()
            self.state = self.BUFFERING
            self._resume_threshold = self._rebuffer_bytes
            if self.verbose:
                print(f'[PLAYOUT] Underrun - pausing playback until {self._rebuffer_bytes / self._bytes_per_sec:.2f}s is buffered again.', flush=True)
            return data + b'\x00' * (needed_bytes - len(data))

    def buffered_secs(self) -> float:
        with self._lock:
            return len(self._buf) / self._bytes_per_sec


# Final audio bytelist (full + untrimmed, no timing info)
accumulated_bytes = []
accum_lock = Lock()

# Realtime output bytelist
realtime_output_bytes = []
realtime_output_lock = Lock()

playout = PlayoutBuffer()


def receive_from_server(sock):
    '''Thread to receive audio from server.'''
    while True:
        try:
            data = sock.recv(4096)  # (4KB) standard buffer size
            if not data:
                break
            playout.push(data)
            print(f'Received {len(data)} bytes of audio data', flush=True)

            # Accumulate the raw bytes (untrimmed)
            with accum_lock:
                accumulated_bytes.append(data)

        except OSError:
            break
        except Exception as e:
            print(f'Error receiving data: {e}')
            break


def audio_callback(outdata, frames, time_, status):
    '''Callback for outputting the received audio. Thin by design - all buffering policy lives in PlayoutBuffer, no stretching happens here at all anymore'''
    needed_bytes = frames * BYTES_PER_SAMPLE * CHANNELS
    data = playout.pull(needed_bytes)
    outdata[:] = np.frombuffer(data, dtype=DTYPE).reshape(-1, CHANNELS)

    # Record exactly what got sent to the speaker this callback - silence padding
    # (prebuffer/underrun) included, in the exact order and timing it played.
    with realtime_output_lock:
        realtime_output_bytes.append(data)


def output_audio_stream():
    '''Runs the output audio stream'''
    import sounddevice as sd  # imported here so the module stays importable for testing without an audio device
    with sd.OutputStream(samplerate=SAMPLING_RATE, channels=CHANNELS, dtype=DTYPE, callback=audio_callback):
        while True:
            sd.sleep(1000)  # Keep stream alive


if __name__ == '__main__':
    # Set up socket connection
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((HOST, PORT))
        print(f'Connected to server at {HOST}:{PORT}. Listening for audio...')
        sock.send('tts'.encode('utf-8'))  # initial message to server, specifies client type

        receiver_thread = threading.Thread(target=receive_from_server, args=(sock,))
        receiver_thread.daemon = True
        receiver_thread.start()

        # Start audio output stream
        output_audio_stream()

    except Exception as e:
        print(f'Failed to connect: {e}')
        sys.exit(1)

    finally:  # on exit
        # Save to final_received_audio.wav (untrimmed, no timing info)
        with accum_lock:
            if accumulated_bytes:
                all_data = b''.join(accumulated_bytes)
                with sf.SoundFile(FINAL_AUDIO_PATH, mode='w', samplerate=SAMPLING_RATE, channels=CHANNELS, subtype='PCM_16') as f:
                    f.write(np.frombuffer(all_data, dtype=DTYPE).reshape(-1, CHANNELS))
                print('Saved final received audio to final_received_audio.wav')

        # Save to realtime_output.wav - exactly what was heard, gaps and all
        with realtime_output_lock:
            if realtime_output_bytes:
                rt_data = b''.join(realtime_output_bytes)
                with sf.SoundFile(REALTIME_OUTPUT_PATH, mode='w', samplerate=SAMPLING_RATE, channels=CHANNELS, subtype='PCM_16') as f:
                    f.write(np.frombuffer(rt_data, dtype=DTYPE).reshape(-1, CHANNELS))
                dur = len(rt_data) / (SAMPLING_RATE * CHANNELS * BYTES_PER_SAMPLE)
                print(f'Saved real-time output audio ({dur:.1f}s, lags/pauses included) to realtime_output.wav')
        sock.close()