# Real-time TTS client with proper playout (jitter) buffer, replacing prev oneshot prebuffer design

# class PlayoutBuffer adds:
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
from scipy import signal  # for time-stretch resampling in adaptive playback speed

from threading import Lock

# Server connection settings (match those used for the server)
HOST = 'localhost'
PORT = 9000
SAMPLING_RATE = 16000  # same as in whisper_online_server.py
CHANNELS = 1
DTYPE = np.int16
BYTES_PER_SAMPLE = np.dtype(DTYPE).itemsize  # 2 for int16

# Playout buffer tuning
# Deep prebuffer, stores fluent audio during easy stretches so it can output it when stutter gap from rate mismatches appear
# startup latency ^ but smoothness also ^
PREBUFFER_SECS = 3.5    # initial fill (deep)
REBUFFER_SECS = 2.0    # for deep refill after underrun
MAX_BACKLOG_SECS = 8.0   # reserve limit for fluent stretches
TARGET_BACKLOG_SECS = 6.0   # trim only when backlog is really really excessive
TRIM_BLOCK_MS = 20
TRIM_RMS_THRESHOLD = 120
MIN_TRIM_RUN_MS = 200
GUARD_BLOCKS = 2

# Adaptive playback speed, slows down when buffer is low
SPEED_MIN = 0.72  # slowest playback (buffer low)
SPEED_MAX = 0.92  # fastest playback (buffer norm)
SPEED_LOW_WATER_SECS = 1.5  # thresh below which to play at min speed
SPEED_HIGH_WATER_SECS = 5.0  # thresh above which to play at max speed

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

        # First start uses the(larger prebuf thresh, later uses smaller rebuff thresh
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
                print(f"[PLAYOUT] Catch-up: trimmed {trimmed_secs:.2f}s of sustained silence "
                      f"(total reclaimed this session: {self.total_trimmed_secs:.2f}s)",
                      flush=True)
        elif self.verbose:
            # If exceeds target but can't find anything valid to trim
            print(f"[PLAYOUT] Catch-up wanted to reclaim {needed_drop / self._bytes_per_sec:.2f}s "
                  f"but found no sustained silence to trim - keeping all content.",
                  flush=True)

    def pull(self, needed_bytes: int) -> bytes:
        '''Called by the aud output callback. Returns exactly needed_bytes (pad if necessary)'''
        with self._lock:
            if self.state == self.BUFFERING:
                if len(self._buf) < self._resume_threshold:
                    return b'\x00' * needed_bytes
                self.state = self.PLAYING
                if self.verbose:
                    print(f"[PLAYOUT] Buffer filled ({len(self._buf) / self._bytes_per_sec:.2f}s) "
                          f"- starting playback.", flush=True)

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
                print(f"[PLAYOUT] Underrun - pausing playback until "
                      f"{self._rebuffer_bytes / self._bytes_per_sec:.2f}s is buffered again.",
                      flush=True)
            return data + b'\x00' * (needed_bytes - len(data))

    def buffered_secs(self) -> float:
        with self._lock:
            return len(self._buf) / self._bytes_per_sec

    def _current_speed(self, buffered_bytes):
        '''Calc playback speed given the current buffer depth'''
        low = SPEED_LOW_WATER_SECS * self._bytes_per_sec
        high = SPEED_HIGH_WATER_SECS * self._bytes_per_sec
        if buffered_bytes <= low:
            return SPEED_MIN
        if buffered_bytes >= high:
            return SPEED_MAX
        frac = (buffered_bytes - low) / (high - low)
        return SPEED_MIN + frac * (SPEED_MAX - SPEED_MIN)

    def pull_stretched(self, needed_bytes: int) -> bytes:
        '''Stretch out bytes of audio based on how drained the buffer is, aka to the buffer-draining speed. (Makes sure not to stretch silence.)'''
        with self._lock:
            # Buffering/underrun handling
            if self.state == self.BUFFERING:
                if len(self._buf) < self._resume_threshold:
                    return b'\x00' * needed_bytes
                self.state = self.PLAYING
                if self.verbose:
                    print(f"[PLAYOUT] Buffer filled ({len(self._buf) / self._bytes_per_sec:.2f}s) "
                          f"- starting playback.", flush=True)

            speed = self._current_speed(len(self._buf))
            frame_bytes = BYTES_PER_SAMPLE * CHANNELS
            out_frames = needed_bytes // frame_bytes
            src_frames_wanted = int(round(out_frames * speed))
            src_bytes_wanted = src_frames_wanted * frame_bytes

            if len(self._buf) >= src_bytes_wanted and src_frames_wanted > 0:
                src = bytes(self._buf[:src_bytes_wanted])
                del self._buf[:src_bytes_wanted]
            else:
                # Partial underrun: take what's left, then re-arm
                src = bytes(self._buf)
                self._buf.clear()
                self.state = self.BUFFERING
                self._resume_threshold = self._rebuffer_bytes
                if self.verbose:
                    print(f"[PLAYOUT] Underrun - pausing playback until "
                          f"{self._rebuffer_bytes / self._bytes_per_sec:.2f}s is buffered again.",
                          flush=True)
                # stretch whatever we got up to the output size, pad remainder
                if len(src) >= frame_bytes:
                    stretched = self._resample_to_frames(src, out_frames)
                    return stretched + b'\x00' * (needed_bytes - len(stretched))
                return src + b'\x00' * (needed_bytes - len(src))

        # Resample outside the lock (CPU work; buffer already updated)
        if abs(speed - 1.0) < 1e-3:
            return src  # no stretch needed
        return self._resample_to_frames(src, out_frames)

    @staticmethod
    def _resample_to_frames(src_bytes, out_frames):
        '''Resample src_bytes (int16 mono) to exactly out_frames samples via
        polyphase filtering (scipy.signal.resample_poly - high quality, no pitch
        artifacts beyond the intended time-stretch).'''
        arr = np.frombuffer(src_bytes, dtype=DTYPE)
        in_frames = len(arr)
        if in_frames == 0 or out_frames == 0:
            return b'\x00' * (out_frames * BYTES_PER_SAMPLE * CHANNELS)
        if in_frames == out_frames:
            return src_bytes
        # resample_poly needs integer up/down; derive from the frame ratio
        g = np.gcd(in_frames, out_frames)
        up = out_frames // g
        down = in_frames // g
        stretched = signal.resample_poly(arr.astype(np.float32), up, down)
        stretched = np.clip(np.round(stretched), -32768, 32767).astype(DTYPE)
        # length can be off by 1-2 samples from rounding; pad/trim to exact
        if len(stretched) < out_frames:
            stretched = np.concatenate([stretched, np.zeros(out_frames - len(stretched), dtype=DTYPE)])
        else:
            stretched = stretched[:out_frames]
        return stretched.tobytes()

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
            print(f"Received {len(data)} bytes of audio data", flush=True)

            # Accumulate the raw bytes (untrimmed)
            with accum_lock:
                accumulated_bytes.append(data)

        except OSError:
            break
        except Exception as e:
            print(f"Error receiving data: {e}")
            break


def audio_callback(outdata, frames, time_, status):
    '''Callback for outputting the received audio. Thin by design - all buffering
    and adaptive-speed policy lives in PlayoutBuffer.'''
    needed_bytes = frames * BYTES_PER_SAMPLE * CHANNELS
    data = playout.pull_stretched(needed_bytes)
    # pull_stretched guarantees exactly needed_bytes, but guard defensively in case
    # a resampling rounding edge ever returns off-by-a-frame
    if len(data) != needed_bytes:
        if len(data) < needed_bytes:
            data = data + b'\x00' * (needed_bytes - len(data))
        else:
            data = data[:needed_bytes]
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
        print(f"Connected to server at {HOST}:{PORT}. Listening for audio...")
        sock.send('tts'.encode('utf-8'))  # initial message to server, specifies client type

        receiver_thread = threading.Thread(target=receive_from_server, args=(sock,))
        receiver_thread.daemon = True
        receiver_thread.start()

        # Start audio output stream
        output_audio_stream()

    except Exception as e:
        print(f"Failed to connect: {e}")
        sys.exit(1)

    finally:  # on exit
        # Save to final_received_audio.wav (untrimmed, no timing info)
        with accum_lock:
            if accumulated_bytes:
                all_data = b''.join(accumulated_bytes)
                with sf.SoundFile(FINAL_AUDIO_PATH, mode='w', samplerate=SAMPLING_RATE, channels=CHANNELS, subtype='PCM_16') as f:
                    f.write(np.frombuffer(all_data, dtype=DTYPE).reshape(-1, CHANNELS))
                print("Saved final received audio to final_received_audio.wav")

        # Save to realtime_output.wav - exactly what was heard, gaps and all
        with realtime_output_lock:
            if realtime_output_bytes:
                rt_data = b''.join(realtime_output_bytes)
                with sf.SoundFile(REALTIME_OUTPUT_PATH, mode='w', samplerate=SAMPLING_RATE, channels=CHANNELS, subtype='PCM_16') as f:
                    f.write(np.frombuffer(rt_data, dtype=DTYPE).reshape(-1, CHANNELS))
                dur = len(rt_data) / (SAMPLING_RATE * CHANNELS * BYTES_PER_SAMPLE)
                print(f"Saved real-time output audio ({dur:.1f}s, lags/pauses included) to realtime_output.wav")
        sock.close()