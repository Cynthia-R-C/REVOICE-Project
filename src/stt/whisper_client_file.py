# whisper STT client — FILE INPUT VERSION
# Streams a WAV file to the server in real-time-paced chunks, instead of reading from the mic. This is a drop-in substitute for whisper_client.py for offline/repeatable testing.

# KEY IDEA: the server has no notion of "mic" vs "file" — it just calls conn.recv() on raw PCM16 bytes (see Connection.non_blocking_receive_audio() in whisper_online_server.py). The only thing that makes mic input behave like real-time streaming is that pyaudio's stream.read(CHUNK_SIZE) BLOCKS for roughly CHUNK_SIZE/RATE seconds. 
# Reading a file has no such blocking, so if you sendall() the whole file at once, the server receives it as one burst which breaks every timing-dependent part of the pipeline (e.g. destutter window/hop timing, min_chunk gating TTS_MAX_WAIT_SEC / SILENCE_TIMEOUT logic, latency stats, backpressure checks, etc)
# This script reproduces the mic's pacing

import argparse
import socket
import threading
import sys
import time
 
import numpy as np
import soundfile as sf
import librosa
 
# Audio settings (must match server's expectations: 16kHz, mono, 16-bit signed integer)
RATE = 16000
CHUNK_SIZE = 4000  # samples per chunk == 0.25s, same as whisper_client.py
CHANNELS = 1
 
# Server connection settings (match those used for the server)
HOST = 'localhost'
PORT = 9000
 
 
def load_wav_as_pcm16(path):
    """Load an audio file, force it to 16kHz mono, and return raw int16 PCM bytes.
    Using librosa.load mirrors exactly what the server does when it decodes
    incoming audio (librosa.load(..., sr=SAMPLING_RATE)), so there's no
    resampling mismatch between what you feed in and what the server assumes."""
    audio, sr = librosa.load(path, sr=RATE, mono=True, dtype=np.float32)
    # float32 [-1, 1] -> int16 PCM, matching finalize_and_send()'s wav*32767 convention
    audio = np.clip(audio, -1.0, 1.0)
    pcm16 = (audio * 32767.0).astype(np.int16)
    return pcm16.tobytes()
 
 
def receive_from_server(sock):
    '''Thread to receive and print transcription from server. Identical to whisper_client.py.'''
    while True:
        try:
            data = sock.recv(4096)
            if not data:
                break
            print(data.decode('utf-8'), flush=True)
        except Exception as e:
            print(f"Error receiving data: {e}")
            break
 
 
def stream_file(sock, pcm_bytes, chunk_size, rate, realtime=True, speed=1.0):
    """Send pcm_bytes to sock in chunk_size-sample pieces, paced to mimic
    real-time mic input. Uses a drift-corrected clock (schedule against an
    absolute start time, not a running sum of sleeps) so timing errors from
    sleep() and I/O jitter don't accumulate over a long file."""
    bytes_per_chunk = chunk_size * 2  # int16 = 2 bytes/sample
    chunk_dur = chunk_size / rate     # seconds of audio per chunk
 
    start = time.perf_counter()
    n_sent = 0
    total_chunks = -(-len(pcm_bytes) // bytes_per_chunk)  # ceil div
 
    for i in range(total_chunks):
        chunk = pcm_bytes[i * bytes_per_chunk:(i + 1) * bytes_per_chunk]
        if not chunk:
            break
 
        sock.sendall(chunk)
        n_sent += 1
 
        if realtime:
            # Target wall-clock time this chunk should have finished sending,
            # relative to start. Sleeping to an absolute target (rather than
            # sleeping chunk_dur every iteration) prevents drift from
            # accumulating across thousands of chunks.
            target_elapsed = (n_sent * chunk_dur) / speed
            actual_elapsed = time.perf_counter() - start
            remaining = target_elapsed - actual_elapsed
            if remaining > 0:
                time.sleep(remaining)
 
    if realtime:
        print(f"Finished streaming {n_sent} chunks "
              f"({n_sent * chunk_dur / speed:.2f}s target duration).")
    else:
        print(f"Finished streaming {n_sent} chunks as fast as possible (non-realtime mode).")
 
 
def main():
    parser = argparse.ArgumentParser(
        description="Stream a WAV file to the whisper_online_server.py STT socket, "
                    "paced to simulate live mic input.")
    parser.add_argument('audio_path', type=str, help="Path to input audio file (any format librosa can read).")
    parser.add_argument('--host', type=str, default=HOST)
    parser.add_argument('--port', type=int, default=PORT)
    parser.add_argument('--chunk-size', type=int, default=CHUNK_SIZE,
                        help="Samples per chunk sent to the server (default 4000, matches whisper_client.py).")
    parser.add_argument('--speed', type=float, default=1.0,
                        help="Playback speed multiplier. 1.0 = real-time (default). "
                             "2.0 sends audio twice as fast as real-time (useful for quick regression runs "
                             "while still exercising the streaming/timing logic, just compressed).")
    parser.add_argument('--no-realtime', action='store_true',
                        help="Disable pacing entirely and send as fast as the socket allows. "
                             "NOTE: this defeats the purpose of streaming simulation — timing-dependent "
                             "logic (destutter hop windows, TTS grouping waits, latency stats) will not "
                             "behave as it would with live audio. Only use this to sanity-check that data "
                             "flows through the pipeline at all, not to validate timing behavior.")
    args = parser.parse_args()
 
    print(f"Loading {args.audio_path}...")
    pcm_bytes = load_wav_as_pcm16(args.audio_path)
    duration = len(pcm_bytes) / 2 / RATE
    pcm_arr = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32767.0
    peak = np.abs(pcm_arr).max() if len(pcm_arr) else 0.0
    rms = np.sqrt(np.mean(pcm_arr ** 2)) if len(pcm_arr) else 0.0
    print(f"Loaded {duration:.2f}s of audio at {RATE}Hz mono. "
          f"Peak amplitude: {peak:.4f}, RMS: {rms:.4f}")
    if peak < 0.02:
        print("WARNING: peak amplitude is very low — this audio may be near-silent "
              "after loading/downmixing. Play it back locally to confirm it's audible "
              "before assuming the pipeline is at fault.")
 
    # Dump exactly what will be streamed, byte-for-byte, so you can play back precisely
    # what the server receives — this rules out load_wav_as_pcm16() itself as the culprit.
    debug_dump_path = args.audio_path + '.streamed_debug.wav'
    sf.write(debug_dump_path, pcm_arr, RATE, subtype='PCM_16')
    print(f"Wrote exact pre-stream audio to {debug_dump_path} for playback verification.")
 
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((args.host, args.port))
        print(f"Connected to server at {args.host}:{args.port}.")
        sock.send('stt'.encode('utf-8'))  # same handshake as whisper_client.py
 
        # IMPORTANT: give the server time to accept() the connection, spawn its
        # client thread, and finish its single conn.recv(1024) handshake read
        # BEFORE we send any audio. With the mic client, pyaudio's blocking
        # stream.read(CHUNK_SIZE) naturally creates a ~250ms gap here (waiting
        # for hardware to fill a buffer), which is why the mic path never hits
        # this. A file is already fully loaded in memory, so without this
        # delay the first audio sendall() can follow the handshake send() by
        # microseconds — on localhost, TCP can coalesce both into one receive
        # buffer, and the server's recv(1024) will swallow up to 1024 bytes of
        # real PCM audio along with "stt", silently dropping it before
        # _audio_receive_loop ever sees it.
        time.sleep(0.3)
    except Exception as e:
        print(f"Failed to connect: {e}")
        sys.exit(1)
 
    receiver_thread = threading.Thread(target=receive_from_server, args=(sock,))
    receiver_thread.daemon = True
    receiver_thread.start()
 
    try:
        stream_file(sock, pcm_bytes, args.chunk_size, RATE,
                    realtime=not args.no_realtime, speed=args.speed)
 
        # Graceful half-close instead of a blind sleep + close.
        # SHUT_WR tells the server "I'm done sending" without tearing down our
        # read side. On the server, non_blocking_receive_audio() then gets b''
        # (clean EOF) instead of a hard reset — this is what lets
        # _audio_receive_loop() set _receive_thread_running = False cleanly,
        # which cascades through _destutter_loop()'s sentinel, process()'s
        # final flush_tts_group(), and handle_stt_client()'s post-process
        # cleanup (WAV save, WER calc, stats report) without an exception.
        # A fixed sleep() before a full close() is just guessing how long that
        # takes — for a long file with a busy TTS/RVC queue it can easily be
        # too short, which is what produced the ConnectionResetError.
        print("Done sending audio. Half-closing write side, waiting for server "
              "to finish and close its end...")
        try:
            sock.shutdown(socket.SHUT_WR)
        except OSError:
            pass  # already closed on the other end
 
        # Wait for the receiver thread to exit, which happens once the server
        # closes its side too (recv() returns b''). Bounded wait so a server
        # bug can't hang this script forever.
        receiver_thread.join(timeout=30.0)
        if receiver_thread.is_alive():
            print("Server didn't close its end within 30s — closing anyway.")
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        sock.close()
 
 
if __name__ == '__main__':
    main()