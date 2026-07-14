# whisper STT client
# FILE INPUT VERSION
# Streams a WAV file to the server in real-time-paced chunks instead of reading from mic

# KEY IDEA: server calls conn.recv() on raw PCM16 bytes (Connection.non_blocking_receive_audio() in whisper_online_server.py)
# The only thing that makes mic input behave like real-time streaming is pyaudio's stream.read(CHUNK_SIZE) BLOCKS for roughly CHUNK_SIZE/RATE secs
# This script reproduces the mic's pacing

import argparse
import socket
import threading
import sys
import time
 
import numpy as np
import soundfile as sf
import librosa
 
# Audio settings (match server)
RATE = 16000
CHUNK_SIZE = 4000  # samples per chunk == 0.25s, same as whisper_client.py
CHANNELS = 1
 
# Server connection settings (match server)
HOST = 'localhost'
PORT = 9000
 
 
def load_wav_as_pcm16(path):
    '''Load audio wav as pcm16 bytes
    Uses librosa like server to avoid mismatch'''
    audio, sr = librosa.load(path, sr=RATE, mono=True, dtype=np.float32)
    # float32 [-1, 1] -> int16 PCM, matching finalize_and_send()'s wav*32767 convention
    audio = np.clip(audio, -1.0, 1.0)
    pcm16 = (audio * 32767.0).astype(np.int16)
    return pcm16.tobytes()
 
 
def receive_from_server(sock):
    '''Thread to receive and print transcription from server, identical to whisper_client.py'''
    while True:
        try:
            data = sock.recv(4096)
            if not data:
                break
            print(data.decode('utf-8'), flush=True)
        except OSError:
            # Expected, not an error: happens when the main thread times out waiting
            # for graceful server-side close and force-closes the socket while this
            # thread is still blocked in recv() on it (WinError 10038/10053-class
            # errors on Windows). Harmless - just exit quietly instead of alarming.
            break
 
 
def stream_file(sock, pcm_bytes, chunk_size, rate, realtime=True, speed=1.0):
    '''Send pcm_bytes to sock in chunk_size-sample pieces paced to mimic
    real-time mic input.'''
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
            # Use absolute start time instead of running sum of sleeps so timing errors don't accumulate over time
            target_elapsed = (n_sent * chunk_dur) / speed
            actual_elapsed = time.perf_counter() - start
            remaining = target_elapsed - actual_elapsed
            if remaining > 0:
                time.sleep(remaining)
 
    if realtime:
        print(f'Finished streaming {n_sent} chunks '
              f'({n_sent * chunk_dur / speed:.2f}s target duration).')
    else:
        print(f'Finished streaming {n_sent} chunks as fast as possible (non-realtime mode).')
 
 
def main():
    parser = argparse.ArgumentParser(
        description='Stream a WAV file to the whisper_online_server.py STT socket, '
                    'paced to simulate live mic input.')
    parser.add_argument('audio_path', type=str, help='Path to input audio file (any format librosa can read).')
    parser.add_argument('--host', type=str, default=HOST)
    parser.add_argument('--port', type=int, default=PORT)
    parser.add_argument('--chunk-size', type=int, default=CHUNK_SIZE,
                        help='Samples per chunk sent to the server (default 4000, matches whisper_client.py).')
    parser.add_argument('--speed', type=float, default=1.0,
                        help='Playback speed multiplier. 1.0 = real-time (default). '
                             '2.0 sends audio twice as fast as real-time (useful for quick regression runs ')
    parser.add_argument('--no-realtime', action='store_true',
                        help='Disable pacing entirely and send as fast as the socket allows. ')
    parser.add_argument('--wait-timeout', type=float, default=180.0,
                        help='Seconds to wait after sending for the server to finish processing and close its end (default 180s)')
    args = parser.parse_args()
 
    print(f'Loading {args.audio_path}...')
    pcm_bytes = load_wav_as_pcm16(args.audio_path)
    duration = len(pcm_bytes) / 2 / RATE
    pcm_arr = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32767.0
    peak = np.abs(pcm_arr).max() if len(pcm_arr) else 0.0
    rms = np.sqrt(np.mean(pcm_arr ** 2)) if len(pcm_arr) else 0.0
    print(f'Loaded {duration:.2f}s of audio at {RATE}Hz mono. '
          f'Peak amplitude: {peak:.4f}, RMS: {rms:.4f}')
    if peak < 0.02:
        print("WARNING: peak amplitude is very low, this audio may be near-silent "
              "after loading/downmixing. Play it back locally to confirm it's audible "
              "before assuming the pipeline is at fault.")
 
    # Playback verification
    debug_dump_path = args.audio_path + '.streamed_debug.wav'
    sf.write(debug_dump_path, pcm_arr, RATE, subtype='PCM_16')
    print(f'Wrote exact pre-stream audio to {debug_dump_path} for playback verification.')
 
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((args.host, args.port))
        print(f'Connected to server at {args.host}:{args.port}.')
        sock.send('stt'.encode('utf-8'))  # same handshake as whisper_client.py
 
        # Give server time to accept conn, spawn client thread, and finish conn.recv handshake before sending audio
        time.sleep(0.3)
    except Exception as e:
        print(f'Failed to connect: {e}')
        sys.exit(1)
 
    receiver_thread = threading.Thread(target=receive_from_server, args=(sock,))
    receiver_thread.daemon = True
    receiver_thread.start()
 
    try:
        stream_file(sock, pcm_bytes, args.chunk_size, RATE,
                    realtime=not args.no_realtime, speed=args.speed)
 
        # Half close to signal end of file w/o closing read side; wait for server to finish and close its end
        print('Done sending audio. Half-closing write side, waiting for server '
              'to finish and close its end...')
        try:
            sock.shutdown(socket.SHUT_WR)
        except OSError:
            pass  # already closed on the other end
 
        # Wait for the receiver thread to exit (once server closes its end)
        receiver_thread.join(timeout=args.wait_timeout)
        if receiver_thread.is_alive():
            print(f'Server did not close its end within {args.wait_timeout:.0f}s, closing anyway.')
            print('Does NOT necessarily mean the server is stuck, may still be finishing (trailing-silence re-transcription, destutter flush, TTS/RVC drain etc)')
            print('If server is still actively logging in its terminal, rerun with larger wait timeout.')
    except KeyboardInterrupt:
        print('\nStopping...')
    finally:
        sock.close()
 
 
if __name__ == '__main__':
    main()