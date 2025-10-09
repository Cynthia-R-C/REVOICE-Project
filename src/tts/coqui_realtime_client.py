# Attempt at basic real time TTS using Coqui

import torch
from TTS.api import TTS
import socket
import threading
import sys
import sounddevice as sd
import soundfile as sf
import numpy as np
import queue

from threading import Lock

# Server connection settings (match those used for the server)
HOST = 'localhost'
PORT = 9000
SAMPLING_RATE = 16000  # same as in whisper_online_server.py
CHANNELS = 1
DTYPE = np.int16
BYTES_PER_SAMPLE = np.dtype(DTYPE).itemsize  # 2 for int16

# Shared buffer for accumulated bytes and a lock
audio_buffer = b''
buffer_lock = Lock()

# Byte list for saving a final audio file
accumulated_bytes = []
accum_lock = Lock()  # For thread safety

def receive_from_server(sock, data_queue):
    '''Thread to receive audio from server.'''
    while True:
        try:
            data = sock.recv(4096)  # (4KB) standard buffer size
            if not data:
                break
            data_queue.put(data)  # Put raw bytes into queue
            print(f"Received {len(data)} bytes of audio data", flush=True)

            # # For debugging: Append to a cumulative file instead of overwriting
            # with open('received_audio.wav', 'ab') as f:
            #     f.write(data)

            # Accumulate the raw bytes
            with accum_lock:
                accumulated_bytes.append(data)
                
        except Exception as e:
            print(f"Error receiving data: {e}")
            break

def audio_callback(outdata, frames, time, status):
    '''Callback for outputting the received audio chunk'''
    global audio_buffer
    needed_bytes = frames * BYTES_PER_SAMPLE * CHANNELS

    with buffer_lock:
        # Pull from queue into buffer until we have enough or queue is empty
        while len(audio_buffer) < needed_bytes and not data_queue.empty():
            audio_buffer += data_queue.get()

        if len(audio_buffer) < needed_bytes:
            # Not enough data: fill with silence
            outdata.fill(0)
            return

        # Extract exactly what's needed
        data = audio_buffer[:needed_bytes]
        audio_buffer = audio_buffer[needed_bytes:]

    # Convert to numpy array for playback
    audio = np.frombuffer(data, dtype=DTYPE).reshape(-1, CHANNELS)
    outdata[:] = audio

def output_audio_stream():
    '''Runs the output audio stream'''
    with sd.OutputStream(samplerate=SAMPLING_RATE, channels=CHANNELS, dtype=DTYPE, callback=audio_callback):
        while True:
            sd.sleep(1000)  # Keep stream alive

# Make audio data queue
data_queue = queue.Queue()

# Set up socket connection
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.connect((HOST, PORT))
    print(f"Connected to server at {HOST}:{PORT}. Listening for audio...")
    sock.send('tts'.encode('utf-8'))  # initial message to server, specifies client type

    # Start receiver thread (updated to pass data_queue)
    receiver_thread = threading.Thread(target=receive_from_server, args=(sock, data_queue))
    receiver_thread.daemon = True
    receiver_thread.start()

    # Start audio output stream
    output_audio_stream()

except Exception as e:
    print(f"Failed to connect: {e}")
    sys.exit(1)

finally:  # on exit or Ctrl+C
    # Save accumulated audio to a WAV file
    with accum_lock:
        if accumulated_bytes:
            all_data = b''.join(accumulated_bytes)
            with sf.SoundFile('final_received_audio.wav', mode='w', samplerate=SAMPLING_RATE, channels=CHANNELS, subtype='PCM_16') as f:
                f.write(np.frombuffer(all_data, dtype=DTYPE).reshape(-1, CHANNELS))
            print("Saved final received audio to final_received_audio.wav")
    sock.close()