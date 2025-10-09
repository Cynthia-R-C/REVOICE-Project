# Attempt at basic real time TTS using Coqui

import torch
from TTS.api import TTS
#import os
import socket
import threading
import sys
import sounddevice as sd
import numpy as np
import queue

# Server connection settings (match those used for the server)
HOST = 'localhost'
PORT = 9000
SAMPLING_RATE = 16000  # same as in whisper_online_server.py
CHANNELS = 1
CHUNK = 4096 // (2 * CHANNELS)  # chunk size for audio streaming, bc receiving at 4096 bytes; calc on p. 119 of log; originally p. 109
DTYPE = np.int16

# Extraneous code from Coqui TTS examples - moved to whisper_online_server.py
    # # Get device
    # device = "cuda" if torch.cuda.is_available() else "cpu"

    # # Initialize TTS with the target model name
    # tts = TTS("tts_models/en/ljspeech/fast_pitch").to(device)

# def receive_from_server(sock):
#     '''Thread to receive audio from server.'''
#     while True:
#         try:
#             data = sock.recv(4096)  # (4KB) standard buffer size
#             if not data:
#                 break
#             data_queue.put(data)  # put data in queue for audio playback
#             print(f"Received {len(data)} bytes of audio data", flush=True)
#         except Exception as e:
#             print(f"Error receiving data: {e}")
#             break

def receive_from_server(sock):
    '''Thread to receive audio from server.'''
    remaining = b''  # Buffer for leftover bytes from previous recv
    while True:
        try:
            chunk = sock.recv(4096)  # Still recv in 4096-byte increments
            if not chunk:
                break
            data = remaining + chunk
            aligned_len = len(data) // 2 * 2  # Largest multiple of 2 (for int16)
            if aligned_len > 0:
                data_queue.put(data[:aligned_len])  # Put aligned bytes into queue
                print(f"Received and queued {aligned_len} aligned bytes of audio data", flush=True)
            remaining = data[aligned_len:]  # Carry over any remainder
        except Exception as e:
            print(f"Error receiving data: {e}")
            break

def audio_callback(output_data, frames, time, status):
    '''Callback for outputting the received audio chunk'''
    try:
        data = data_queue.get_nowait()  # get data from queue if available
    except queue.Empty:
        output_data.fill(0)  # if no data, output silence
        return
    
    # Ensure buffer length is a multiple of element size
    if len(data) % np.dtype(DTYPE).itemsize != 0:
        data = data[:len(data) - (len(data) % np.dtype(DTYPE).itemsize)]
        print(f"Adjusted data length to {len(data)} bytes for proper dtype alignment", flush=True)
    
    audio = np.frombuffer(data, dtype=DTYPE)  # convert bytes back to numpy array
    
    # reshape to match (frames, channels)
    audio = audio.reshape(-1, CHANNELS)

    # If audio chunk is smaller than expected, pad with zeros
    if audio.shape[0] < frames:
        padding = np.zeros((frames - audio.shape[0], CHANNELS), dtype=DTYPE)
        audio = np.vstack((audio, padding))
    elif audio.shape[0] > frames:
        audio = audio[:frames, :]  # truncate if too long

    output_data[:] = audio[:frames]  # output the audio chunk

def output_audio_stream():
    '''Runs the output audio stream'''
    with sd.OutputStream(samplerate=SAMPLING_RATE, channels=CHANNELS, dtype=DTYPE, callback=audio_callback):
        while True:
            sd.sleep(1000)  # keep stream alive
    
# Make audio data queue
data_queue = queue.Queue()

# Set up socket connection
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.connect((HOST, PORT))
    print(f"Connected to server at {HOST}:{PORT}. Listening for audio...")
    sock.send('tts'.encode('utf-8'))  # initial message to server, specifies client type

    # Start receiver thread
    receiver_thread = threading.Thread(target=receive_from_server, args=(sock,))
    receiver_thread.daemon = True
    receiver_thread.start()

    # Start audio output stream thread
    output_audio_stream()
    # audio_thread = threading.Thread(target=output_audio_stream)
    # audio_thread.daemon = True
    # audio_thread.start()

except Exception as e:
    print(f"Failed to connect: {e}")
    sys.exit(1)