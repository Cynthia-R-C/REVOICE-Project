# Attempt at basic real time TTS using Coqui

import torch
from TTS.api import TTS
#import os
import socket
import threading
import sys
import sd

# Server connection settings (match those used for the server)
HOST = 'localhost'
PORT = 9000

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize TTS with the target model name
tts = TTS("tts_models/en/ljspeech/fast_pitch").to(device)

def receive_from_server(sock):
    """Thread to receive transcription from server."""
    while True:
        try:
            data = sock.recv(4096)  # (4KB) standard buffer size; could try a smaller buffer to decrease latency
            if not data:
                break
            txt = data.decode('utf-8'), flush=True  # save decoded data
            output_audio(txt)  # run text through CoquiTTS and output sound in real time
        except Exception as e:
            print(f"Error receiving data: {e}")
            break

def output_audio(txt):
    '''Outputs the received text chunk as audio in real time'''
    wav = tts.tts(txt)
    # stream to sounddevice
    sd.play(wav, samplerate=tts.synthesizer.output_sample_rate)

# Set up socket connection
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.connect((HOST, PORT))
    print(f"Connected to server at {HOST}:{PORT}. Speak into your mic...")
except Exception as e:
    print(f"Failed to connect: {e}")
    sys.exit(1)

# Start receiver thread
receiver_thread = threading.Thread(target=receive_from_server, args=(sock,))
receiver_thread.daemon = True
receiver_thread.start()