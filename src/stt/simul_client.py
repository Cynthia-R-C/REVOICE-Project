# SimulStreaming STT client
# Sends audio to server and receives real-time transcription from server

import pyaudio
import socket
import threading
import sys

# Audio settings (must match server's expectations: 16kHz, mono, 16-bit signed integer)
RATE = 16000
CHUNK_SIZE = 1024  # Adjust if needed for smoother streaming
FORMAT = pyaudio.paInt16
CHANNELS = 1

# Server connection settings (match those used for the server)
HOST = 'localhost'
PORT = 9000

def receive_from_server(sock):
    '''Thread to receive and print transcription from server.'''
    while True:
        try:
            data = sock.recv(4096)  # (4KB) standard buffer size; could try a smaller buffer to decrease latency
            if not data:
                break
            print(data.decode('utf-8'), flush=True)  # print the decoded data
        except Exception as e:
            print(f"Error receiving data: {e}")
            break

# Set up socket connection
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.connect((HOST, PORT))
    print(f"Connected to server at {HOST}:{PORT}. Speak into your mic...")
    #sock.send('stt'.encode('utf-8'))  # initial message to server, specifies client type
    
except Exception as e:
    print(f"Failed to connect: {e}")
    sys.exit(1)

# Start receiver thread
receiver_thread = threading.Thread(target=receive_from_server, args=(sock,))
receiver_thread.daemon = True
receiver_thread.start()

# Set up PyAudio for mic input
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE)

try:
    while True:
        data = stream.read(CHUNK_SIZE)
        sock.sendall(data)
except KeyboardInterrupt:
    print("\nStopping...")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
    sock.close()