#!/usr/bin/env python3
from whisper_online import *  #

# Add CoquiTTS stuff
from TTS.api import TTS
import torch   # for running on GPU
from threading import Thread
import queue  # for TTS text queue
import librosa  # for resampling SR

import sys
import argparse
import os
import logging
import numpy as np

# Calculating WER and latency
import time
from jiwer import wer

logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()

# server options
parser.add_argument("--host", type=str, default='localhost')
parser.add_argument("--port", type=int, default=9000)
parser.add_argument("--warmup-file", type=str, dest="warmup_file", 
        help="The path to a speech audio wav file to warm up Whisper so that the very first chunk processing is fast. It can be e.g. https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav .")

# options from whisper_online
add_shared_args(parser)
args = parser.parse_args()

set_logging(args,logger,other="")

# setting whisper object by args 

SAMPLING_RATE = 16000

size = args.model
language = args.lan
asr, online = asr_factory(args)
min_chunk = args.min_chunk_size
latencies = []    # create list of latencies to track latency of each audio chunk so as to calculate average latency at end

# Device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# Set up TTS
# Initialize CoquiTTS with the target model name
tts = TTS("tts_models/en/ljspeech/fast_pitch").to(device)

# TTS Constants
TTS_SR = tts.synthesizer.output_sample_rate  # TTS sampling rate

# Toggles
SAVE_TRANSCRIPT = True
tts_flag = False  # becomes true when a TTS client connects to the server

# warm up the ASR because the very first transcribe takes more time than the others. 
# Test results in https://github.com/ufal/whisper_streaming/pull/81
msg = "Whisper is not warmed up. The first chunk processing may take longer."
if args.warmup_file:
    if os.path.isfile(args.warmup_file):
        a = load_audio_chunk(args.warmup_file,0,1)
        asr.transcribe(a)
        logger.info("Whisper is warmed up.")
    else:
        logger.critical("The warm up file is not available. "+msg)
        sys.exit(1)
else:
    logger.warning(msg)


######### Server objects

import line_packet
import socket

class Connection:
    '''it wraps conn object'''
    PACKET_SIZE = 32000*5*60 # 5 minutes # was: 65536

    def __init__(self, conn):
        self.conn = conn
        self.last_line = ""

        self.conn.setblocking(True)

    def send(self, line):
        '''it doesn't send the same line twice, because it was problematic in online-text-flow-events'''
        if line == self.last_line:
            return
        line_packet.send_one_line(self.conn, line)  # defined in line_packet.py
        self.last_line = line

    def receive_lines(self):
        in_line = line_packet.receive_lines(self.conn)  # defined in line_packet.py
        return in_line

    def non_blocking_receive_audio(self):
        try:
            r = self.conn.recv(self.PACKET_SIZE)
            return r
        except ConnectionResetError:
            return None


import io
import soundfile

# wraps socket and ASR object, and serves one client connection. 
# next client should be served by a new instance of this object
class ServerProcessor:

    def __init__(self, c, online_asr_proc, min_chunk, out_file=None, tts_queue=None):
        self.connection = c
        self.online_asr_proc = online_asr_proc
        self.min_chunk = min_chunk

        self.out_file = out_file  # for storing transcription results in a text file; None if not toggled on
        self.tts_queue = tts_queue  # queue for TTS text to be sent to TTS client

        self.last_end = None

        self.is_first = True

    def receive_audio_chunk(self):
        '''
        receive all audio that is available by this time
        blocks operation if less than self.min_chunk seconds is available
        unblocks if connection is closed or a chunk is available

        Note: modified to return not only all the audio that is available by this time but also a list of the start times for each
        '''
        
        out = []
        times = []  # a list of the times a chunk in the out list is received

        minlimit = self.min_chunk*SAMPLING_RATE  # min samples
        while sum(len(x) for x in out) < minlimit:  # keep receiving until it reaches min chunk size
            raw_bytes = self.connection.non_blocking_receive_audio()
            if not raw_bytes: # if empty
                break
#            print("received audio:",len(raw_bytes), "bytes", raw_bytes[:10])
            sf = soundfile.SoundFile(io.BytesIO(raw_bytes), channels=1,endian="LITTLE",samplerate=SAMPLING_RATE, subtype="PCM_16",format="RAW")  # audio is in raw PCM 16
            audio, _ = librosa.load(sf,sr=SAMPLING_RATE,dtype=np.float32)
            out.append(audio)
            times.append(time.perf_counter())  # append the start time of this audio to the times list

        if not out:  # if no audio
            return None, None
        
        conc = np.concatenate(out)

        if self.is_first and len(conc) < minlimit:  # can't be less than min limit
            return None, None
        
        self.is_first = False
        return np.concatenate(out), times   # returns the out list and the start times of each audio piece in the out list

    def format_output_transcript(self,o):
        '''
        # output format in stdout is like:
        # 0 1720 Takhle to je
        # - the first two words are:
        #    - beg and end timestamp of the text segment, as estimated by Whisper model. The timestamps are not accurate, but they're useful anyway
        # - the next words: segment transcript

        # This function differs from whisper_online.output_transcript in the following:
        # succeeding [beg,end] intervals are not overlapping because ELITR protocol (implemented in online-text-flow events) requires it.
        # Therefore, beg, is max of previous end and current beg outputed by Whisper.
        # Usually it differs negligibly, by appx 20 ms.
        '''

        if o[0] is not None:
            beg, end = o[0]*1000,o[1]*1000
            if self.last_end is not None:
                beg = max(beg, self.last_end)

            self.last_end = end
            print("%1.0f %1.0f %s" % (beg,end,o[2]),flush=True,file=sys.stderr)
            return "%1.0f %1.0f %s" % (beg,end,o[2])
        else:
            logger.debug("No text in this segment")
            return None
        
    def format_output_transcript_text_only(self,o):
        '''A copy of the above method except doesn't print the timestamps; for my own testing purposes
        Does not print out the text to the terminal'''
        if o[0] is not None:
            return o[2]  # this is the text portion
        else:
            return None

    def send_result(self, o):
        '''Edited to do all of the following:
         1. Send the current transcript to the STT client
         2. Update the transcript text file if toggled'''
        msg = self.format_output_transcript(o)
        if msg is not None:
            self.connection.send(msg)  # send to STT client
            self.put_to_tts(self.strip_timestamps(msg))
        
        if SAVE_TRANSCRIPT:
            # Write the text to the output transcript file
            txt = self.format_output_transcript_text_only(o)
            if txt is not None:
                self.out_file.write(txt)

    def strip_timestamps(self, txt):
        '''Strips the timestamps from the text'''
        parts = txt.split(' ')
        if len(parts) > 2:
            return ' '.join(parts[2:])  # join all parts after the first two (the timestamps)
        else:
            return ''  # if there is no text after the timestamps, return empty string

    def put_to_tts(self, txt):
        '''Puts the newly generated STT text to the TTS queue;
        Will be called no matter if TTS flag is on or off
        Assumes tts_queue is not None
        Assumes text is not None'''
        self.tts_queue.put(txt)
        logger.info("New text added to TTS queue.")


    def process(self):
        '''handle one stt client connection'''
        self.online_asr_proc.init()
        while True:
            a, startTimes = self.receive_audio_chunk()
            if a is None:
                break
            self.online_asr_proc.insert_audio_chunk(a)
            o = online.process_iter()
            try:
                self.send_result(o)  # sends it to the client and if toggled on to the transcript file and TTS queue

                # Now add latencies of that audio just then to latencies list
                endTime = time.perf_counter()
                for startTime in startTimes:
                    latencies.append(endTime - startTime)

            except BrokenPipeError:
                logger.info("broken pipe -- connection closed?")
                break

#        o = online.finish()  # this should be working
#        self.send_result(o)

def calc_avg_latency(latencies):
    '''Calculates the average latency given a list of latencies'''
    if len(latencies) == 0:
        return 0.0
    return sum(latencies) / len(latencies)

def calc_wer(transcript_path, ref_path):
    '''Returns the WER given the transcript file and the reference file we are comparing it to'''

    # Read reference file
    with open(ref_path, 'r') as ref:
        ref_txt = ref.read()
    
    # Read transcript text
    with open(transcript_path, 'r') as t:
        transc_txt = t.read()
    
    return wer(ref_txt, transc_txt)



class Server:
    Clients = [] # list of client threads

    def __init__(self, sock, HOST, PORT):
        '''Initializes TCP socket over IPv4. Accepts 2 connections max.'''
        self.tts_queue = queue.Queue()  # queue for TTS text to be sent to TTS client
        self.socket = sock
        self.socket.bind((HOST, PORT))
        self.socket.listen(2)  # allow 2 clients to wait in line
        logger.info('Listening on'+str((HOST, PORT)))

    def listen(self):
        '''Listens for new clients and spawns a new thread for each client'''
        while True:
            conn, addr = self.socket.accept()  # conn is the socket, used conn instead of socket to stay consistent with the dev naming system for the original whisper online server
            logger.info('Connected to client on {}'.format(addr))

            # First message will be the client type = tts or stt
            client_type = conn.recv(1024).decode('utf-8').strip()
            client = {'conn/socket': conn, 'addr': addr, 'type': client_type}
            logger.info(f"{client_type} client has connected.")

            Server.Clients.append(client)

            if client_type == 'tts':
                client_thread = Thread(target=self.handle_tts_client, args=(client,self.tts_queue,))

            elif client_type == 'stt':
                client_thread = Thread(target=self.handle_stt_client, args=(client,self.tts_queue,))
                
            client_thread.start()

    def handle_tts_client(self, client, tts_queue):
        '''Handles one TTS client connection'''
        tts_flag = True  # set the TTS flag to true when a TTS client connects to the server
        #client_type = client['type']
        conn = client['conn/socket']

        while True:
            try:
                # As long as queue is not empty, get text from queue, convert it to audio, and send it over
                if not tts_queue.empty():
                    text = tts_queue.get()  # this should also remove it from the queue
                    logger.debug("Text received from TTS queue.")

                    # Generate speech
                    logger.debug("GENERATING TTS audio...")
                    wav = tts.tts(text)

                    logger.debug("TTS audio generated from text.")
                    wav = np.array(wav)   # convert to np array to avoid memory issues
                    #logger.debug("wav shape:", wav.shape, "dtype:", wav.dtype)

                    # Debugging
                    print(f"TTS output sample rate: {tts.synthesizer.output_sample_rate}")
                    print(f"Original wav shape: {wav.shape}, dtype: {wav.dtype}")
                    soundfile.write(f'original_{TTS_SR}hz.wav', wav, TTS_SR)  # Save original before resampling

                    # Resample to 16kHz if needed
                    if TTS_SR != SAMPLING_RATE:
                        wav = librosa.resample(wav, orig_sr=TTS_SR, target_sr=SAMPLING_RATE)
                        logger.debug(f"Resampled TTS audio from {TTS_SR}Hz to {SAMPLING_RATE}Hz.")

                        # Debugging
                        print(f"Resampled wav shape: {wav.shape}, dtype: {wav.dtype}")
                        sf.write(f'resampled_{SAMPLING_RATE}hz.wav', wav, SAMPLING_RATE)  # Save after resampling

                    # Convert to 16-bit PCM
                    wav_pcm = (wav * 32767).astype(np.int16)
                   
                    # Send the packet of audio data
                    try:
                        logger.debug("Sending audio to TTS client...")
                        conn.sendall(wav_pcm.tobytes())
                        logger.debug("Audio sent to TTS client.")
                    except BrokenPipeError:  # in case something goes wrong with the connection
                        logger.info("broken pipe -- connection closed?")
                        break
            
            # To end the client connection from the terminal press Ctrl+C
            except KeyboardInterrupt:
                logger.info("TTS client handler stopping...")
                break
        
        conn.close()
        tts_flag = False  # set the TTS flag to false when the TTS client disconnects from the server
        Server.Clients.remove(client)  # remove client from client list
        logger.info('Connection to tts client closed')


    def handle_stt_client(self, client, tts_queue):
        '''Handles one STT client connection'''
        global latencies   # to fix the local variable referenced before assignment error
        client_type = client['type']
        conn = client['conn/socket']
        addr = client['addr']
        
        out_file = None
        if SAVE_TRANSCRIPT:
            out_file = open('transcript.txt', 'w')  # open a new file for writing the transcript

        # Now keep waiting for audio from this client and process it
        # No while true loop here, the server processor handles the while true stuff for stt
        connection = Connection(conn)
        proc = ServerProcessor(connection, online, args.min_chunk_size, out_file, tts_queue)
        logger.info('Starting to process audio from STT client...')
        proc.process()
        conn.close()
        Server.Clients.remove(client)  # remove client from client list
        logger.info('Connection to stt client closed')

        if SAVE_TRANSCRIPT:
            # transcription file WER calculation
            out_file.close()
            logger.info('Transcript file written.')
            txt_wer = calc_wer('mrs_dalloway.txt', 'transcript.txt')
            logger.info(f"WER: {txt_wer:.3f}")

        # Latency calculation
        logger.info(f"Average latency: {calc_avg_latency(latencies):.3f}s")
        latencies = []   # clear latencies list for next client session


# Server code
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        server = Server(sock, args.host, args.port)
        server.listen()

logger.info("Server stopping...")
sys.exit(0)

### Old code from whisper_online_server.py below for reference ###
# server loop
# with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:  # each new client connection
#     s.bind((args.host, args.port))
#     s.listen(1)
#     logger.info('Listening on'+str((args.host, args.port)))
#     while True:
#         conn, addr = s.accept()
#         logger.info('Connected to client on {}'.format(addr))
#         out_file = open('transcript.txt', 'w')  # open a new file for writing the transcript
#         connection = Connection(conn)
#         proc = ServerProcessor(connection, online, args.min_chunk_size, out_file)
#         proc.process()
#         conn.close()
#         logger.info('Connection to client closed')

#         # transcription file WER calculation
#         out_file.close()
#         logger.info('Transcript file written.')
#         txt_wer = calc_wer('mrs_dalloway.txt', 'transcript.txt')
#         logger.info(f"WER: {txt_wer:.3f}")

#         # Latency calculation
#         logger.info(f"Average latency: {calc_avg_latency(latencies):.3f}s")
#         latencies = []   # clear latencies list for next client session

# logger.info('Connection closed, terminating.')
