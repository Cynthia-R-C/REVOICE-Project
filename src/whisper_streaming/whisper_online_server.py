#!/usr/bin/env python3
from whisper_online import *  #
import line_packet
import socket

import torch   # for running on GPU
from threading import Thread
import queue  # for TTS text queue
import librosa  # for resampling SR

import sys
import argparse
import os
import logging
import numpy as np

# ======= Calculating WER and latency ======= #
import time
from jiwer import wer
reference_file = 'english_patient.txt'  # reference text for WER calculation

latencies = []    # create list of latencies to track latency of each audio chunk so as to calculate average latency at end
start_times = {}  # dict of average start time of current text segment
# key is segment ID (segment start time)
# value is average perf counter time of receiving the audio chunks
stt_destut_ls = []
tts_destut_ls = []
rvc_ls = []
tts_synth_ls = []
stt_synth_ls = []
buffer_and_queue_wait_ls = []


# ======= DESTUTTERING IMPORTS/CONSTANTS ======= #
import sys
destutter_dir = os.path.abspath('C:\\Users\\crc24\\Documents\\VS_Code_Python_Folder\\ScienceFair2025\\src\\destutter')
sys.path.append(destutter_dir)  # add destutter folder to paths to search
from destutterer import Destutterer

SAMPLING_RATE = 16000
CONFIG_PATH = 'C:\\Users\\crc24\\Documents\\VS_Code_Python_Folder\\ScienceFair2025\\interspeech2024-code\\sed\\examples\\stutter_event\\s0\\conf\\train_stutternet.yaml'
CKPT_PATH = 'C:\\Users\\crc24\\Documents\\VS_Code_Python_Folder\\ScienceFair2025\\interspeech2024-code\\exp\\stutternet_en\\36.pt'
CMVN_PATH = 'C:\\Users\\crc24\\Documents\\VS_Code_Python_Folder\\ScienceFair2025\\interspeech2024-code\\data\\train\\global_cmvn'

# Device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"


# Finding the right base for online
def get_base_online():
        '''Sets online var to be the correct base based on if VAC is enabled'''
        # If VAC is enabled, online is a VACOnlineASRProcessor
        if isinstance(online, VACOnlineASRProcessor):
            return online.online    # the real OnlineASRProcessor inside
        else:
            return online          # plain OnlineASRProcessor
        
# base_online = get_base_online()

# min_chunk = args.min_chunk_size


# ======= Testing ======= #
GROUP = 'latency_testing'
# TRIAL = '1'  #  will just have to manually rename the file as I test unless I wanna stop the server and restart it just to update the constant in file naming and that’s not worth it
TRANSCRIPT_PATH = f'test_results/{GROUP}/transcript.txt'
STATS_PATH = f'test_results/{GROUP}/stats.txt'


# ======= Other Toggles ======= #
SAVE_TRANSCRIPT = True
tts_flag = False  # becomes true when a TTS client connects to the server
RVC_FLAG = True   # choose whether to enable RVC or not
TXT_DESTUT = False # whether or not to do text destuttering
AUD_DESTUT = False  # whether or not to do audio 

USE_COQUI = False
USE_MELO = True

if USE_COQUI and USE_MELO:  # just in case
    raise ValueError("Do not set both USE_COQUI and USE_MELO to True.")

# Add TTS stuff
if USE_COQUI:
    from TTS.api import TTS as CoquiTTS  # CoquiTTS
elif USE_MELO:
    from melo.api import TTS as MeloTTS  # MeloTTS, which has better prosody control

# Coqui settings
# TTS_MODEL = 'tts_models/multilingual/multi-dataset/xtts_v2'
COQUI_MODEL = 'tts_models/en/ljspeech/fast_pitch'

# Melo settings
MELO_LANGUAGE = 'EN'
MELO_SPEAKER = 'EN-Default'
MELO_SPEED = 0.8

TTS_GROUPING_ENABLED = True
ARTIFIC_INTON = True   # whether or not to normalize text groups with punctuation before TTS conversion
TTS_MAX_WAIT_SEC = 10
TTS_MIN_WORDS = 1
TTS_MAX_WORDS = 20
TTS_END_PUNCT = '.?!,:;'


# Main function within if __name__ == '__main__' to prevent infinite process spawning on Windows
def main():
    # Use global keywords so the rest of script can see these objects
    global args, asr, online, base_online, rvc_converter, min_chunk, size, language
    global tts, TTS_SR, destutterer_stt, destutterer_tts, melo_speaker_ids
    
 
    # ======= Logging and Arguments ======= #
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser()

    parser.add_argument("--host", type=str, default='localhost')
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--warmup-file", type=str, dest="warmup_file", 
            help="The path to a speech audio wav file to warm up Whisper so that the very first chunk processing is fast. It can be e.g. https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav .")

    # options from whisper_online
    add_shared_args(parser)
    args = parser.parse_args()

    set_logging(args,logger,other="")


    # ======= Whisper Settings ======= #
    size = args.model
    language = args.lan
    asr, online = asr_factory(args)

    base_online = get_base_online()

    min_chunk = args.min_chunk_size


    # ======= Set Up TTS ======= #
    if USE_COQUI:
        tts = CoquiTTS(COQUI_MODEL).to(device)
        TTS_SR = tts.synthesizer.output_sample_rate
        melo_speaker_ids = None

    elif USE_MELO:
        tts = MeloTTS(language=MELO_LANGUAGE, device=device)
        melo_speaker_ids = tts.hps.data.spk2id
        TTS_SR = tts.hps.data.sampling_rate

    else:
        print('[WARNING] Not using either CoquiTTS or MeloTTS')
    
    # # Initialize CoquiTTS with the target model name
    # tts = TTS(TTS_MODEL).to(device)
    # # TTS Constants
    # TTS_SR = tts.synthesizer.output_sample_rate  # TTS sampling rate


    # ======= Set Up Destutterers ======= #
    destutterer_stt = Destutterer(config_path=CONFIG_PATH,
                          ckpt_path=CKPT_PATH,
                          cmvn_path=CMVN_PATH,
                          sr=SAMPLING_RATE,
                          device=device)
    destutterer_tts = Destutterer(config_path=CONFIG_PATH,
                          ckpt_path=CKPT_PATH,
                          cmvn_path=CMVN_PATH,
                          sr=SAMPLING_RATE,
                          device=device)


    # ======= Set Up RVC ======= #
    from rvc_conversion import RVC
    rvc_converter = RVC() # initialize once; latency-optimized 




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

    # Start the socket server
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        server = Server(sock, args.host, args.port)
        server.listen()

    logger.info("Server stopping...")
    sys.exit(0)


######### Server objects

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

        # For TTS grouping buffer
        self.tts_text_buffer = []
        self.tts_buffer_start_time = None      # perf_counter when this group started buffering
        self.tts_group_beg = None              # transcript timestamp of first chunk in group
        self.tts_group_end = None              # transcript timestamp of latest chunk in group
        self.tts_group_start_perf = None       # perf_counter corresponding to first chunk's arrival
        
        self.last_text_received_time = time.perf_counter()  # track when last text was received for auto buffer flushing after inactivity
        self.SILENCE_TIMEOUT = 0.3  # Seconds of silence before forcing flush

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
        

    def send_result(self, o):
        '''Edited to do all of the following:
         1. Send the current transcript to the STT client
         2. Update the transcript text file if toggled'''
        msg = self.format_output_transcript(o)  # string of timestamps + pure text
        if msg is not None:
            self.connection.send(msg)  # send to STT client

            # Toggle between grouping and not grouping
            if TTS_GROUPING_ENABLED:
                self.group_and_put_to_tts(o)
            else:
                self.put_to_tts(o)
        
            if SAVE_TRANSCRIPT:
                # Write the text to the output transcript file
                self.out_file.write(o[2])

    def flush_tts_group(self):
        '''Force any buffered grouped TTS text into the queue.'''

        # If empty
        if not self.tts_text_buffer:
            return

        full_text = " ".join(self.tts_text_buffer).strip()

        # If empty after stripping
        if not full_text:
            return

        # Send the rest without waiting for grouping conditions
        grouped_o = (self.tts_group_beg, self.tts_group_end, full_text)
        # queue_t0 = time.perf_counter()  # Using self.tts_buffer_start_time instead
        self.tts_queue.put((grouped_o, self.tts_buffer_start_time, self.tts_group_start_perf))
        logger.info(f'Flushed final TTS grouped chunk: {full_text!r}')

        # Reset all the necessary stuff
        self.tts_text_buffer = []
        self.tts_buffer_start_time = None
        self.tts_group_beg = None
        self.tts_group_end = None
        self.tts_group_start_perf = None

    def group_and_put_to_tts(self, o):
        '''Do prosody-based group enqueuing rather than direct TTS queuing'''
        text = o[2].strip()
        if not text:
            return

        now = time.perf_counter()

        # Keeps track for stuff like calculating if waited too long
        if self.tts_buffer_start_time is None:
            self.tts_buffer_start_time = now  # start time for tts buffer latency tracking

        # If this is the start of a prosody group, record start timestamp
        if not self.tts_text_buffer:
            self.tts_group_beg = o[0]
            self.tts_group_start_perf = start_times.get(o[0], now)

        self.tts_group_end = o[1]
        self.tts_text_buffer.append(text)

        full_text = " ".join(self.tts_text_buffer).strip()  # add to text buffer
        word_count = len(full_text.split())

        # Determine whether conditions for grouping / pushing are met
        ends_cleanly = len(text) > 0 and text[-1] in TTS_END_PUNCT
        waited_too_long = (now - self.tts_buffer_start_time) >= TTS_MAX_WAIT_SEC
        enough_words = word_count >= TTS_MIN_WORDS
        too_many_words = word_count >= TTS_MAX_WORDS

        if ends_cleanly or too_many_words or (waited_too_long and enough_words):
            if ARTIFIC_INTON and full_text and full_text[-1] not in TTS_END_PUNCT:
                full_text += ","  # artificially add for better intonation in TTS

            grouped_o = (self.tts_group_beg, self.tts_group_end, full_text)
            # queue_t0 = time.perf_counter()   # not using this anymore, using the grouped buffer start time instead to get the full picture of the latency

            # Send the group's true start perf-counter through the queue too
            self.tts_queue.put((grouped_o, self.tts_buffer_start_time, self.tts_group_start_perf))
            logger.info(f'TTS grouped chunk queued: {full_text!r}')

            self.tts_text_buffer = []
            self.tts_buffer_start_time = None
            self.tts_group_beg = None
            self.tts_group_end = None
            self.tts_group_start_perf = None


    def put_to_tts(self, o):
        '''Old function.
        Puts the newly generated STT o to the TTS queue;
        Will be called no matter if TTS flag is on or off
        Assumes tts_queue is not None
        Assumes text is not None'''
        queue_t0 = time.perf_counter()   # latency tracking for time spent waiting in the queue
        chunk_start_perf = start_times.get(o[0], queue_t0)
        self.tts_queue.put((o, queue_t0, chunk_start_perf))  # adjust shape for regular vs grouped queuing (changed shape from before)
        logger.info("New o added to TTS queue.")


    def process(self):
        '''handle one stt client connection'''
        self.online_asr_proc.init()
        while True:
            a, startTimes = self.receive_audio_chunk()
            if a is None:
                break
            self.online_asr_proc.insert_audio_chunk(a)
            stt_synth_t0 = time.perf_counter()
            o = online.process_iter()   # o[0]: beg, o[1]: end, o[2]: text string

            now = time.perf_counter()

            stt_synth_t1 = time.perf_counter()
            logger.info(f'[LATENCY] STT processing took {stt_synth_t1 - stt_synth_t0:.3f}s')
            stt_synth_ls.append(stt_synth_t1 - stt_synth_t0)  # add to STT processing latency list

            if o[0] is not None:  # if audio is not blank

                # Find average start time perf counter and add to global tuple with ID
                avg_start_time = calc_avg(startTimes)
                start_times[o[0]] = avg_start_time

                self.last_text_received_time = now

                # ============== STT DESTUTTERING LOGIC ================ #

                if TXT_DESTUT:

                    # Track destuttering logic start time
                    t0 = time.perf_counter()

                    t_to_buffer = base_online.buffer_time_offset  # time between global stream start and audio buffer start
                    audio_buffer = base_online.audio_buffer  # current audio buffer - a list of samples
                    beg_time = o[0]  # beg time of text (global)
                    end_time = o[1]  # end time of text (global)
                    text = o[2]      # text of current segment
                    
                    t_maxs, stutter_word_idxs = destutterer_stt.get_destutter_info('stt', t_to_buffer, audio_buffer, beg_time, end_time, text)  # get t_maxs and stutter word indices
                    
                    ## Simple: SOUND REP ##
                    destutterer_stt.r_destutter()

                    ## Medium: WORD REP ##
                    # Only run this if word rep detected in model(?) might not be necessary, could just run this without model
                    if t_maxs['/wr'] is not None:
                        destutterer_stt.wr_destutter()

                    ## Complex: INTERJECTIONS ##
                    destutterer_stt.i_destutter(stutter_word_idxs)

                    # Update output o with destuttered text
                    new_txt = destutterer_stt.get_text()
                    o = (o[0], o[1], new_txt)

                    # See how long STT destuttering took
                    t1 = time.perf_counter()
                    logger.info(f"[LATENCY] STT get_destutter_info took {t1 - t0:.3f}s")
                    stt_destut_ls.append(t1 - t0)  # add to list of stt destuttering latencies


                # ============== END DESTUTTERING LOGIC ================ #
            
                try:
                    self.send_result(o)  # sends it to the client and if toggled on to the transcript file and TTS queue

                    # # Now add latencies of that audio just then to latencies list  # old latencies calculation code
                    # endTime = time.perf_counter()  # perf_counter is more precise
                    # for startTime in startTimes:
                    #     latencies.append(endTime - startTime)

                except BrokenPipeError:
                    logger.info("broken pipe -- connection closed?")
                    break

            else:
                if TTS_GROUPING_ENABLED and self.tts_text_buffer and (now - self.last_text_received_time) > self.SILENCE_TIMEOUT:
                    logger.info(f'\n\nSilence timeout of {self.SILENCE_TIMEOUT}s reached. Flushing buffer.\n\n')
                    self.flush_tts_group()

        if TTS_GROUPING_ENABLED:  # flush if leftover stuff left in tts buffer
            self.flush_tts_group()

#        o = online.finish()  # this should be working
#        self.send_result(o)

def calc_avg(l):
    '''Calculates the average given a list of floats'''
    # Used for both latencies and startTimes (adding the avg startTime from startTimes into start_times)
    if len(l) == 0:
        return 0.0
    return sum(l) / len(l)

def calc_wer(transcript_path, ref_path):
    '''Returns the WER given the transcript file and the reference file we are comparing it to'''

    # Read reference file
    with open(ref_path, 'r') as ref:
        ref_txt = ref.read()
    
    # Read transcript text
    with open(transcript_path, 'r') as t:
        transc_txt = t.read()
    
    return wer(ref_txt, transc_txt)


def synthesize_text(text):
    '''Helper function for TTS synthesis'''

    if USE_COQUI:
        return tts.tts(text)

    elif USE_MELO:
        speaker_id = melo_speaker_ids[MELO_SPEAKER]

        # Melo writes to file, so use temp file and read it back
        temp_path = 'temp_melo_tts.wav'
        tts.tts_to_file(
            text=text,
            speaker_id=speaker_id,
            output_path=temp_path,
            speed=MELO_SPEED
        )

        wav, sr = librosa.load(temp_path, sr=None, mono=True)
        return wav

    else:
        raise ValueError('No TTS backend selected.')


class Server:
    Clients = [] # list of client threads

    def __init__(self, sock, HOST, PORT):
        '''Initializes TCP socket over IPv4. Accepts 2 connections max.'''
        self.tts_queue = queue.Queue()  # queue for STT o to be sent to TTS client
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
                    # for regular, queue contents = (o, buffer_start_t, chunk_start_perf), where o is regular (beg_stamp, end_stamp, txt)
                    # for grouped, queue contents = (grouped_o, buffer_start_t, group_start_perf), where o is (group_beg_stamp, group_end_stamp, full_grouped_txt)
                    (o, buffer_start_t, group_start_perf) = tts_queue.get()  # this should also remove it from the queue
                    queue_t1 = time.perf_counter()
                    buffer_and_queue_wait_ls.append(queue_t1 - buffer_start_t)  # add to list of times spent waiting in the queue
                    logger.info(f'[LATENCY] Total time in buffer & TTS queue: {queue_t1 - buffer_start_t:.3f}s')

                    logger.debug("o received from TTS queue.")

                    text = o[2]

                    # Generate speech
                    logger.debug("GENERATING TTS audio...")
                    tts_synth_t0 = time.perf_counter()

                    # Artificially normalize text by adding periods to pauses in text so TTS better captures intonation
                    if ARTIFIC_INTON:
                        if text and text[-1] not in TTS_END_PUNCT:
                            text += ","

                    # wav = tts.tts(text)
                    wav = synthesize_text(text)
                    tts_synth_t1 = time.perf_counter()
                    logger.info(f'[LATENCY] TTS synthesis took {tts_synth_t1 - tts_synth_t0:.3f}s')
                    tts_synth_ls.append(tts_synth_t1 - tts_synth_t0)  # add to TTS synthesis latency list

                    logger.debug("TTS audio generated from text.")
                    wav = np.array(wav)   # convert to np array to avoid memory issues
                    #logger.debug("wav shape:", wav.shape, "dtype:", wav.dtype)

                    # Debugging
                    print(f"TTS output sample rate: {TTS_SR}")
                    print(f"Original wav shape: {wav.shape}, dtype: {wav.dtype}")
                    soundfile.write(f'original_{TTS_SR}hz.wav', wav, TTS_SR)  # Save original before resampling

                    # Resample to 16kHz if needed
                    if TTS_SR != SAMPLING_RATE:
                        wav = librosa.resample(wav, orig_sr=TTS_SR, target_sr=SAMPLING_RATE)
                        logger.debug(f"Resampled TTS audio from {TTS_SR}Hz to {SAMPLING_RATE}Hz.")

                        # Debugging
                        print(f"Resampled wav shape: {wav.shape}, dtype: {wav.dtype}")
                        soundfile.write(f'resampled_{SAMPLING_RATE}hz.wav', wav, SAMPLING_RATE)  # Save after resampling

                    
                    # ============== TTS DESTUTTERING LOGIC ================ #

                    if AUD_DESTUT:

                        # Latency time tracker
                        t0 = time.perf_counter()

                        # TTS audio destuttering logic: prolongations & blocks

                        t_to_buffer = base_online.buffer_time_offset  # time between global stream start and audio buffer start
                        audio_buffer = base_online.audio_buffer  # current audio buffer - a list of samples
                        beg_time = o[0]  # beg time of text (global)
                        end_time = o[1]  # end time of text (global)

                        p_times, b_times = destutterer_tts.get_destutter_info('tts', t_to_buffer, audio_buffer, beg_time, end_time, text)  # get segment times to cut

                        ## PROLONGATION ##
                        wav = destutterer_tts.p_destutter(wav, p_times[0], p_times[1])

                        ## BLOCK ##
                        wav = destutterer_tts.b_destutter(wav, b_times[0], b_times[1])

                        # Print section latency
                        t1 = time.perf_counter()
                        logger.info(f'[LATENCY] TTS get_destutter_info took {t1 - t0:.3f}s')
                        tts_destut_ls.append(t1 - t0)  # add to list of tts destuttering latencies


                    # ============== DESTUTTERING LOGIC END ================ #


                    # RVC logic (if flag enabled)
                    tr0 = time.perf_counter()
                    if RVC_FLAG:

                        # Only use RVC if audio is long enough to be meaningful
                        min_samples = 2000  # ~0.125s at 16kHz
                        if wav.shape[0] >= min_samples:
                            # Pass through RVC conversion
                            logger.debug("[RVC] Passing TTS audio through RVC...")
                            new_aud = rvc_converter.vc(wav)  # outputs 2D array, need to squeeze to 1D
                            
                            if new_aud.shape[0] == 0:
                                # In case RVC returns empty because the audio < block frames and is too short (non-ideal), preserve original speech
                                logger.warning("[RVC] RVC returned empty, preserving original.")
                            else:
                                wav = new_aud.squeeze(axis=1)  # squeeze to 1D
                                logger.debug("[RVC] TTS audio processed through RVC.")
                        else:
                            logger.debug(f"[RVC] Skipping RVC for short audio ({wav.shape[0]} samples)")

                    
                    tr1 = time.perf_counter()
                    logger.info(f'[LATENCY] RVC processing took {tr1 - tr0:.3f}s')
                    rvc_ls.append(tr1 - tr0)  # add to RVC latency list

                    # Convert to 16-bit PCM
                    wav_pcm = (wav * 32767).astype(np.int16)
                    
                    # Record end time for latency
                    endTime = time.perf_counter()
                    # Now add latencies of that audio just then to latencies list
                    latencies.append(endTime - group_start_perf)

                    # Remove start time for this segment from the start_times dictionary
                    # But clean up only if that key exists in the dict
                    if o[0] in start_times:
                        del start_times[o[0]]
                   
                    # Send the packet of audio data
                    try:
                        logger.debug("Sending audio to TTS client...")
                        conn.sendall(wav_pcm.tobytes())
                        logger.debug("Audio sent to TTS client.")
                    except (BrokenPipeError, ConnectionResetError):  # in case something goes wrong with the connection
                        logger.info("broken pipe / connection reset -- connection closed?")
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
        # to fix the local variable referenced before assignment error
        global latencies   
        global stt_destut_ls
        global tts_destut_ls
        global stt_synth_ls
        global tts_synth_ls
        global buffer_and_queue_wait_ls

        client_type = client['type']
        conn = client['conn/socket']
        addr = client['addr']
        
        out_file = None
        if SAVE_TRANSCRIPT:
            out_file = open(TRANSCRIPT_PATH, 'w')  # open a new file for writing the transcript

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
            txt_wer = calc_wer(TRANSCRIPT_PATH, reference_file)
            logger.info(f"WER: {txt_wer:.3f}")

        # Latency calculation
        logger.info(f'Average latency: {calc_avg(latencies):.3f}s')
        logger.info(f'Average STT synthesis latency: {calc_avg(stt_synth_ls):.3f}s')
        logger.info(f'Average TTS synthesis latency: {calc_avg(tts_synth_ls):.3f}s')
        logger.info(f'Average time spent waiting in buffer & TTS queue: {calc_avg(buffer_and_queue_wait_ls):.3f}s')
        logger.info(f'Average STT destuttering latency: {calc_avg(stt_destut_ls):.3f}s')
        logger.info(f'Average TTS destuttering latency: {calc_avg(tts_destut_ls):.3f}s')
        logger.info(f'Average RVC latency: {calc_avg(rvc_ls):.3f}s')

        # Save in stats file
        stats = open(STATS_PATH, 'w')
        stats.write(f"WER: {txt_wer:.3f}\n")
        stats.write(f'Average latency: {calc_avg(latencies):.3f}s\n')
        stats.write(f'Average STT synthesis latency: {calc_avg(stt_synth_ls):.3f}s\n')
        stats.write(f'Average TTS synthesis latency: {calc_avg(tts_synth_ls):.3f}s\n')
        stats.write(f'Average time spent waiting in buffer & TTS queue: {calc_avg(buffer_and_queue_wait_ls):.3f}s\n')
        stats.write(f'Average STT destuttering latency: {calc_avg(stt_destut_ls):.3f}s\n')
        stats.write(f'Average TTS destuttering latency: {calc_avg(tts_destut_ls):.3f}s\n')
        stats.write(f'Average RVC latency: {calc_avg(rvc_ls):.3f}s\n')
        stats.close()

        latencies = []   # clear latencies list for next client session
        stt_destut_ls = []  # clear list
        tts_destut_ls = []  # clear list





# ========= Main server code ======= #
if __name__ == '__main__':
    main()


# # Server code
# with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
#         server = Server(sock, args.host, args.port)
#         server.listen()

# logger.info("Server stopping...")
# sys.exit(0)

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
