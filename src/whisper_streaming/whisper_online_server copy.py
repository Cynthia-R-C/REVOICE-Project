#!/usr/bin/env python3
from whisper_online import *  #
import line_packet
import socket

import torch   # for running on GPU
from threading import Thread
import queue  # for TTS text queue
import collections  # for audio receive deque
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

start_times = {}  # dict of average start time of current text segment
# key is segment ID (segment start time)
# value is average perf counter time of receiving the audio chunks

from latency_tracking import LatencyRecord, LatencyTracker
tracker = LatencyTracker()


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
GROUP = 'latency_testing_2'
# TRIAL = '1'  #  will just have to manually rename the file as I test unless I wanna stop the server and restart it just to update the constant in file naming and that’s not worth it
TRANSCRIPT_PATH = f'test_results/{GROUP}/transcript.txt'
STATS_PATH = f'test_results/{GROUP}/stats.txt'
AUD_DESTUT_OUTPUT_PATH = f'test_results/{GROUP}/aud_destut_output.wav'


# ======= Other Toggles ======= #
SAVE_TRANSCRIPT = True
tts_flag = False  # becomes true when a TTS client connects to the server
RVC_FLAG = True   # choose whether to enable RVC or not
TXT_DESTUT = True # whether or not to do text destuttering
AUD_DESTUT = True  # whether or not to do audio 
SAVE_AUD_DESTUT_OUTPUT = True  # save post-audio-destutter audio to a wav for inspection

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
TTS_MAX_WAIT_SEC = 0.3   # max seconds to stay in buffer; adds dash with this pause time
SILENCE_TIMEOUT = 1.1   # Seconds of silence before forcing flush; adds period with this pause time
TTS_END_PUNCT = '.?!,:;'


# Main function within if __name__ == '__main__' to prevent infinite process spawning on Windows
def main():
    # Use global keywords so the rest of script can see these objects
    global args, asr, online, base_online, rvc_converter, min_chunk, size, language
    global tts, TTS_SR, destutterer_stt, melo_speaker_ids
    
 
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
    # Note: destutterer_tts removed — audio destuttering (prolongations/blocks) now
    # happens pre-STT in receive_audio_chunk via destutterer_stt.aud_destutter_chunk()


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
            return r  # b'' means clean disconnect; non-empty bytes = data
        except (ConnectionResetError, ConnectionAbortedError, OSError):
            raise  # let _audio_receive_loop handle it


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
        self.tts_buffer_ids = []               # o[0] keys of every chunk added to the current group

        self.last_text_received_time = time.perf_counter()  # track when last text was received for auto buffer flushing after inactivity
        self.SILENCE_TIMEOUT = SILENCE_TIMEOUT  # Seconds of silence before forcing flush

        # Latency: store per-chunk STT timestamps here until the group is flushed, at which point they are averaged into the LatencyRecord for that group
        self.stt_synth_times = {}   # o[0] -> (stt_synth_start, stt_synth_end)
        self.stt_destut_times = {}  # o[0] -> (stt_destut_start, stt_destut_end)
        self.aud_destut_times = {}  # o[0] -> (aud_destut_start, aud_destut_end); keyed after STT runs
        self._last_aud_destut_times = None  # temp holding slot until o[0] is known
        self.processed_queue_times = {}   # o[0] -> (pq_enter, pq_exit)
        self._last_processed_queue_times = None  # temp holding slot until o[0] is known

        # Accumulate post-audio-destutter chunks for saving to wav at session end
        self.aud_destut_chunks = []  # filled in receive_audio_chunk when SAVE_AUD_DESTUT_OUTPUT is on

        # AUDIO RECEIVE THREAD
        # Continuously drains the socket into this deque so audio is never dropped while process_iter() is blocking for 5-15s inside Whisper.
        # Each entry = (raw_bytes, arrival_perf_counter)
        self._audio_deque = collections.deque()
        self._receive_thread_running = True
        self._receive_thread = Thread(target=self._audio_receive_loop, daemon=True)
        self._receive_thread.start()

        # AUDIO DESTUTTER THREAD
        # Assembles chunks from _audio_deque, runs aud_destutter_chunk off the main
        # thread, and puts processed results into _processed_queue.
        # Each entry = (conc, times, aud_times, pq_enter_time)
        # None sentinel signals end of stream.
        self._processed_queue = queue.Queue()
        self._last_processed_queue_times = None  # (pq_enter, pq_exit) held until o[0] known
        self._destutter_thread = Thread(target=self._destutter_loop, daemon=True)
        self._destutter_thread.start()

        # TTS BUFFER FLUSH
        self._flush_timer_thread = Thread(target=self._tts_flush_timer_loop, daemon=True)
        self._flush_timer_thread.start()

    def _tts_flush_timer_loop(self):
        '''Background thread: fires flush_tts_group() as soon as TTS_MAX_WAIT_SEC has elapsed since the buffer started filling, independently of whether new STT text has arrived. Preserves all prosody logic'''
        while self._receive_thread_running:
            time.sleep(0.05)  # check every 50ms — fine-grained enough, cheap enough
            if (self.tts_text_buffer
                    and self.tts_buffer_start_time is not None
                    and (time.perf_counter() - self.tts_buffer_start_time) >= TTS_MAX_WAIT_SEC):
                logger.info(f'[TIMER FLUSH] TTS_MAX_WAIT_SEC elapsed, flushing.')
                self.flush_tts_group()

    def _audio_receive_loop(self):
        '''Background thread: continuously drains raw bytes from the socket
        into self._audio_deque so audio is never dropped while process_iter()
        is blocking inside Whisper.
        Sets _receive_thread_running = False when the connection closes so
        receive_audio_chunk() knows to stop waiting and return None.'''
        while self._receive_thread_running:
            try:
                raw_bytes = self.connection.non_blocking_receive_audio()
            except Exception:
                # Any socket error = connection gone
                self._receive_thread_running = False
                break

            if raw_bytes:
                self._audio_deque.append((raw_bytes, time.perf_counter()))
            elif raw_bytes == b'' or raw_bytes is None:
                # recv() returning empty bytes means the client closed the connection
                self._receive_thread_running = False
                break
            else:
                # Shouldn't happen, but treat as transient and retry
                time.sleep(0.005)

    def _destutter_loop(self):
        '''Background thread: assembles min_chunk-sized pieces from _audio_deque, runs aud_destutter_chunk (slow StutterNet) here off the main thread, then puts results into _processed_queue
        Whisper and destuttering now run in parallel 
        
        When the queue is backed up, merges new chunks into the oldest waiting item so Whisper still hears all audio, just in larger batches'''
        minlimit = self.min_chunk * SAMPLING_RATE
        is_first_local = True

        while True:
            out = []
            times = []

            while sum(len(x) for x in out) < minlimit:
                if self._audio_deque:
                    raw_bytes, arrival = self._audio_deque.popleft()
                    sf = soundfile.SoundFile(io.BytesIO(raw_bytes), channels=1,
                                            endian="LITTLE", samplerate=SAMPLING_RATE,
                                            subtype="PCM_16", format="RAW")
                    audio, _ = librosa.load(sf, sr=SAMPLING_RATE, dtype=np.float32)
                    out.append(audio)
                    times.append(arrival)
                else:
                    if not self._receive_thread_running:
                        break
                    time.sleep(0.005)

            if not out:
                self._processed_queue.put(None)
                break

            conc = np.concatenate(out)

            if is_first_local and len(conc) < minlimit:
                continue
            is_first_local = False

            # Audio destuttering runs HERE, off the main thread.
            # Skip if Whisper is already backed up — no point burning GPU on
            # audio that will just sit in the queue, and it frees CUDA for Whisper.
            aud_times = None
            if AUD_DESTUT and self._processed_queue.qsize() == 0:
                t0 = time.perf_counter()
                conc = destutterer_stt.aud_destutter_chunk(conc)
                t1 = time.perf_counter()
                aud_times = (t0, t1)

            if SAVE_AUD_DESTUT_OUTPUT:
                self.aud_destut_chunks.append(conc.copy())

            pq_enter = time.perf_counter()
            new_item = (conc, times, aud_times, pq_enter)

            # If the queue is backed up, merge this chunk into the oldest waiting item rather than letting the queue grow unboundedly
            # Whisper still hears all audio, just in a larger batch
            if self._processed_queue.qsize() >= 2:
                try:
                    old_conc, old_times, old_aud_times, old_pq_enter = self._processed_queue.get_nowait()
                    merged_conc = np.concatenate([old_conc, conc])
                    merged_times = old_times + times  # preserve all arrival timestamps
                    # Keep the older pq_enter and aud_times so latency records
                    # reflect when this group first entered the queue
                    self._processed_queue.put_nowait((merged_conc, merged_times, old_aud_times, old_pq_enter))
                    logger.warning("[BACKPRESSURE] Merged chunk into queue — Whisper falling behind")
                except queue.Empty:
                    # Race condition: queue drained between qsize() check and get()
                    # Just push normally
                    self._processed_queue.put(new_item)
            else:
                self._processed_queue.put(new_item)

    def receive_audio_chunk(self):
        '''Pulls the next processed chunk from _processed_queue.
        Near-instant — all the heavy work happens in the background threads.'''
        item = self._processed_queue.get()
        if item is None:
            return None, None
        conc, times, aud_times, pq_enter = item
        pq_exit = time.perf_counter()
        self._last_aud_destut_times = aud_times
        self._last_processed_queue_times = (pq_enter, pq_exit)
        return conc, times


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
                self.out_file.flush()  # flush immediately so partial transcripts survive crashes

    def flush_tts_group(self):
        '''Force any buffered grouped TTS text into the queue.'''

        # If empty
        if not self.tts_text_buffer:
            return

        full_text = " ".join(self.tts_text_buffer).strip()

        # If empty after stripping
        if not full_text:
            return
        
        if ARTIFIC_INTON and full_text[-1] not in TTS_END_PUNCT:
            full_text += "."    # artificial intonation PERIOD - pause

        # Send the rest without waiting for grouping conditions
        grouped_o = (self.tts_group_beg, self.tts_group_end, full_text)

        rec = self._make_latency_record()
        self.tts_queue.put((grouped_o, rec))
        logger.info(f'\n\n\n[FLUSH] Flushed final TTS grouped chunk: {full_text!r}\n\n\n')

        # Reset all the necessary stuff
        self.tts_text_buffer = []
        self.tts_buffer_start_time = None
        self.tts_group_beg = None
        self.tts_group_end = None
        self.tts_group_start_perf = None
        self.tts_buffer_ids = []

    def _make_latency_record(self):
        '''Builds a LatencyRecord for the current group, averaging STT times across all chunks in the group.'''
        ids = self.tts_buffer_ids

        # For the STT times not already in the latency records, calculate the average synth and destutter times across all chunks in the group and include those averages in the LatencyRecord for the group
        # Use vals in STT dicts
        stt_starts = [self.stt_synth_times[cid][0] for cid in ids if cid in self.stt_synth_times]
        stt_ends = [self.stt_synth_times[cid][1] for cid in ids if cid in self.stt_synth_times]
        stt_destut_starts = [self.stt_destut_times[cid][0] for cid in ids if cid in self.stt_destut_times]
        stt_destut_ends = [self.stt_destut_times[cid][1] for cid in ids if cid in self.stt_destut_times]
        aud_destut_starts = [self.aud_destut_times[cid][0] for cid in ids if cid in self.aud_destut_times]
        aud_destut_ends   = [self.aud_destut_times[cid][1] for cid in ids if cid in self.aud_destut_times]
        pq_enters = [self.processed_queue_times[cid][0] for cid in ids if cid in self.processed_queue_times]
        pq_exits  = [self.processed_queue_times[cid][1] for cid in ids if cid in self.processed_queue_times]

        # Add to latency record obj
        rec = LatencyRecord(
            chunk_id = self.tts_group_beg,
            group_start_perf = self.tts_group_start_perf,
            buffer_enter = self.tts_buffer_start_time,
            tts_queue_enter = time.perf_counter(),
            stt_synth_start = sum(stt_starts) / len(stt_starts) if stt_starts else None,
            stt_synth_end = sum(stt_ends) / len(stt_ends) if stt_ends else None,
            stt_destut_start = sum(stt_destut_starts) / len(stt_destut_starts) if stt_destut_starts else None,
            stt_destut_end = sum(stt_destut_ends) / len(stt_destut_ends) if stt_destut_ends else None,
            aud_destut_start = sum(aud_destut_starts) / len(aud_destut_starts) if aud_destut_starts else None,
            aud_destut_end   = sum(aud_destut_ends)   / len(aud_destut_ends)   if aud_destut_ends   else None,
            processed_queue_enter = sum(pq_enters) / len(pq_enters) if pq_enters else None,
            processed_queue_exit  = sum(pq_exits)  / len(pq_exits)  if pq_exits  else None,
        )

        # Clean up the per-chunk dicts to avoid unbounded growth
        for cid in ids:
            self.stt_synth_times.pop(cid, None)
            self.stt_destut_times.pop(cid, None)
            self.aud_destut_times.pop(cid, None)
            self.processed_queue_times.pop(cid, None)

        return rec

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
        self.tts_buffer_ids.append(o[0])

        full_text = " ".join(self.tts_text_buffer).strip()  # add to text buffer
        word_count = len(full_text.split())

        # Determine whether conditions for grouping / pushing are met
        ends_cleanly = len(text) > 0 and text[-1] in TTS_END_PUNCT
        waited_too_long = (now - self.tts_buffer_start_time) >= TTS_MAX_WAIT_SEC

        if ends_cleanly or waited_too_long:
            if ARTIFIC_INTON and full_text and full_text[-1] not in TTS_END_PUNCT:
                full_text += "-"  # artificially comma add for better intonation in TTS - "continuation" not pause

            grouped_o = (self.tts_group_beg, self.tts_group_end, full_text)

            # Create LatencyRecord obj for this group using STT times
            rec = self._make_latency_record()
            self.tts_queue.put((grouped_o, rec))
            logger.info(f'TTS grouped chunk queued: {full_text!r}')

            self.tts_text_buffer = []
            self.tts_buffer_start_time = None
            self.tts_group_beg = None
            self.tts_group_end = None
            self.tts_group_start_perf = None
            self.tts_buffer_ids = []


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

            if o[0] is not None:  # if audio is not blank

                # Store STT synth times keyed by segment ID for later inclusion in the group's LatencyRecord
                self.stt_synth_times[o[0]] = (stt_synth_t0, stt_synth_t1)

                # Now that we have o[0], key the pre-STT audio destutter times too
                if self._last_aud_destut_times is not None:
                    self.aud_destut_times[o[0]] = self._last_aud_destut_times
                    self._last_aud_destut_times = None

                # Key the processed queue wait times too
                if self._last_processed_queue_times is not None:
                    self.processed_queue_times[o[0]] = self._last_processed_queue_times
                    self._last_processed_queue_times = None

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
                    self.stt_destut_times[o[0]] = (t0, t1)


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
                    logger.info(f'\n\n\n[TIMEOUT FLUSH] Silence timeout of {self.SILENCE_TIMEOUT}s reached. Flushing buffer.\n\n\n')
                    self.flush_tts_group()

        if TTS_GROUPING_ENABLED:  # flush if leftover stuff left in tts buffer
            self.flush_tts_group()

        # Stop background threads — _destutter_loop will drain the deque then
        # send a None sentinel to _processed_queue automatically
        self._receive_thread_running = False

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


def split_into_subphrases(text: str) -> list:
    '''Splits a TTS text chunk into sub-phrases at natural prosody boundaries.
    Splits on .?! and mid-sentence pauses (,;:—) while keeping the punc attached to the preceding phrase'''
    import re
    MIN_SUBPHRASE_WORDS = 2

    # Split after any of these characters, keeping the delimiter on the left piece
    parts = re.split(r'(?<=[.?!,;:\-—])\s+', text.strip())

    # Merge fragments that are too short into the next part
    merged = []
    carry = ''
    for part in parts:
        combined = (carry + ' ' + part).strip() if carry else part
        if len(combined.split()) <= MIN_SUBPHRASE_WORDS and part != parts[-1]:
            carry = combined  # too short, carry forward
        else:
            merged.append(combined)
            carry = ''
    if carry:
        # Leftover carry at end — append to last phrase or as its own
        if merged:
            merged[-1] = (merged[-1] + ' ' + carry).strip()
        else:
            merged.append(carry)

    return merged if merged else [text]


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
        self.rvc_queue = queue.Queue()  # queue for TTS audio to be sent to RVC if RVC_FLAG is on
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


    def warmup_models(self, rvc_queue_obj):
        '''Sends first signal to warm up TTS and RVC models (STT already warms itself up)'''
        logger.info("Warming up TTS and RVC models...")
        
        # Warm up TTS
        warmup_text = 'Warmup'
        try:
            wav = synthesize_text(warmup_text)
            wav = np.array(wav)
            
            # Warm up RVC
            if RVC_FLAG:
                # Pass a dummy rec with chunk_id warmup so finalize_and_send knows to ignore it
                dummy_rec = LatencyRecord(chunk_id='warmup', group_start_perf=time.perf_counter())
                rvc_queue_obj.put((wav, dummy_rec))
                
        except Exception as e:
            logger.error(f"Warmup failed: {e}")


    def handle_tts_client(self, client, tts_queue):
        '''Handles one TTS client connection
        Manages the TTS pipeline with optional parallel RVC processing'''
        tts_flag = True  # set the TTS flag to true when a TTS client connects to the server
        #client_type = client['type']
        conn = client['conn/socket']

        print("TTS client connected.")
        
        # Helper for cleanup and sending code
        def finalize_and_send(wav, rec):
            '''Helper for cleanup and sending code + records overall latency'''

            # If this is warmup data, just exit early
            if rec is None or rec.chunk_id == 'warmup':
                logger.info("Warmup chunk processed successfully.")
                return

            # Convert to 16-bit PCM
            wav_pcm = (wav * 32767).astype(np.int16)

            # Stamp the moment audio is sent and hand record to tracker
            rec.audio_sent = time.perf_counter()
            tracker.add(rec)

            # Remove start time for this segment from the start_times dictionary
            # But clean up only if that key exists in the dict
            if rec.chunk_id in start_times:
                del start_times[rec.chunk_id]
           
            # Send the packet of audio data
            logger.debug("Sending audio to TTS client...")
            conn.sendall(wav_pcm.tobytes())
            logger.debug("Audio sent to TTS client.")
       

        # RVC parallel thread
        def rvc_worker_loop():
            try:
                while True:
                    item = self.rvc_queue.get()
                    if item is None: 
                        break
                    
                    wav, rec = item

                    # Stamp RVC queue exit and start of RVC processing
                    rec.rvc_queue_exit = time.perf_counter()
                    tr0 = time.perf_counter()
                    
                    new_aud = rvc_converter.vc(wav)  # outputs 2D array, need to squeeze to 1D
                            
                    if new_aud.shape[0] == 0:
                        # In case RVC returns empty because the audio < block frames and is too short (non-ideal), preserve original speech
                        logger.warning("[RVC] RVC returned empty, preserving original.")
                    else:
                        wav = new_aud.squeeze(axis=1)  # squeeze to 1D
                        logger.debug("[RVC] TTS audio processed through RVC.")

                    tr1 = time.perf_counter()
                    rec.rvc_start = tr0
                    rec.rvc_end   = tr1
                    logger.info(f'[LATENCY] RVC processing took {tr1 - tr0:.3f}s')

                    finalize_and_send(wav, rec)

            except (BrokenPipeError, ConnectionResetError):
                logger.info("broken pipe / connection reset (RVC worker) -- connection closed?")
            
            except Exception as e:
                logger.error(f"RVC Thread Error: {e}")

            finally:
                self.rvc_queue.task_done()

        if RVC_FLAG:
            Thread(target=rvc_worker_loop, daemon=True).start()

        
        self.warmup_models(self.rvc_queue)  # warm up

        try:
            while True:
                # As long as queue is not empty, get text from queue, convert it to audio, and send it over
                try:
                    # queue contents = (grouped_o, rec)
                    (o, rec) = tts_queue.get(timeout=0.1)  # this should also remove it from the queue; block for 0.1 to free up CPU
                
                except queue.Empty:
                    continue

                try:
                    # Stamp TTS queue exit
                    rec.tts_queue_exit = time.perf_counter()
                    logger.info(f'[LATENCY] Total time in buffer & TTS queue: {rec.buffer_and_queue_dur:.3f}s')

                    logger.debug("o received from TTS queue.")

                    text = o[2]

                    # ========= Sub-phrase splitting for pipelined TTS --> RVC ========= 
                    # Split text on natural phrase boundaries so the first sub-phrase reaches the client after ~0.4s instead of waiting for the full chunk (~1.6s TTS + ~1.2s RVC)
                    # Each sub-phrase flows through TTS --> resample --> RVC independently, overlapping with the next sub-phrase's TTS synthesis.
                    sub_phrases = split_into_subphrases(text)
                    logger.info(f'[SUB-PHRASE] Split {text!r} into {len(sub_phrases)} sub-phrases: {sub_phrases}')

                    # Only the first sub-phrase gets the latency record so end-to-end timing reflects when the user first hears audio
                    # Subsequent sub-phrases share the same group_start_perf but get lightweight records so the tracker isn't skewed
                    is_first_subphrase = True

                    for sub_text in sub_phrases:
                        if not sub_text.strip():
                            continue

                        # Generate speech for this sub-phrase
                        logger.debug(f"GENERATING TTS audio for sub-phrase: {sub_text!r}")
                        tts_synth_t0 = time.perf_counter()
                        wav = synthesize_text(sub_text)
                        tts_synth_t1 = time.perf_counter()
                        logger.info(f'[LATENCY] TTS synthesis took {tts_synth_t1 - tts_synth_t0:.3f}s for {sub_text!r}')

                        wav = np.array(wav)

                        # Resample to 16kHz if needed
                        resample_t0 = time.perf_counter()
                        if TTS_SR != SAMPLING_RATE:
                            wav = librosa.resample(wav, orig_sr=TTS_SR, target_sr=SAMPLING_RATE)
                        resample_t1 = time.perf_counter()

                        # Build a latency record for this sub-phrase.
                        # First sub-phrase inherits the full rec (with all upstream
                        # timings already stamped). Later sub-phrases get a fresh
                        # record sharing the same group_start_perf so end-to-end
                        # is still measured from the original audio arrival.
                        if is_first_subphrase:
                            sub_rec = rec
                            is_first_subphrase = False
                        else:
                            sub_rec = LatencyRecord(
                                chunk_id        = rec.chunk_id,
                                group_start_perf= rec.group_start_perf,
                                buffer_enter    = rec.buffer_enter,
                                tts_queue_enter = rec.tts_queue_enter,
                                tts_queue_exit  = rec.tts_queue_exit,
                            )

                        sub_rec.tts_synth_start = tts_synth_t0
                        sub_rec.tts_synth_end   = tts_synth_t1
                        sub_rec.resample_start  = resample_t0
                        sub_rec.resample_end    = resample_t1

                        # Send to RVC (or directly finalize if RVC disabled)
                        if RVC_FLAG:
                            min_samples = 2000  # ~0.125s at 16kHz
                            if wav.shape[0] >= min_samples:
                                sub_rec.rvc_queue_enter = time.perf_counter()
                                self.rvc_queue.put((wav, sub_rec))
                            else:
                                logger.debug(f"[RVC] Skipping RVC for short sub-phrase audio ({wav.shape[0]} samples)")
                                finalize_and_send(wav, sub_rec)
                        else:
                            finalize_and_send(wav, sub_rec)

                except (BrokenPipeError, ConnectionResetError):
                    logger.info("broken pipe / connection reset -- connection closed?")
                    raise

                finally:
                    self.tts_queue.task_done()

        except (KeyboardInterrupt, BrokenPipeError, ConnectionResetError):
            logger.info("TTS client handler stopping...")

        finally:
            if RVC_FLAG:
                self.rvc_queue.put(None)
                # Finish last bit of RVC before closing
                self.rvc_queue.join()
            conn.close()
            Server.Clients.remove(client)  # remove client from client list
            logger.info("Connection to tts client closed")


    def handle_stt_client(self, client, tts_queue):
        '''Handles one STT client connection'''
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

        # Wait for TTS queue (and by extension the RVC queue) to fully drain before
        # reporting stats. Without this, in-flight TTS/RVC chunks haven't been added
        # to the tracker yet and the stats file ends up empty.
        logger.info('Waiting for TTS/RVC pipeline to finish draining...')
        tts_queue.join()
        logger.info('TTS queue drained.')

        # Ensure output directories exist for all three output files
        for _path in [AUD_DESTUT_OUTPUT_PATH, TRANSCRIPT_PATH, STATS_PATH]:
            _dir = os.path.dirname(_path)
            if _dir:
                os.makedirs(_dir, exist_ok=True)

        # Save post-audio-destutter audio for inspection
        if SAVE_AUD_DESTUT_OUTPUT and proc.aud_destut_chunks:
            try:
                all_audio = np.concatenate(proc.aud_destut_chunks)
                soundfile.write(AUD_DESTUT_OUTPUT_PATH, all_audio, SAMPLING_RATE)
                logger.info(f'Saved post-audio-destutter audio to {AUD_DESTUT_OUTPUT_PATH}')
            except Exception as e:
                logger.error(f'Failed to save aud_destut_output: {e}')

        txt_wer = None
        if SAVE_TRANSCRIPT:
            try:
                out_file.close()
                logger.info('Transcript file written.')
                txt_wer = calc_wer(TRANSCRIPT_PATH, reference_file)
                logger.info(f"WER: {txt_wer:.3f}")
            except Exception as e:
                logger.error(f'Failed to close/score transcript: {e}')

        # Write WER to stats file first, then let tracker append latency stats
        try:
            with open(STATS_PATH, 'w') as stats:
                if txt_wer is not None:
                    stats.write(f"WER: {txt_wer:.3f}\n")
            tracker.report_and_reset(stats_path=STATS_PATH)
        except Exception as e:
            logger.error(f'Failed to write stats: {e}')





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