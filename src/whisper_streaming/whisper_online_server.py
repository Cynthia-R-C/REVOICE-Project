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
import math
from scipy import signal  # for the resample-stretch before RVC

# ======= Calculating WER and latency ======= #
import time
from jiwer import wer
reference_file = 'english_patient.txt'  # reference text for WER calculation

start_times = {}  # dict of average start time of current text segment
# key is segment ID (segment start time)
# value is average perf counter time of receiving the audio chunks

from latency_tracking import LatencyRecord, LatencyTracker
tracker = LatencyTracker()

# ======= Session Logging ======= #
class Log:
    '''Duplicates every terminal output to also a log file (wrapping sys.stdout/sys.stderr), writes flush immediately (not buffered until close) so the log file stays valid in case of killing mid session'''

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

    def isatty(self):
        # Some libraries (e.g. tqdm) check this
        return hasattr(self.streams[0], 'isatty') and self.streams[0].isatty()


def start_session_logging(log_path):
    '''Redirects stdout and stderr so everything printed also goes to log_path'''
    log_file = open(log_path, 'a', encoding='utf-8', buffering=1)  # line-buffered
    sys.stdout = Log(sys.stdout, log_file)
    sys.stderr = Log(sys.stderr, log_file)
    return log_file


# ======= DESTUTTERING IMPORTS/CONSTANTS ======= #
import sys
destutter_dir = os.path.abspath('C:\\Users\\crc24\\Documents\\VS_Code_Python_Folder\\ScienceFair2025\\src\\destutter')
sys.path.append(destutter_dir)  # add destutter folder to paths to search
from destutterer import Destutterer

SAMPLING_RATE = 16000
CLASS_MODEL = 'ConvLSTM'  # 'ConvLSTM' or 'StutterNet'

if CLASS_MODEL == 'StutterNet':
    CONFIG_PATH = 'C:\\Users\\crc24\\Documents\\VS_Code_Python_Folder\\ScienceFair2025\\interspeech2024-code\\sed\\examples\\stutter_event\\s0\\conf\\train_stutternet.yaml'
    CKPT_PATH = 'C:\\Users\\crc24\\Documents\\VS_Code_Python_Folder\\ScienceFair2025\\interspeech2024-code\\exp\\stutternet_en\\36.pt'
    CMVN_PATH = 'C:\\Users\\crc24\\Documents\\VS_Code_Python_Folder\\ScienceFair2025\\interspeech2024-code\\data\\train\\stutternet_global_cmvn'
elif CLASS_MODEL == 'ConvLSTM':
    CONFIG_PATH = 'C:\\Users\\crc24\\Documents\\VS_Code_Python_Folder\\ScienceFair2025\\interspeech2024-code\\sed\\examples\\stutter_event\\s0\\conf\\train_convlstm.yaml'
    CKPT_PATH = 'C:\\Users\\crc24\\Documents\\VS_Code_Python_Folder\\ScienceFair2025\\interspeech2024-code\\exp\\convlstm_en\\99.pt'
    CMVN_PATH = 'C:\\Users\\crc24\\Documents\\VS_Code_Python_Folder\\ScienceFair2025\\interspeech2024-code\\data\\train\\global_cmvn'
else:
    raise ValueError(f"CLASS_MODEL must be 'StutterNet' or 'ConvLSTM', not {CLASS_MODEL}")

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
GROUP = 'convlstm'
# TRIAL = '1'  #  will just have to manually rename the file as I test unless I wanna stop the server and restart it just to update the constant in file naming and that’s not worth it
TRANSCRIPT_PATH = f'test_results/{GROUP}/transcript.txt'
STATS_PATH = f'test_results/{GROUP}/stats.txt'
AUD_DESTUT_OUTPUT_PATH = f'test_results/{GROUP}/aud_destut_output.wav'


# ===== Adaptive Slowdown ===== #
PITCH_STRETCH_ENABLED = True
STRETCH_MIN = 1.0    # healthy buffer estimate -> no stretch
STRETCH_MAX = 1.18   # low buffer estimate -> stretch this much to buy time
STRETCH_LOW_WATER_SECS = 1.5
STRETCH_HIGH_WATER_SECS = 4.0
CLIENT_PREBUFFER_SECS = 3.5  # MUST match PREBUFFER_SECS in coqui_realtime_client.py - can't see the client's actual buffer so estimate it off own send timestamps + this constant


# ======= Other Toggles ======= #
SAVE_TRANSCRIPT = True
tts_flag = False  # becomes true when a TTS client connects to the server
RVC_FLAG = False   # choose whether to enable RVC or not
TXT_DESTUT = True # whether or not to do text destuttering
AUD_DESTUT = True  # whether or not to do audio 
SAVE_AUD_DESTUT_OUTPUT = True  # save post-audio-destutter audio to a wav for inspection
USE_MODEL_FOR_TXT = False   # whether to use the model for text destuttering at all

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
MELO_SPEAKER = 'EN-US'
MELO_SPEED = 0.8

TTS_GROUPING_ENABLED = True
ARTIFIC_INTON = True   # whether to add a fallback dash or period when the punctuation model does not find a real ending in time
TTS_MAX_WAIT_SEC = 2.0   # backup cutoff for a long run the punc model hasn't found a real ending for yet
# keep above SILENCE_TIMEOUT so real pauses win over this timer
FLUSH_TIMER_ENABLED = False  # whether to run the background TTS flush timer loop
SILENCE_TIMEOUT = 1.1   # Seconds of silence before forcing flush, uses the punctuation model on whatever is left over


# Main function within if __name__ == '__main__' to prevent infinite process spawning on Windows
def main():
    # Use global keywords so the rest of script can see these objects
    global args, asr, online, base_online, rvc_converter, min_chunk, size, language, BASE_PITCH
    global tts, TTS_SR, destutterer_stt, melo_speaker_ids
    
 
    # ======= Logging and Arguments ======= #
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser()

    parser.add_argument("--host", type=str, default='localhost')
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--warmup-file", type=str, dest="warmup_file", 
            help="The path to a speech audio wav file to warm up Whisper so that the very first chunk processing is fast. It can be e.g. https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav .")
    parser.add_argument("--log-file", type=str, default=None,
            help="Path to save a full copy of session terminal output")

    # options from whisper_online
    add_shared_args(parser)
    args = parser.parse_args()

    log_path = args.log_file or f"test_results/{GROUP}/session_log_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    start_session_logging(log_path)
    print(f"[SESSION LOG] Saving full terminal output to: {os.path.abspath(log_path)}")

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


    # ======= Set Up Punctuation Model ======= #
    # for fixing prosody output
    from deepmultilingualpunctuation import PunctuationModel
    global punct_model
    punct_model = PunctuationModel()

    # ======= Set Up Destutterers ======= #
    destutterer_stt = Destutterer(config_path=CONFIG_PATH,
                          ckpt_path=CKPT_PATH,
                          cmvn_path=CMVN_PATH,
                          sr=SAMPLING_RATE,
                          device=device)
    # Note: destutterer_tts removed, audio destut now happens pre-STT in receive_audio_chunk via destutterer_stt.aud_destutter_chunk()


    # ======= Set Up RVC ======= #
    from rvc_conversion import RVC
    rvc_converter = RVC()
    BASE_PITCH = rvc_converter.gui.gui_config.pitch  # whatever pitch shift is already configured (currently -9), stretch compensation stacks on top of this




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
        self.punct_queue = queue.Queue()  # queue for tts buffering + punctuation model work, runs on its own thread

        self.last_end = None

        self.is_first = True

        # For TTS grouping buffer
        self.tts_text_buffer = []
        self.tts_buffer_start_time = None      # perf_counter when this group started buffering
        self.tts_group_beg = None              # transcript timestamp of first chunk in group
        self.tts_group_end = None              # transcript timestamp of latest chunk in group
        self.tts_group_start_perf = None       # perf_counter corresponding to first chunk's arrival
        self.tts_buffer_ids = []               # o[0] keys of every chunk added to the current group
        self.tts_buffer_ends = []              # o[1] end timestamp for every chunk added, lines up with tts_buffer_ids

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
        if FLUSH_TIMER_ENABLED:
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
        '''Background thread: assembles min_chunk-sized pieces from _audio_deque,
        runs aud_destutter_chunk (slow StutterNet) off the main thread, then puts
        results into _processed_queue for process() to consume.
        Sends a None sentinel when the stream ends.'''
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
                # Stream is ending, flush audio destutter buffer
                if AUD_DESTUT:
                    final_conc = destutterer_stt.flush_aud_buffer()
                    if len(final_conc) > 0:
                        if SAVE_AUD_DESTUT_OUTPUT:
                            self.aud_destut_chunks.append(final_conc.copy())
                        self._processed_queue.put((final_conc, times, None, time.perf_counter()))
                self._processed_queue.put(None)
                break

            conc = np.concatenate(out)

            if is_first_local and len(conc) < minlimit:
                continue
            is_first_local = False

            # Audio destuttering runs HERE, off the main thread.

            # There used to be a backpressure check of skipping chunks when Whisper was behind; however after adding in rolling buffer for audio destuttering, aud destut is no longer stateless, so this must be run in order
            
            aud_times = None
            if AUD_DESTUT:
                t0 = time.perf_counter()
                conc = destutterer_stt.aud_destutter_chunk(conc)
                t1 = time.perf_counter()
                aud_times = (t0, t1)

            if SAVE_AUD_DESTUT_OUTPUT:
                self.aud_destut_chunks.append(conc.copy())

            # aud_destutter_chunk() now holds some audio back internally as a rolling lookahead margin, so conc could come back empty
            # Skip pushing an empty item downstream
            if len(conc) == 0:
                continue

            pq_enter = time.perf_counter()
            # Always push every chunk, never merge or drop: merging oldest waiting item causes data loss bc the other queued items are skipped by Whisper's internal cursor
            self._processed_queue.put((conc, times, aud_times, pq_enter))

    def receive_audio_chunk(self):
        '''Pulls the next processed chunk from _processed_queue.
        Uses a short timeout loop so it can notice when the stream ends even if
        the sentinel arrives slightly late (avoids blocking forever).'''
        while True:
            try:
                item = self._processed_queue.get(timeout=0.1)
            except queue.Empty:
                # No item yet — check if the stream is fully done
                if not self._receive_thread_running and self._processed_queue.empty():
                    return None, None
                continue

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
                self.punct_queue.put(('commit', o))  # hand off to the punctuation worker thread
            else:
                self.put_to_tts(o)
        
            if SAVE_TRANSCRIPT:
                # Write the text to the output transcript file
                self.out_file.write(o[2])
                self.out_file.flush()  # flush immediately so partial transcripts survive crashes

    def _punct_worker_loop(self):
        '''Runs tts buffering + punctuation model on its own thread, keeps whisper from waiting on it'''
        while True:
            item = self.punct_queue.get()
            if item is None:
                self.punct_queue.task_done()
                break
            kind, payload = item
            try:
                if kind == 'commit':
                    self.group_and_put_to_tts(payload)
                elif kind == 'check_silence':
                    now = time.perf_counter()
                    if self.tts_text_buffer and (now - self.last_text_received_time) > self.SILENCE_TIMEOUT:
                        logger.info(f'\n\n\n[TIMEOUT FLUSH] Silence timeout of {self.SILENCE_TIMEOUT}s reached. Flushing buffer.\n\n\n')
                        self.flush_tts_group()
                elif kind == 'final_flush':
                    if TTS_GROUPING_ENABLED:
                        self.flush_tts_group()
            except Exception as e:
                # one bad item shouldn't kill this thread for rest of session
                logger.error(f'Punctuation worker failed on this item, skipping it and continuing: {e}')
            finally:
                self.punct_queue.task_done()

    def flush_tts_group(self):
        '''Force any buffered grouped TTS text into the queue.'''

        # If empty
        if not self.tts_text_buffer:
            return

        self._flush_group_through(len(self.tts_text_buffer) - 1, time.perf_counter(), tag='[FLUSH] Flushed final TTS grouped chunk')


    def _punct_break_index(self, full_text):
        '''Asks the punctuation model where a real sentence ending falls, returns the (fragment index, labeled words) or (None, None)'''
        words = full_text.split()
        try:
            clean = punct_model.preprocess(full_text)
            labeled = punct_model.predict(clean)
        except Exception as e:
            logger.warning(f'Punctuation model failed, fall back on the backup timer: {e}')
            return None, None

        if len(labeled) != len(words):
            return None, None  # don't try to line up for now

        # figure out which word index each buffered fragment ends on
        frag_word_end = []
        count = 0
        for frag in self.tts_text_buffer:
            count += len(frag.split())
            frag_word_end.append(count - 1)

        for frag_idx, word_idx in enumerate(frag_word_end):
            if labeled[word_idx][1] in '.?':
                return frag_idx, labeled
        return None, labeled

    def _join_labeled(self, labeled):
        '''Turns (word, label, score) tuple back into one punctuated string, same join logic the model package uses'''
        result = ""
        for word, label, _ in labeled:
            result += word
            if label == "0":
                result += " "
            if label in ".,?-:":
                result += label + " "
        return result.strip()

    def _flush_group_through(self, last_idx, now, labeled=None, fallback_char='.', tag='TTS grouped chunk queued'):
        '''Flushes buffered fragments 0 through last_idx, keeps anything after that buffered for next time. 
        Param labeled = reuse the preductions with more context'''
        flush_text = " ".join(self.tts_text_buffer[:last_idx + 1]).strip()
        if not flush_text:
            return

        flush_ids = self.tts_buffer_ids[:last_idx + 1]
        flush_ends = self.tts_buffer_ends[:last_idx + 1]

        if labeled is not None:
            word_count = sum(len(frag.split()) for frag in self.tts_text_buffer[:last_idx + 1])
            punctuated = self._join_labeled(labeled[:word_count])
        else:
            # restore_punctuation already preprocesses the text itself, feeding it already preprocessed text was the bug that crashed the whole stt thread last run
            try:
                punctuated = punct_model.restore_punctuation(flush_text)
            except Exception as e:
                logger.warning(f'Punctuation model failed on final flush, sending the raw text instead: {e}')
                punctuated = flush_text

        if ARTIFIC_INTON and punctuated and punctuated[-1].isalnum():
            punctuated += fallback_char  # the model left this ending on a plain word, so fall back on this instead

        group_start_perf = start_times.get(flush_ids[0], now)
        grouped_o = (flush_ids[0], flush_ends[-1], punctuated)
        rec = self._make_latency_record(ids=flush_ids, chunk_id=flush_ids[0], group_start_perf=group_start_perf, buffer_enter=self.tts_buffer_start_time)
        self.tts_queue.put((grouped_o, rec))
        logger.info(f'{tag}: {punctuated!r}')

        # keep whatever is left over buffered for the next round instead of throwing it away
        self.tts_text_buffer = self.tts_text_buffer[last_idx + 1:]
        self.tts_buffer_ids = self.tts_buffer_ids[last_idx + 1:]
        self.tts_buffer_ends = self.tts_buffer_ends[last_idx + 1:]

        if self.tts_text_buffer:
            self.tts_buffer_start_time = now
            self.tts_group_beg = self.tts_buffer_ids[0]
            self.tts_group_start_perf = start_times.get(self.tts_buffer_ids[0], now)
            self.tts_group_end = self.tts_buffer_ends[-1]
        else:
            self.tts_buffer_start_time = None
            self.tts_group_beg = None
            self.tts_group_end = None
            self.tts_group_start_perf = None

    def _make_latency_record(self, ids=None, chunk_id=None, group_start_perf=None, buffer_enter=None):
        '''Builds a LatencyRecord for the current group, averaging STT times across all chunks in the group.'''
        if ids is None:
            ids = self.tts_buffer_ids
        if chunk_id is None:
            chunk_id = self.tts_group_beg
        if group_start_perf is None:
            group_start_perf = self.tts_group_start_perf
        if buffer_enter is None:
            buffer_enter = self.tts_buffer_start_time

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
            chunk_id = chunk_id,
            group_start_perf = group_start_perf,
            buffer_enter = buffer_enter,
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
        '''Does prosody-based group enqueuing + breaks based on punctuation model'''
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
        self.tts_buffer_ends.append(o[1])

        full_text = " ".join(self.tts_text_buffer).strip()  # add to text buffer

        # ask the punctuation model where sentence endings are
        punct_t0 = time.perf_counter()
        break_idx, labeled = self._punct_break_index(full_text)
        punct_t1 = time.perf_counter()
        logger.info(f'[LATENCY] Punctuation check took {punct_t1 - punct_t0:.3f}s on {len(full_text.split())} words')

        waited_too_long = (now - self.tts_buffer_start_time) >= TTS_MAX_WAIT_SEC

        # only trust a break once at least one more real fragment exists past it, aka the model already had a chance to extend the sentence into that fragment and chose not to
        if break_idx is not None and break_idx < len(self.tts_text_buffer) - 1:
            self._flush_group_through(break_idx, now, labeled=labeled)

        elif waited_too_long:
            self._flush_group_through(len(self.tts_text_buffer) - 1, now, fallback_char='-')


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
        # clear audio buffer for new session
        destutterer_stt.reset_aud_buffer()

        # this thread does the tts buffering + punctuation work off the main loop
        Thread(target=self._punct_worker_loop, daemon=True).start()

        while True:
            a, startTimes = self.receive_audio_chunk()
            if a is None:
                break
            self.online_asr_proc.insert_audio_chunk(a)
            stt_synth_t0 = time.perf_counter()
            o = online.process_iter()   # o[0]: beg, o[1]: end, o[2]: text string

            now = time.perf_counter()

            # o[0] is None both when the person is truly quiet and when whisper just has not agreed on new words yet, those are not the same thing, so check the actual gate instead of guessing off o[0]
            if self.online_asr_proc.energy_gate.is_open:
                self.last_text_received_time = now

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

                # ============== STT DESTUTTERING LOGIC ================ #

                if TXT_DESTUT:

                    # Track destuttering logic start time
                    t0 = time.perf_counter()

                    t_to_buffer = base_online.buffer_time_offset  # time between global stream start and audio buffer start
                    audio_buffer = base_online.audio_buffer  # current audio buffer - a list of samples
                    beg_time = o[0]  # beg time of text (global)
                    end_time = o[1]  # end time of text (global)
                    text = o[2]      # text of current segment
                    
                    if USE_MODEL_FOR_TXT:
                        t_maxs, stutter_word_idxs = destutterer_stt.get_destutter_info('stt', t_to_buffer, audio_buffer, beg_time, end_time, text)  # get t_maxs and stutter word indices
                    else:
                        # get_destutter_info() is the only place that normally sets destutterer_stt.text etc., so manually set it here
                        destutterer_stt.text = text
                        destutterer_stt.words = text.split()
                        destutterer_stt.beg_time = beg_time
                        destutterer_stt.end_time = end_time
                    
                    ## Simple: SOUND REP ##
                    destutterer_stt.r_destutter()

                    ## Medium: WORD REP ##
                    # Only run this if word rep detected in model(?) might not be necessary, could just run this without model
                    if USE_MODEL_FOR_TXT:
                        if t_maxs['/wr'] is not None:
                            destutterer_stt.wr_destutter()
                    else:
                        destutterer_stt.wr_destutter()

                    ## Complex: INTERJECTIONS ##
                    if USE_MODEL_FOR_TXT:
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

                except Exception as e:
                    # one bad chunk shouldn't kill transcription for rest of session, log and continue
                    logger.error(f'send_result failed on this chunk, skipping it and continuing: {e}')

            else:
                if TTS_GROUPING_ENABLED:
                    self.punct_queue.put(('check_silence', None))  # worker thread checks its own buffer state

        # Stop background threads — _destutter_loop will drain the deque then
        # send a None sentinel to _processed_queue automatically
        self._receive_thread_running = False

        o = online.finish()  # this should be working
        self.send_result(o)  # flush comes after this not before bc this can still add more

        if TTS_GROUPING_ENABLED:
            self.punct_queue.put(('final_flush', None))

        self.punct_queue.join()   # wait for worker to catch up on stuff queued above
        self.punct_queue.put(None)   # stop signal

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


def stretch_audio_for_rvc(wav, stretch):
    '''Resamples wav to change its duration by stretch (>1 = slower)'''
    if abs(stretch - 1.0) < 0.02:
        return wav  # basically no stretch, don't bother
    in_frames = len(wav)
    out_frames = max(1, int(round(in_frames * stretch)))
    g = np.gcd(in_frames, out_frames)
    stretched = signal.resample_poly(wav.astype(np.float32), out_frames // g, in_frames // g)
    return stretched.astype(np.float32)


def get_pitch_correction(stretch):
    '''How many semitones to shift up to undo the pitch drop from stretch_audio_for_rvc.
    Derivation: stretching duration drops pitch to a ratio of 1/stretch, so correction = 12*log2(stretch) semitones'''
    if abs(stretch - 1.0) < 0.02:
        return 0.0
    return 12 * math.log2(stretch)


def estimate_client_buffer_secs(buffer_state):
    '''Guesses how much audio is sitting in the client's playout buffer right now'''
    if buffer_state['stream_start_perf'] is None:
        return 0.0  # haven't sent anything yet this session
    elapsed = time.perf_counter() - buffer_state['stream_start_perf']
    played_est = max(0.0, elapsed - CLIENT_PREBUFFER_SECS)
    return max(0.0, buffer_state['total_sent_secs'] - played_est)


def stretch_factor_for_depth(depth_secs):
    '''Calculates what stretch factor is needed based on the estimated buffer depth'''
    if depth_secs <= STRETCH_LOW_WATER_SECS:
        return STRETCH_MAX
    if depth_secs >= STRETCH_HIGH_WATER_SECS:
        return STRETCH_MIN
    frac = (depth_secs - STRETCH_LOW_WATER_SECS) / (STRETCH_HIGH_WATER_SECS - STRETCH_LOW_WATER_SECS)
    return STRETCH_MAX + frac * (STRETCH_MIN - STRETCH_MAX)


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
                rvc_queue_obj.put((wav, dummy_rec, BASE_PITCH))  # no stretch for warmup, just base pitch
                
        except Exception as e:
            logger.error(f"Warmup failed: {e}")


    def handle_tts_client(self, client, tts_queue):
        '''Handles one TTS client connection
        Manages the TTS pipeline with optional parallel RVC processing'''
        tts_flag = True  # set the TTS flag to true when a TTS client connects to the server
        #client_type = client['type']
        conn = client['conn/socket']

        print("TTS client connected.")

        # tracks how much audio we've sent + when we started, used to estimate
        # the client's buffer depth for adaptive stretching (see estimate_client_buffer_secs)
        buffer_state = {'total_sent_secs': 0.0, 'stream_start_perf': None}

        # Helper for cleanup and sending code
        def finalize_and_send(wav, rec):
            '''Helper for cleanup and sending code + records overall latency'''

            # If this is warmup data, just exit early
            if rec is None or rec.chunk_id == 'warmup':
                logger.info("Warmup chunk processed successfully.")
                return

            # Convert to 16-bit PCM
            wav_pcm = (wav * 32767).astype(np.int16)

            # update buffer_state so we can keep estimating client buffer depth
            if buffer_state['stream_start_perf'] is None:
                buffer_state['stream_start_perf'] = time.perf_counter()
            buffer_state['total_sent_secs'] += len(wav_pcm) / SAMPLING_RATE

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
                    
                    wav, rec, f0_key = item

                    # Stamp RVC queue exit and start of RVC processing
                    rec.rvc_queue_exit = time.perf_counter()
                    tr0 = time.perf_counter()

                    rvc_converter.set_pitch_key(f0_key)  # set right before vc() so that this item definitely uses it
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

                # task_done must always be called once per get(), even if we break early,
                # so that tts_queue.join() in handle_stt_client doesn't hang forever.
                try:
                    # Stamp TTS queue exit
                    rec.tts_queue_exit = time.perf_counter()
                    logger.info(f'[LATENCY] Total time in buffer & TTS queue: {rec.buffer_and_queue_dur:.3f}s')

                    logger.debug("o received from TTS queue.")

                    text = o[2]

                    # Generate speech
                    logger.debug("GENERATING TTS audio...")
                    tts_synth_t0 = time.perf_counter()

                    wav = synthesize_text(text)
                    tts_synth_t1 = time.perf_counter()
                    rec.tts_synth_start = tts_synth_t0
                    rec.tts_synth_end   = tts_synth_t1
                    logger.info(f'[LATENCY] TTS synthesis took {tts_synth_t1 - tts_synth_t0:.3f}s')

                    logger.debug("TTS audio generated from text.")
                    wav = np.array(wav)   # convert to np array to avoid memory issues

                    # Resample to 16kHz if needed
                    resample_t0 = time.perf_counter()
                    if TTS_SR != SAMPLING_RATE:
                        wav = librosa.resample(wav, orig_sr=TTS_SR, target_sr=SAMPLING_RATE)
                        logger.debug(f"Resampled TTS audio from {TTS_SR}Hz to {SAMPLING_RATE}Hz.")
                    resample_t1 = time.perf_counter()
                    rec.resample_start = resample_t0
                    rec.resample_end   = resample_t1
                    logger.info(f'[LATENCY] Resampling took {resample_t1 - resample_t0:.3f}s')

                    
                    # ============== DESTUTTERING LOGIC END ================ #
                    # (Audio destuttering for /p and /b now happens pre-STT in
                    #  receive_audio_chunk — see destutterer_stt.aud_destutter_chunk)


                    # adaptive stretch, now runs regardless of RVC_FLAG
                    f0_key = BASE_PITCH
                    if PITCH_STRETCH_ENABLED:
                        depth_est = estimate_client_buffer_secs(buffer_state)
                        stretch = stretch_factor_for_depth(depth_est)
                        wav = stretch_audio_for_rvc(wav, stretch)
                        f0_key = BASE_PITCH + get_pitch_correction(stretch)
                        logger.info(f'[STRETCH] est buffer {depth_est:.2f}s -> stretch {stretch:.3f}x -> f0_up_key {f0_key:.2f}')

                    # RVC logic (if flag enabled)
                    if RVC_FLAG:
                        # Only use RVC if audio is long enough to be meaningful
                        min_samples = 2000  # ~0.125s at 16kHz
                        if wav.shape[0] >= min_samples:
                            # Stamp RVC queue enter and put in RVC queue
                            rec.rvc_queue_enter = time.perf_counter()
                            self.rvc_queue.put((wav, rec, f0_key))
                        else:
                            logger.debug(f"[RVC] Skipping RVC for short audio ({wav.shape[0]} samples)")

                    else:
                        # run cleanup and send
                        finalize_and_send(wav, rec)

                except (BrokenPipeError, ConnectionResetError):
                    logger.info("broken pipe / connection reset -- connection closed?")
                    # task_done still called in finally below before break takes effect
                    raise  # re-raise to exit the outer while True via the except below

                finally:
                    # Always mark this item done so tts_queue.join() never hangs
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

        # Wait for TTS queue (and by extension the RVC queue) to fully drain before reporting stats
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