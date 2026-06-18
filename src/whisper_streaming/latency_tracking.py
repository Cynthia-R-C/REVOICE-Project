'''
Replacement for the scattered global-list latency system.

1. Replaces all the global lists with single import:

       from latency_tracking import LatencyRecord, LatencyTracker
       tracker = LatencyTracker()

2. One LatencyRecord obj per grouped chunk created when it first enters the pipeline (i.e. when group_start_perf is first known) and record each part of the pipeline as chunk flows through:

       rec = LatencyRecord(chunk_id=o[0], group_start_perf=group_start_perf)

    Note: STT avg values are also added in during initialization so both STT and TTS data is accounted for

3. At the end of each session call tracker.report_and_reset() to print and save averages, then the tracker resets itself automatically.
'''

from __future__ import annotations
import time
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# LatencyRecord  - one instance per grouped TTS chunk, travels with the data

@dataclass
class LatencyRecord:
    '''
    Carries every timing stamp for one chunk through the full pipeline.
    All times from time.perf_counter()

    group_start_perf  - self.tts_group_start_perf

    stt_synth_start   - just before  o = online.process_iter()
    stt_synth_end     - just after   o = online.process_iter()

    stt_destut_start  - just before STT destutter block
    stt_destut_end    - just after  STT destutter block (None if TXT_DESTUT = False)

    buffer_enter      - when the chunk first enters the TTS text buffer, 
                        i.e. self.tts_buffer_start_time in group_and_put_to_tts()

    tts_queue_enter   - time.perf_counter() right before tts_queue.put()
                        inside group_and_put_to_tts() / flush_tts_group()

    tts_queue_exit    - time.perf_counter() right after tts_queue.get()
                        in handle_tts_client()

    tts_synth_start   - just before synthesize_text()
    tts_synth_end     - just after  synthesize_text()

    tts_destut_start  - just before TTS destutter block
    tts_destut_end    - just after  TTS destutter block (None if AUD_DESTUT = False)

    rvc_queue_enter   - right before rvc_queue.put()

    rvc_queue_exit    - right after rvc_queue.get() inside rvc_worker_loop()

    rvc_start         - just before rvc_converter.vc()
    rvc_end           - just after  rvc_converter.vc() (and squeeze) (None if RVC_FLAG = False)

    audio_sent        - just before conn.sendall() (inside finalize_and_send()
    '''

    chunk_id: object          # o[0] - the segment start timestamp used as dict key

    # anchor
    group_start_perf: float  = 0.0   # when the first audio chunk of this group arrived

    # STT synthesis
    stt_synth_start: Optional[float] = None
    stt_synth_end: Optional[float] = None

    # STT destutter (optional)
    stt_destut_start: Optional[float] = None
    stt_destut_end: Optional[float] = None

    # Audio destutter pre-STT (prolongations & blocks, via aud_destutter_chunk)
    aud_destut_start: Optional[float] = None
    aud_destut_end: Optional[float] = None

    # Time chunk spent waiting in _processed_queue for Whisper to be ready
    processed_queue_enter: Optional[float] = None   # when chunk was put into _processed_queue
    processed_queue_exit: Optional[float] = None   # when receive_audio_chunk() pulled it out

    # buffer + queue
    buffer_enter: Optional[float] = None   # when text first entered the grouping buffer
    tts_queue_enter: Optional[float] = None   # just before tts_queue.put()
    tts_queue_exit: Optional[float] = None   # just after  tts_queue.get()

    # TTS synthesis
    tts_synth_start: Optional[float] = None
    tts_synth_end: Optional[float] = None

    # Resampling (between TTS synth and TTS destutter / RVC)
    resample_start: Optional[float] = None
    resample_end: Optional[float] = None

    # TTS destutter removed - audio destuttering now happens pre-STT

    # RVC queue + processing
    rvc_queue_enter: Optional[float] = None   # just before rvc_queue.put()
    rvc_queue_exit: Optional[float] = None   # just after  rvc_queue.get()
    rvc_start: Optional[float] = None
    rvc_end: Optional[float] = None

    # done
    audio_sent: Optional[float] = None   # just before conn.sendall()

    
    # Derived durations - all return None if either endpoint is missing, so missing/disabled stages are simply excluded from averages.

    def _dur(self, t0: Optional[float], t1: Optional[float]) -> Optional[float]:
        if t0 is None or t1 is None:
            return None
        return t1 - t0

    @property
    def stt_synth_dur(self) -> Optional[float]:
        return self._dur(self.stt_synth_start, self.stt_synth_end)

    @property
    def stt_destut_dur(self) -> Optional[float]:
        return self._dur(self.stt_destut_start, self.stt_destut_end)

    @property
    def aud_destut_dur(self) -> Optional[float]:
        return self._dur(self.aud_destut_start, self.aud_destut_end)

    @property
    def processed_queue_wait_dur(self) -> Optional[float]:
        '''Time the chunk sat in _processed_queue waiting for Whisper to be free.
        This is the main source of latency in the parallel destutter pipeline —
        the destutter thread produces chunks faster than Whisper consumes them,
        so chunks queue up.'''
        return self._dur(self.processed_queue_enter, self.processed_queue_exit)

    @property
    def buffer_and_queue_dur(self) -> Optional[float]:
        '''
        Full wait from when text first entered the grouping buffer until
        the TTS worker dequeued it.  This is what your old
        buffer_and_queue_wait_ls measured.
        '''
        return self._dur(self.buffer_enter, self.tts_queue_exit)

    @property
    def tts_queue_only_dur(self) -> Optional[float]:
        '''Time spent sitting in the tts_queue specifically (not the buffer).'''
        return self._dur(self.tts_queue_enter, self.tts_queue_exit)

    @property
    def resample_dur(self) -> Optional[float]:
        return self._dur(self.resample_start, self.resample_end)

    @property
    def tts_synth_dur(self) -> Optional[float]:
        return self._dur(self.tts_synth_start, self.tts_synth_end)

    @property
    def rvc_queue_wait_dur(self) -> Optional[float]:
        '''
        Time the audio sat idle in rvc_queue waiting for the worker to
        pick it up.  In a parallel pipeline this is the true 'idle' cost -
        it is NOT part of any useful work, so it is subtracted from the
        end-to-end latency when the pipeline is parallel.
        '''
        return self._dur(self.rvc_queue_enter, self.rvc_queue_exit)

    @property
    def rvc_dur(self) -> Optional[float]:
        return self._dur(self.rvc_start, self.rvc_end)

    @property
    def end_to_end(self) -> Optional[float]:
        '''
        Wall-clock time from first audio arrival → audio sent to client.

        For a PARALLEL RVC pipeline the rvc_queue idle wait is subtracted
        because it doesn't represent real processing time - the main thread
        was doing other work during that window.

        For a SERIES pipeline (RVC_FLAG=False or synchronous RVC) the
        rvc_queue_wait_dur is 0 / None so nothing is subtracted.
        '''
        if self.audio_sent is None:
            return None
        total = self.audio_sent - self.group_start_perf
        idle  = self.rvc_queue_wait_dur or 0.0
        return total - idle



# LatencyTracker  - collects records, computes averages, reports

class LatencyTracker:

    def __init__(self):
        self._records: list[LatencyRecord] = []

    def add(self, rec: LatencyRecord) -> None:
        """Call this in finalize_and_send(), after audio_sent is stamped."""
        self._records.append(rec)

  
    # Internal helpers

    @staticmethod
    def _avg(values: list[float]) -> Optional[float]:
        clean = [v for v in values if v is not None]
        return sum(clean) / len(clean) if clean else None

    @staticmethod
    def _fmt(val: Optional[float]) -> str:
        return f"{val:.3f}s" if val is not None else "N/A"

    def _collect(self, attr: str) -> list[Optional[float]]:
        return [getattr(r, attr) for r in self._records]


    # Public API

    def report_and_reset(self, stats_path: Optional[str] = None,
                         skip_first: bool = True) -> None:
        '''
        Print average latencies for all pipeline stages and optionally
        write them to stats_path.  Then clears internal state so the
        tracker is ready for the next session.

        skip_first=True drops record[0] to discard cold-start outliers,
        matching the old `l.pop(0)` behaviour.
        '''
        records = self._records
        if skip_first and len(records) > 1:
            records = records[1:]

        if not records:
            logger.info("No latency records to report.")
            self._records = []
            return

        def avg_of(attr: str) -> Optional[float]:
            vals = [getattr(r, attr) for r in records]
            return self._avg(vals)

        lines = [
            f"Average end-to-end latency:            {self._fmt(avg_of('end_to_end'))}",
            f"Average audio destutter latency:       {self._fmt(avg_of('aud_destut_dur'))}",
            f"Average processed queue wait:          {self._fmt(avg_of('processed_queue_wait_dur'))}",
            f"Average STT synthesis latency:         {self._fmt(avg_of('stt_synth_dur'))}",
            f"Average STT destutter latency:         {self._fmt(avg_of('stt_destut_dur'))}",
            f"Average buffer + TTS queue wait:       {self._fmt(avg_of('buffer_and_queue_dur'))}",
            f"Average TTS queue wait (only):         {self._fmt(avg_of('tts_queue_only_dur'))}",
            f"Average TTS synthesis latency:         {self._fmt(avg_of('tts_synth_dur'))}",
            f"Average resampling latency:            {self._fmt(avg_of('resample_dur'))}",
            f"Average RVC queue idle wait:           {self._fmt(avg_of('rvc_queue_wait_dur'))}",
            f"Average RVC processing latency:        {self._fmt(avg_of('rvc_dur'))}",
            f"(Chunks measured: {len(records)})",
        ]

        for line in lines:
            logger.info(line)

        if stats_path:
            with open(stats_path, 'a') as f:
                f.write("\n".join(lines) + "\n")
            logger.info(f"Latency stats written to {stats_path}")

        # Reset for next session
        self._records = []