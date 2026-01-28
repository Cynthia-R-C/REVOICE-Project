# Taken from RVC gui_v1.py
# Modified code to customize for my pipeline
# Created 1/21/2026 (Cynthia Chen)

# Imports
import time
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import librosa
import shutil
import torchaudio.transforms as tat

# Imports pt 2 (originally from name == __main__ block in gui_v1.py)
import json  # file handling
import multiprocessing
import re  # error tracking
import threading
import time
import traceback  # error tracking
from multiprocessing import Queue, cpu_count
from queue import Empty  # queues

import librosa  # signal processing
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.transforms as tat

# RVC imports
rvc_root = os.path.abspath('C:\\Users\\crc24\\Documents\\VS_Code_Python_Folder\\ScienceFair2025\\Retrieval-based-Voice-Conversion-WebUI')
rvc_lib = os.path.abspath('C:\\Users\\crc24\\Documents\\VS_Code_Python_Folder\\ScienceFair2025\\Retrieval-based-Voice-Conversion-WebUI\\infer\\lib')
rvc_torchgate = os.path.abspath('C:\\Users\\crc24\\Documents\\VS_Code_Python_Folder\\ScienceFair2025\\Retrieval-based-Voice-Conversion-WebUI\\tools\\torchgate')
rvc_config = os.path.abspath('C:\\Users\\crc24\\Documents\\VS_Code_Python_Folder\\ScienceFair2025\\Retrieval-based-Voice-Conversion-WebUI\\configs')
rvc_i18n = os.path.abspath('C:\\Users\\crc24\\Documents\\VS_Code_Python_Folder\\ScienceFair2025\\Retrieval-based-Voice-Conversion-WebUI\\i18n')

# add rvc folder to paths to search
sys.path.append(rvc_root)  
sys.path.append(rvc_lib)
sys.path.append(rvc_torchgate)
sys.path.append(rvc_config)
sys.path.append(rvc_i18n)

from rtrvc import rvc_for_realtime # core voice conversion logic
from torchgate import TorchGate # noise reduction
from rtrvc import phase_vocoder
from config import Config  # settings
from i18n import I18nAuto  # translation

i18n = I18nAuto()

# Constants
SAMPLERATE = 16000
CHANNELS = 1

# Custom print function
def printt(strr, *args):
    if len(args) == 0:
        print(strr)
    else:
        print(strr % args)



# RVC wrapper class
class RVC:

    def __init__(self):

        # device = rvc_for_realtime.config.device
        # device = torch.device(
        #     "cuda"
        #     if torch.cuda.is_available()
        #     else ("mps" if torch.backends.mps.is_available() else "cpu")
        # )
        current_dir = os.getcwd()

        # Create input and output queue to share data between processes
        self.inp_q = Queue()
        self.opt_q = Queue()

        # Limit n_cpu to the smaller of system's cpu count or 8 to avoid overwhelming resources
        n_cpu = min(cpu_count(), 8)

        self.gui = GUI()  # GUI class instance

        # Launches that many Harvest processes as daemons (background workers that exit with the main program)
        if self.gui.gui_config.f0method == "harvest" and self.gui.gui_config.n_cpu > 1:
            for _ in range(n_cpu):
                p = Harvest(self.inp_q, self.opt_q)
                p.daemon = True
                p.start()

        self.fifo = np.zeros((0,CHANNELS), dtype=np.float32)  # FIFO buffer for audio chunks
        self.gui.setup_vc(self.inp_q, self.opt_q)  # setup vc

    def vc(self, audio_data: np.ndarray) -> np.ndarray:
        '''Runs RVC on given audio_data numpy array and returns converted audio as numpy array
        audio_data: 2D numpy array (mono) float32; shape = (block_frame, 1)'''

        # Check for empty array
        if audio_data is None:
            return np.zeros((0,CHANNELS), dtype=np.float32)

        if self.gui.val_config():  # validate config before starting vc

            printt("cuda_is_available: %s", torch.cuda.is_available())
            
            aud = np.asarray(audio_data, dtype=np.float32)  # ensure float32 numpy array

            # Ensure has right shape before concatenating with buffer, i.e. 2D array
            if len(aud.shape) == 1:
                aud = np.expand_dims(aud, axis=1)  # make into 2D array with shape (n,1)
            elif len(aud.shape) == 2 and aud.shape[1] != CHANNELS:
                raise ValueError(f'Input audio_data has invalid number of channels; audio_data has shape {audio_data.shape}, expected {CHANNELS} channels.')

            # FIFO buffer
            self.fifo = np.concatenate([self.fifo, aud], axis=0)  # add audio to FIFO buffer

            out_chunks = []  # empty array
            bf = self.gui.block_frame

            # Split into RVC-processable chunks
            while self.fifo.shape[0] >= bf:
                chunk = self.fifo[:bf]  # add to chunk
                self.fifo = self.fifo[bf:]  # remove from buffer
                out_chunks.append(self.gui.process_chunk(chunk)) # append processed chunk

            if not out_chunks:  # if ends up empty
                return np.zeros((0,CHANNELS), dtype=np.float32)

            # Return full audio array - combine all chunks into one array of audio
            return np.concatenate(out_chunks, axis=0)
        
        else:
            raise ValueError('RVC processing failed due to invalid configs.')


# Class for Harvest processing algorithm
class Harvest(multiprocessing.Process):

    # Initializes with input queue inp_q and output queue opt_q
    def __init__(self, inp_q, opt_q):
        multiprocessing.Process.__init__(self)
        self.inp_q = inp_q
        self.opt_q = opt_q

    # Loops to get data from inp_q and uses pyworld.harvest to extract pitch f0 and times t from x at 16000 Hz sampling rate, puts ts into opt_q if res_f0 has at least n_cpu entries
    def run(self):
        import numpy as np
        import pyworld

        while 1:

            # Get data from inp_q
            idx, x, res_f0, n_cpu, ts = self.inp_q.get()
            # ts = timestamp; e.g. float or int, that represents when a specific audio chunk was received or processed
            # Needed because different chunks of the audio are processed in parallel

            # Extract pitch f- and times t from x at 16000 Hz sr = f0_ceil 1100, f0_floor=50, frame_period=10ms
            f0, t = pyworld.harvest(
                x.astype(np.double),
                fs=16000,
                f0_ceil=1100,
                f0_floor=50,
                frame_period=10,
            )

            # Stores f0 in res_f0 at idx
            res_f0[idx] = f0
            if len(res_f0.keys()) >= n_cpu:  # if all f0s from all cpus are ready
                self.opt_q.put(ts)


# Defines class to hold all user-adjustiable settings as a single object
# Purpose: centralize config for easy access and updates in the GUI, allowing users to tweak real-time conversion parameters with Harvest or noise handling without hardcoding values 
class GUIConfig:
    def __init__(self) -> None:
        self.pth_path: str = "C:\\Users\\crc24\\Documents\\VS_Code_Python_Folder\\ScienceFair2025\\src\\whisper_streaming\\rvc_voice_models\\xiao-jp.pth"   # hardcoded for now
        self.index_path: str = "C:\\Users\\crc24\\Documents\\VS_Code_Python_Folder\\ScienceFair2025\\src\\whisper_streaming\\rvc_voice_models\\added_IVF3205_Flat_nprobe_1_xiao-jp_v2.index"  # hardcoded for now
        self.pitch: int = 0
        self.formant=0.0
        self.sr_type: str = "sr_model"
        self.block_time: float = 0.25  # s
        self.threhold: int = -60
        self.crossfade_time: float = 0.05
        self.extra_time: float = 2.5
        self.I_noise_reduce: bool = False
        self.O_noise_reduce: bool = False
        self.use_pv: bool = False
        self.rms_mix_rate: float = 0.0
        self.index_rate: float = 0.0
        self.n_cpu: int = 4
        self.f0method: str = "rmvpe"
        # removed hostapi and device stuff

# Sets up the GUI class as the core controller for the interface
class GUI:
    def __init__(self) -> None:
        self.gui_config = GUIConfig()  # make GUIConfig object for settings
        self.config = Config()  # global app config
        self.function = "vc"  # pretrained model loader
        self.delay_time = 0
        # removed audio device and hostapi stuff


    # Validates if certain files are OK before starting vc
    # modified from set_values
    def val_config(self) -> bool:

        # if missing .pth file
        if len(self.gui_config.pth_path.strip()) == 0:  
            print(i18n("请选择pth文件"))
            return False
        
        # if missing index file
        if len(self.gui_config.index_path.strip()) == 0:
            print(i18n("请选择index文件"))
            return False
        
        # if the pth file path contains non-ASCII values like Chinese
        pattern = re.compile("[^\x00-\x7F]+")
        if pattern.findall(self.gui_config.pth_path):
            print(i18n("pth文件路径不可包含中文"))  # oh this is actually funny lol because since this was made in China probably a lot of people would make this error
            return False
        
        # if index file path contains Chinese
        if pattern.findall(self.gui_config.index_path):
            print(i18n("index文件路径不可包含中文"))
            return False
        
        # deleted all the setting gui config vals based on GUI code

        return True
        

    # Method for setting up voice conversion (taken from start_vc in gui_v1.py)
    def setup_vc(self, inp_q, opt_q) -> None:

        # Initialze voice conversion engine
        torch.cuda.empty_cache()  # free unused GPU memory, preventing crashes
        self.rvc = rvc_for_realtime.RVC(  # imported from rtrvc.py
            self.gui_config.pitch,
            self.gui_config.formant,
            self.gui_config.pth_path,
            self.gui_config.index_path,
            self.gui_config.index_rate,
            self.gui_config.n_cpu,
            inp_q,
            opt_q,
            self.config,
            self.rvc if hasattr(self, "rvc") else None,
        )

        # Samplerate calculations - matches the sampling rate ot the model or device to ensure audio compatibility
        # Harvest requires 16000 Hz input, so resampling will be done later
        self.gui_config.samplerate = SAMPLERATE  # modified for pipeline

        # Channel count (e.g. mono/stereo) aligns with hardware
        self.gui_config.channels = CHANNELS  # modified for pipeline
        self.zc = self.gui_config.samplerate // 100
        self.block_frame = (
            int(
                np.round(
                    self.gui_config.block_time
                    * self.gui_config.samplerate
                    / self.zc
                )
            )
            * self.zc
        )

        # Frame calculations
        # Calculates zc (cycle size) as samplerate // 100 for alignment
        # These frame sizes define how audio is chunked and overlapped, balancing latency (smaller block_time) and stability
        self.block_frame_16k = 160 * self.block_frame // self.zc  # audio chunk size in samples scaled for 16000 Hz
        self.crossfade_frame = (  # for SOLA blending
            int(
                np.round(
                    self.gui_config.crossfade_time
                    * self.gui_config.samplerate
                    / self.zc
                )
            )
            * self.zc
        )
        self.sola_buffer_frame = min(self.crossfade_frame, 4 * self.zc)
        self.sola_search_frame = self.zc
        self.extra_frame = (
            int(
                np.round(
                    self.gui_config.extra_time
                    * self.gui_config.samplerate
                    / self.zc
                )
            )
            * self.zc
        )

        # Buffer tensors setup
        # Sets up memory buffers for audio processing; purpose is to pre-allocate tensors and arrays for holding and manipulating audio data, enabling smooth real-time streaming with Harvest and SOLA blending
        self.input_wav: torch.Tensor = torch.zeros(  # tensor for input audio
            self.extra_frame
            + self.crossfade_frame
            + self.sola_search_frame
            + self.block_frame,
            device=self.config.device,
            dtype=torch.float32,
        )

        self.input_wav_denoise: torch.Tensor = self.input_wav.clone()  # cloned for noise reduction

        # for resampled 16000 Hz audio
        self.input_wav_res: torch.Tensor = torch.zeros(
            160 * self.input_wav.shape[0] // self.zc,
            device=self.config.device,
            dtype=torch.float32,
        )

        # For volume calculations
        self.rms_buffer: np.ndarray = np.zeros(4 * self.zc, dtype="float32")

        # For SOLA overlap
        self.sola_buffer: torch.Tensor = torch.zeros(
            self.sola_buffer_frame, device=self.config.device, dtype=torch.float32
        )

        # SOLA cloned for noise reduction
        # This is just math at this point
        self.nr_buffer: torch.Tensor = self.sola_buffer.clone()
        self.output_buffer: torch.Tensor = self.input_wav.clone()
        self.skip_head = self.extra_frame // self.zc
        self.return_length = (
            self.block_frame + self.sola_buffer_frame + self.sola_search_frame
        ) // self.zc
        self.fade_in_window: torch.Tensor = (
            torch.sin(
                0.5
                * np.pi
                * torch.linspace(
                    0.0,
                    1.0,
                    steps=self.sola_buffer_frame,
                    device=self.config.device,
                    dtype=torch.float32,
                )
            )
            ** 2
        )
        self.fade_out_window: torch.Tensor = 1 - self.fade_in_window

        # Resamplers and Torchgate
        # Prepares tools of raudio transofrmation; purpose = set up resampling and noise gating to match Harvest's requirements and clean audio, then start the stream

        # Create resampler to convert to 16000 Hz
        self.resampler = tat.Resample(
            orig_freq=self.gui_config.samplerate,
            new_freq=16000,
            dtype=torch.float32,
        ).to(self.config.device)

        # If the model's target rate differs from samplerate, create resampler2 to sample back
        if self.rvc.tgt_sr != self.gui_config.samplerate:
            self.resampler2 = tat.Resample(
                orig_freq=self.rvc.tgt_sr,
                new_freq=self.gui_config.samplerate,
                dtype=torch.float32,
            ).to(self.config.device)

        # otherwise no resample2
        else:
            self.resampler2 = None

        # Noise gating to improve input quality
        self.tg = TorchGate(
            sr=self.gui_config.samplerate, n_fft=4 * self.zc, prop_decrease=0.9
        ).to(self.config.device)  # initialize noise gate with set parameters e.g. FFT


    # Begins processing each audio chunk in real time; purpose is to start audio callback, track processing time, and convert input to mono fo rconsistent handling in Harvest pitch extraction
    def process_chunk(self, indata: np.ndarray) -> np.ndarray:
        '''New version of audio_callback() from gui_v1.py.
        indata = mono float32 np audio array with chunk size self.block_frame samples; shape with dim 2 = (block_frame, 1)
        outdata = same datatype and shape as indata'''

        # record start time
        start_time = time.perf_counter()

        # convert input audio from stereo to mono
        # How shape goes in this method: downmix to mono for processing, then reexpand to 2D array for output (still only mono audio though)
        if indata.ndim == 2:
            indata = librosa.to_mono(indata.T)  # convert to mono
        # indata = mic input chunk

        # Silences quiet audio segments to reduce noise; purpose is to gate low-volume input based on the user's threshold, improving clarity before Harvest pitch processing
        # Why is this guy's code so horrible to read why can't they use proper top-down and subroutined programming so split everything into smaller chunks and no spacing at all oh my god
        # Oh wait whoever reads this you won't get what I'm saying because I fixed it all with my comments, you're welcome T-T
        # I've come again three months later, thank you past me for your comments and reorganization ;v;
        if self.gui_config.threhold > -60:  # if silence threshold is above -60 dB

            # append rms buffer to indata and calculate the volume (RMS) with librosa, slice, convert RMS to dB, and zero segments below threshold
            indata = np.append(self.rms_buffer, indata)

            # Calculate volume
            rms = librosa.feature.rms(
                y=indata, frame_length=4 * self.zc, hop_length=self.zc
            )[:, 2:]

            # update rms buffer
            self.rms_buffer[:] = indata[-4 * self.zc :]

            # Slice indata
            indata = indata[2 * self.zc - self.zc // 2 :]

            # Convert RMS to dB
            db_threhold = (
                librosa.amplitude_to_db(rms, ref=1.0)[0] < self.gui_config.threhold
            )

            # Zero segements below threshold
            for i in range(db_threhold.shape[0]):
                if db_threhold[i]:
                    indata[i * self.zc : (i + 1) * self.zc] = 0
            indata = indata[self.zc // 2 :]

        # Input buffer shift
        # Move older audio in input_wav forward, add new indata as torch tensor on the device, shifts input_wav_res similarly for resampled audio
        # Ensures continuous audio flow, providing context for Harvest's pitch extraction and SOLA blending
        self.input_wav[: -self.block_frame] = self.input_wav[
            self.block_frame :
        ].clone()
        self.input_wav[-indata.shape[0] :] = torch.from_numpy(indata).to(
            self.config.device
        )
        self.input_wav_res[: -self.block_frame_16k] = self.input_wav_res[
            self.block_frame_16k :
        ].clone()

        # input noise reduction and resampling
        # Purpose: clean and convert audio to 16000 Hz for Harvest, ensuring high-quality input

        # if I_noise_reduce is enabled, apply TorchGate, fade with windows, update with buffers etc.
        if self.gui_config.I_noise_reduce:

            self.input_wav_denoise[: -self.block_frame] = self.input_wav_denoise[
                self.block_frame :
            ].clone()

            input_wav = self.input_wav[-self.sola_buffer_frame - self.block_frame :]

            input_wav = self.tg(
                input_wav.unsqueeze(0), self.input_wav.unsqueeze(0)
            ).squeeze(0)

            input_wav[: self.sola_buffer_frame] *= self.fade_in_window

            input_wav[: self.sola_buffer_frame] += (
                self.nr_buffer * self.fade_out_window
            )

            self.input_wav_denoise[-self.block_frame :] = input_wav[
                : self.block_frame
            ]

            self.nr_buffer[:] = input_wav[self.block_frame :]

            self.input_wav_res[-self.block_frame_16k - 160 :] = self.resampler(
                self.input_wav_denoise[-self.block_frame - 2 * self.zc :]
            )[160:]

        # Otherwise directly resample to 16000
        else:
            self.input_wav_res[-160 * (indata.shape[0] // self.zc + 1) :] = (
                self.resampler(self.input_wav[-indata.shape[0] - 2 * self.zc :])[
                    160:
                ]
            )

        # inference
        # Performs voice conversion on the input audio
        # purpose is to generate converted audio using the RVC model with Harvest pitch if selected, or output raw audio in monitoring mode

        # if in voice conversion mode
        if self.function == "vc":
            # call rvc.infer to generate the right audio
            infer_wav = self.rvc.infer(
                self.input_wav_res,
                self.block_frame_16k,
                self.skip_head,
                self.return_length,
                self.gui_config.f0method,
            )

            # If needed upsample with resampler2
            if self.resampler2 is not None:
                infer_wav = self.resampler2(infer_wav)
        
        # extra input noise reduction
        elif self.gui_config.I_noise_reduce:
            infer_wav = self.input_wav_denoise[self.extra_frame :].clone()

        # don't go through the second resampling
        else:
            infer_wav = self.input_wav[self.extra_frame :].clone()
        
        # output noise reduction
        if self.gui_config.O_noise_reduce and self.function == "vc":
            self.output_buffer[: -self.block_frame] = self.output_buffer[
                self.block_frame :
            ].clone()
            self.output_buffer[-self.block_frame :] = infer_wav[-self.block_frame :]
            infer_wav = self.tg(
                infer_wav.unsqueeze(0), self.output_buffer.unsqueeze(0)
            ).squeeze(0)
        
        # volume envelop mixing
        # Adjusts output volume to match input; purpose is to balance loudness for natural-sounding converted audio
        if self.gui_config.rms_mix_rate < 1 and self.function == "vc":
            if self.gui_config.I_noise_reduce:
                input_wav = self.input_wav_denoise[self.extra_frame :]
            else:
                input_wav = self.input_wav[self.extra_frame :]
            rms1 = librosa.feature.rms(
                y=input_wav[: infer_wav.shape[0]].cpu().numpy(),
                frame_length=4 * self.zc,
                hop_length=self.zc,
            )
            rms1 = torch.from_numpy(rms1).to(self.config.device)
            rms1 = F.interpolate(
                rms1.unsqueeze(0),
                size=infer_wav.shape[0] + 1,
                mode="linear",
                align_corners=True,
            )[0, 0, :-1]
            rms2 = librosa.feature.rms(
                y=infer_wav[:].cpu().numpy(),
                frame_length=4 * self.zc,
                hop_length=self.zc,
            )
            rms2 = torch.from_numpy(rms2).to(self.config.device)
            rms2 = F.interpolate(
                rms2.unsqueeze(0),
                size=infer_wav.shape[0] + 1,
                mode="linear",
                align_corners=True,
            )[0, 0, :-1]
            rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-3)
            infer_wav *= torch.pow(
                rms1 / rms2, torch.tensor(1 - self.gui_config.rms_mix_rate)
            )

        # SOLA algorithm from https://github.com/yxlllc/DDSP-SVC
        # Aligns audio chunks using SOLA
        # Purpose: stitch converted audio seamlessly, avoiding clicks in real-time output
        conv_input = infer_wav[
            None, None, : self.sola_buffer_frame + self.sola_search_frame
        ]
        cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
        cor_den = torch.sqrt(
            F.conv1d(
                conv_input**2,
                torch.ones(1, 1, self.sola_buffer_frame, device=self.config.device),
            )
            + 1e-8
        )
        if sys.platform == "darwin":
            _, sola_offset = torch.max(cor_nom[0, 0] / cor_den[0, 0])
            sola_offset = sola_offset.item()
        else:
            sola_offset = torch.argmax(cor_nom[0, 0] / cor_den[0, 0])
        printt("sola_offset = %d", int(sola_offset))
        infer_wav = infer_wav[sola_offset:]
        if "privateuseone" in str(self.config.device) or not self.gui_config.use_pv:
            infer_wav[: self.sola_buffer_frame] *= self.fade_in_window
            infer_wav[: self.sola_buffer_frame] += (
                self.sola_buffer * self.fade_out_window
            )
        else:
            infer_wav[: self.sola_buffer_frame] = phase_vocoder(
                self.sola_buffer,
                infer_wav[: self.sola_buffer_frame],
                self.fade_out_window,
                self.fade_in_window,
            )
        self.sola_buffer[:] = infer_wav[
            self.block_frame : self.block_frame + self.sola_buffer_frame
        ]

        # Output and time calculation
        # Converts infer_wav to NumPy, repeats for channels, and sets outdata; calculates and displays inference time if voice channel flag is on
        # Delivers final audio and displays latency, helping users optimize Harvest settings
        outdata = (
            infer_wav[: self.block_frame]
            .repeat(self.gui_config.channels, 1)
            .t()
            .cpu()
            .numpy()
        )

        total_time = time.perf_counter() - start_time
        printt("Infer time: %.2f", total_time)

        return outdata