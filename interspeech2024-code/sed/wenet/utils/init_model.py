# Copyright (c) 2022 Binbin Zhang (binbzha@qq.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

# --- New safe import patch for init_model.py (Windows + Pylance friendly) ---
from typing import Any, cast

print("[Info] Applying safe import patch for init_model.py")

# Pre-declare all optional symbols so Pylance knows they exist
TransducerJoint: Any = None
ConvPredictor: Any = None
EmbeddingPredictor: Any = None
RNNPredictor: Any = None
Transducer: Any = None
K2Model: Any = None

ASRModel: Any = None
SEDModel: Any = None
GlobalCMVN: Any = None
CTC: Any = None
BiTransformerDecoder: Any = None
TransformerDecoder: Any = None
ConformerEncoder: Any = None
TransformerEncoder: Any = None
S3prlFrontend: Any = None

BranchformerEncoder: Any = None
EBranchformerEncoder: Any = None
SqueezeformerEncoder: Any = None
EfficientConformerEncoder: Any = None

Paraformer: Any = None
Predictor: Any = None

load_cmvn: Any = None  # from wenet.utils.cmvn
ConvLSTM: Any = None   # from wenet.stutter.convlstm
StutterNet: Any = None # from wenet.stutter.stutternet

def try_import(module_path: str, names: list[str]) -> None:
    """Try importing selected symbols from a module, else keep them as None."""
    try:
        module = __import__(module_path, fromlist=names)
        for name in names:
            globals()[name] = getattr(module, name)
    except Exception as e:
        print(f"[Warning] Skipped optional module {module_path} ({e})")

# Optional ASR/transducer
try_import("wenet.transducer.joint", ["TransducerJoint"])
try_import("wenet.transducer.predictor", ["ConvPredictor", "EmbeddingPredictor", "RNNPredictor"])
try_import("wenet.transducer.transducer", ["Transducer"])
try_import("wenet.k2.model", ["K2Model"])

# Transformer / SED / CMVN / CTC
try_import("wenet.transformer.asr_model", ["ASRModel"])
try_import("wenet.transformer.sed_model", ["SEDModel"])
try_import("wenet.transformer.cmn", ["GlobalCMVN"])
try_import("wenet.transformer.ctc", ["CTC"])
try_import("wenet.transformer.decoder", ["BiTransformerDecoder", "TransformerDecoder"])
try_import("wenet.transformer.encoder", ["ConformerEncoder", "TransformerEncoder"])
try_import("wenet.transformer.wav2vec2_encoder", ["S3prlFrontend"])

# Encoders (optional)
try_import("wenet.branchformer.encoder", ["BranchformerEncoder"])
try_import("wenet.e_branchformer.encoder", ["EBranchformerEncoder"])
try_import("wenet.squeezeformer.encoder", ["SqueezeformerEncoder"])
try_import("wenet.efficient_conformer.encoder", ["EfficientConformerEncoder"])

# Other optional architectures
try_import("wenet.paraformer.paraformer", ["Paraformer"])
try_import("wenet.cif.predictor", ["Predictor"])

# Utilities + StutterNet (these are the ones you actually use)
try_import("wenet.utils.cmvn", ["load_cmvn"])
try_import("wenet.stutter.convlstm", ["ConvLSTM"])
try_import("wenet.stutter.stutternet", ["StutterNet"])

print("[Info] init_model.py import patch completed.")




def init_model(configs):
    # if configs['cmvn_file'] is not None:
    #     mean, istd = load_cmvn(configs['cmvn_file'], configs['is_json_cmvn'])
    #     global_cmvn = GlobalCMVN(
    #         torch.from_numpy(mean).float(),
    #         torch.from_numpy(istd).float())
    # else:
    #     global_cmvn = None

    cmvn_file = configs.get('cmvn_file', None)
    if cmvn_file is None and 'cmvn' in configs:
        # Fallback to explicit CLI argument
        cmvn_file = configs['cmvn']

        if cmvn_file is not None:
            print(f"[Info] Using CMVN file: {cmvn_file}")
            cmvn = load_cmvn(cmvn_file)
        else:
            print("[Warning] No CMVN file found in configs; proceeding without CMVN.")
            cmvn = None


    input_dim = configs['input_dim']
    vocab_size = configs['output_dim']

    if configs.get('convlstm', False):
        model = ConvLSTM(vocab_size=vocab_size,
                            global_cmvn=global_cmvn,
                            **configs['convlstm_conf'])
        return model
    if configs.get('stutternet', False):
        model = StutterNet(vocab_size=vocab_size,)
        return model

    encoder_type = configs.get('encoder', 'conformer')
    decoder_type = configs.get('decoder', 'bitransformer')

    if encoder_type == 'conformer':
        encoder = ConformerEncoder(input_dim,
                                   global_cmvn=global_cmvn,
                                   **configs['encoder_conf'])
    elif encoder_type == 'squeezeformer':
        encoder = SqueezeformerEncoder(input_dim,
                                       global_cmvn=global_cmvn,
                                       **configs['encoder_conf'])
    elif encoder_type == 'efficientConformer':
        encoder = EfficientConformerEncoder(
            input_dim,
            global_cmvn=global_cmvn,
            **configs['encoder_conf'],
            **configs['encoder_conf']['efficient_conf']
            if 'efficient_conf' in configs['encoder_conf'] else {})
    elif encoder_type == 'branchformer':
        encoder = BranchformerEncoder(input_dim,
                                      global_cmvn=global_cmvn,
                                      **configs['encoder_conf'])
    elif encoder_type == 'e_branchformer':
        encoder = EBranchformerEncoder(input_dim,
                                       global_cmvn=global_cmvn,
                                       **configs['encoder_conf'])
    elif encoder_type == "wav2vec2":
        encoder = S3prlFrontend(**configs['frontend_conf'])
    else:
        encoder = TransformerEncoder(input_dim,
                                     global_cmvn=global_cmvn,
                                     **configs['encoder_conf'])
    if decoder_type == 'transformer':
        decoder = TransformerDecoder(vocab_size, encoder.output_size(),
                                     **configs['decoder_conf'])
    elif decoder_type is None:
        decoder = None
    else:
        assert 0.0 < configs['model_conf']['reverse_weight'] < 1.0
        assert configs['decoder_conf']['r_num_blocks'] > 0
        decoder = BiTransformerDecoder(vocab_size, encoder.output_size(),
                                       **configs['decoder_conf'])

    if decoder is not None:
        ctc = CTC(vocab_size, encoder.output_size())

    # Init joint CTC/Attention or Transducer model
    if 'predictor' in configs:
        predictor_type = configs.get('predictor', 'rnn')
        if predictor_type == 'rnn':
            predictor = RNNPredictor(vocab_size, **configs['predictor_conf'])
        elif predictor_type == 'embedding':
            predictor = EmbeddingPredictor(vocab_size,
                                           **configs['predictor_conf'])
            configs['predictor_conf']['output_size'] = configs[
                'predictor_conf']['embed_size']
        elif predictor_type == 'conv':
            predictor = ConvPredictor(vocab_size, **configs['predictor_conf'])
            configs['predictor_conf']['output_size'] = configs[
                'predictor_conf']['embed_size']
        else:
            raise NotImplementedError(
                "only rnn, embedding and conv type support now")
        configs['joint_conf']['enc_output_size'] = configs['encoder_conf'][
            'output_size']
        configs['joint_conf']['pred_output_size'] = configs['predictor_conf'][
            'output_size']
        joint = TransducerJoint(vocab_size, **configs['joint_conf'])
        model = Transducer(vocab_size=vocab_size,
                           blank=0,
                           predictor=predictor,
                           encoder=encoder,
                           attention_decoder=decoder,
                           joint=joint,
                           ctc=ctc,
                           **configs['model_conf'])
    elif 'paraformer' in configs:
        predictor = Predictor(**configs['cif_predictor_conf'])
        model = Paraformer(vocab_size=vocab_size,
                           encoder=encoder,
                           decoder=decoder,
                           ctc=ctc,
                           predictor=predictor,
                           **configs['model_conf'])
    else:
        print(configs)
        if configs.get('sed', False):
            model = SEDModel(vocab_size=vocab_size,
                             encoder=encoder)
        else:
            model = ASRModel(vocab_size=vocab_size,
                             encoder=encoder,
                             decoder=decoder,
                             ctc=ctc,
                             **configs['model_conf'])
    return model