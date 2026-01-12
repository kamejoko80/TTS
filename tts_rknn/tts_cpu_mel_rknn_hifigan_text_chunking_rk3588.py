import argparse
import time
import re
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from rknnlite.api import RKNNLite

from TTS.api import TTS
from TTS.utils.manage import ModelManager
from TTS.config import load_config
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.synthesis import synthesis

import torch.serialization
_orig_load = torch.load
torch.load = lambda *args, **kwargs: _orig_load(*args, **{**kwargs, "weights_only": False})

def time_scale_mel_tc(mel_tc, speed):
    if abs(speed - 1.0) < 1e-6:
        return mel_tc
    t, c = mel_tc.shape
    new_t = max(1, int(round(t / speed)))
    x = torch.from_numpy(mel_tc.T).unsqueeze(0)
    y = F.interpolate(x, size=new_t, mode="linear", align_corners=False)
    return y.squeeze(0).transpose(0, 1).cpu().numpy()


def pad_or_trim_mel_tc(mel_tc, target_T):
    T = mel_tc.shape[0]
    if T == target_T:
        return mel_tc
    if T > target_T:
        return mel_tc[:target_T]
    pad = np.repeat(mel_tc[-1:, :], target_T - T, axis=0)
    return np.concatenate([mel_tc, pad], axis=0)


def extract_mel_from_synthesis(out):
    mel = out["outputs"]["model_outputs"]
    if isinstance(mel, (list, tuple)):
        mel = mel[0]
    return mel


def split_text_natural(text, min_chars=6):
    parts = re.split(r'([.,!?;])', text)
    chunks = []
    buf = ""

    for p in parts:
        buf += p
        if p in ".,!?;":
            buf = buf.strip()
            if len(buf) >= min_chars and len(buf.split()) >= 2:
                chunks.append(buf)
                buf = ""
            else:
                buf += " "

    if buf.strip():
        if len(buf.strip()) >= min_chars and len(buf.split()) >= 2:
            chunks.append(buf.strip())

    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocoder_rknn", required=True)
    ap.add_argument("--tts_model", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--out_wav", default="out.wav")
    ap.add_argument("--mel_frames", type=int, default=1200)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--tail_ms", type=float, default=0.0)
    ap.add_argument("--npu_core", type=int, default=0)
    args = ap.parse_args()

    mm = ModelManager()
    _, voc_cfg_path, _ = mm.download_model("vocoder_models/en/ljspeech/hifigan_v2")
    voc_cfg = load_config(voc_cfg_path)
    voc_ap = AudioProcessor(verbose=False, **voc_cfg.audio)

    voc_sr = voc_cfg.audio.sample_rate
    hop_length = voc_cfg.audio.hop_length
    n_mels = voc_cfg.audio.num_mels

    rknn = RKNNLite()
    rknn.load_rknn(args.vocoder_rknn)
    rknn.init_runtime(core_mask=args.npu_core)

    tts = TTS(args.tts_model, gpu=False)
    tts_model = tts.synthesizer.tts_model
    tts_cfg = tts.synthesizer.tts_config

    wav_all = []
    total_mel_time = 0.0
    total_vocoder_time = 0.0

    text_chunks = split_text_natural(args.text)

    for chunk in text_chunks:
        if len(chunk.strip()) < 6 or len(chunk.split()) < 2:
            continue

        t0 = time.perf_counter()
        out = synthesis(
            model=tts_model,
            text=chunk,
            CONFIG=tts_cfg,
            use_cuda=False,
            speaker_id=None,
            style_wav=None,
            use_griffin_lim=True,
            d_vector=None,
            language_id=None,
        )
        t1 = time.perf_counter()

        mel = extract_mel_from_synthesis(out).detach().cpu().float().numpy()
        if mel.ndim == 3:
            mel = mel[0]
        if mel.shape[0] != n_mels:
            mel = mel.T

        mel = tts_model.ap.denormalize(mel)
        mel_tc = voc_ap.normalize(mel.T)
        mel_tc = time_scale_mel_tc(mel_tc, args.speed)

        real_frames = mel_tc.shape[0]
        mel_tc = pad_or_trim_mel_tc(mel_tc, args.mel_frames)

        mel_in = mel_tc.T[np.newaxis].astype(np.float32)

        t2 = time.perf_counter()
        wav = rknn.inference(inputs=[mel_in])[0].squeeze()
        t3 = time.perf_counter()

        trim = real_frames * hop_length
        if args.tail_ms > 0:
            trim += int(voc_sr * args.tail_ms / 1000)

        wav_all.append(wav[:trim])

        total_mel_time += (t1 - t0)
        total_vocoder_time += (t3 - t2)

    wav = np.concatenate(wav_all)

    peak = np.max(np.abs(wav))
    if peak > 1e-6:
        wav = wav / peak * 0.95

    wav = np.clip(wav, -1.0, 1.0)

    sf.write(args.out_wav, wav, voc_sr)
    rknn.release()

    duration = len(wav) / voc_sr
    total_time = total_mel_time + total_vocoder_time

    print("========== RTF REPORT ==========")
    print(f"Audio duration      : {duration:.3f} s")
    print(f"Mel inference (CPU)  : {total_mel_time:.3f} s")
    print(f"Vocoder (NPU)        : {total_vocoder_time:.3f} s")
    print(f"Total processing    : {total_time:.3f} s")
    print(f"RTF (vocoder only)  : {total_vocoder_time / duration:.3f}")
    print(f"RTF (end-to-end)    : {total_time / duration:.3f}")
    print("================================")


if __name__ == "__main__":
    main()
