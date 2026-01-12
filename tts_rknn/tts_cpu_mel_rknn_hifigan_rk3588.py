import argparse
import time
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

# --- CRITICAL FIX FOR PYTORCH 2.6+ ---
import torch.serialization
_orig_load = torch.load
torch.load = lambda *args, **kwargs: _orig_load(*args, **{**kwargs, "weights_only": False})
# -------------------------------------


def time_scale_mel_tc(mel_tc: np.ndarray, speed: float) -> np.ndarray:
    if abs(speed - 1.0) < 1e-6:
        return mel_tc
    t, c = mel_tc.shape
    new_t = max(1, int(round(t / speed)))
    x = torch.from_numpy(mel_tc.T).unsqueeze(0)
    y = F.interpolate(x, size=new_t, mode="linear", align_corners=False)
    return y.squeeze(0).transpose(0, 1).cpu().numpy()


def pad_or_trim_mel_tc(mel_tc: np.ndarray, target_T: int) -> np.ndarray:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocoder_rknn", required=True)
    ap.add_argument("--tts_model", default="tts_models/en/ljspeech/tacotron2-DDC")
    ap.add_argument("--text", required=True)
    ap.add_argument("--out_wav", default="out.wav")
    ap.add_argument("--mel_frames", type=int, default=1200)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--tail_ms", type=float, default=0.0)
    ap.add_argument("--npu_core", type=int, default=0)
    args = ap.parse_args()

    # Load vocoder config
    mm = ModelManager()
    _, voc_cfg_path, _ = mm.download_model("vocoder_models/en/ljspeech/hifigan_v2")
    voc_cfg = load_config(voc_cfg_path)
    voc_ap = AudioProcessor(verbose=False, **voc_cfg.audio)

    voc_sr = int(voc_cfg.audio.sample_rate)
    hop_length = int(voc_cfg.audio.hop_length)
    n_mels = int(voc_cfg.audio.num_mels)

    # Init RKNN
    rknn = RKNNLite()
    rknn.load_rknn(args.vocoder_rknn)
    rknn.init_runtime(core_mask=args.npu_core)

    # Load TTS
    tts = TTS(args.tts_model, gpu=False)
    tts_model = tts.synthesizer.tts_model
    tts_cfg = tts.synthesizer.tts_config

    # -------- Text → Mel (CPU) --------
    t0 = time.perf_counter()
    out = synthesis(
        model=tts_model,
        text=args.text,
        CONFIG=tts_cfg,
        use_cuda=False,
        speaker_id=None,
        style_wav=None,
        use_griffin_lim=True,
        d_vector=None,
        language_id=None,
    )
    t1 = time.perf_counter()

    mel_t = extract_mel_from_synthesis(out)
    mel = mel_t.detach().cpu().float().numpy()

    if mel.ndim == 3:
        mel = mel[0]
    if mel.shape[0] != n_mels:
        mel = mel.T

    mel = tts_model.ap.denormalize(mel)
    mel_tc = voc_ap.normalize(mel.T)

    mel_tc = time_scale_mel_tc(mel_tc, args.speed)
    real_mel_frames = mel_tc.shape[0]
    mel_tc = pad_or_trim_mel_tc(mel_tc, args.mel_frames)

    mel_time = t1 - t0

    # -------- Vocoder (RKNN NPU) --------
    mel_in = mel_tc.T[np.newaxis, :, :].astype(np.float32)

    t2 = time.perf_counter()
    wav = rknn.inference(inputs=[mel_in])[0].squeeze()
    t3 = time.perf_counter()

    vocoder_time = t3 - t2

    # Trim output
    trim_samples = real_mel_frames * hop_length
    if args.tail_ms > 0:
        trim_samples += int(voc_sr * args.tail_ms / 1000)

    wav = wav[:trim_samples]

    # Fix audio volume
    peak = np.max(np.abs(wav))
    if peak > 1e-6:
        wav = wav / peak * 0.95

    wav = np.clip(wav, -1.0, 1.0)

    sf.write(args.out_wav, wav, voc_sr)
    rknn.release()

    audio_duration = len(wav) / voc_sr
    total_time = mel_time + vocoder_time

    print("========== RTF REPORT ==========")
    print(f"Audio duration      : {audio_duration:.3f} s")
    print(f"Mel inference (CPU)  : {mel_time:.3f} s")
    print(f"Vocoder (NPU)        : {vocoder_time:.3f} s")
    print(f"Total processing    : {total_time:.3f} s")
    print(f"RTF (vocoder only)  : {vocoder_time / audio_duration:.3f}")
    print(f"RTF (end-to-end)    : {total_time / audio_duration:.3f}")
    print("================================")


if __name__ == "__main__":
    main()
