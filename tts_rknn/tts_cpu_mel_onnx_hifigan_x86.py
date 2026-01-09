import argparse
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import onnxruntime as ort

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
    if speed <= 0:
        raise ValueError("speed must be > 0")
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
    if "outputs" in out and "model_outputs" in out["outputs"]:
        mel = out["outputs"]["model_outputs"]
        if isinstance(mel, (list, tuple)):
            mel = mel[0]
        return mel
    raise RuntimeError(f"Cannot find mel in synthesis output keys: {list(out.keys())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocoder_onnx", required=True)
    ap.add_argument("--tts_model", default="tts_models/en/ljspeech/tacotron2-DDC")
    ap.add_argument("--text", required=True)
    ap.add_argument("--out_wav", default="test_T1200.wav")
    ap.add_argument("--mel_frames", type=int, default=1200)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--tail_ms", type=float, default=0.0,
                    help="Optional tail in milliseconds (e.g. 50 for natural decay)")
    args = ap.parse_args()

    # Load vocoder config (HiFiGAN v2)
    mm = ModelManager()
    _, voc_cfg_path, _ = mm.download_model("vocoder_models/en/ljspeech/hifigan_v2")
    voc_cfg = load_config(voc_cfg_path)
    voc_ap = AudioProcessor(verbose=False, **voc_cfg.audio)

    voc_sr = int(voc_cfg.audio.sample_rate)
    hop_length = int(voc_cfg.audio.hop_length)
    n_mels = int(voc_cfg.audio.num_mels)

    print(f"> Using vocoder ONNX: {args.vocoder_onnx}")
    print(f"> Expecting mel shape: [1, {n_mels}, {args.mel_frames}]")

    # Load TTS
    tts = TTS(args.tts_model, gpu=False)
    tts_model = tts.synthesizer.tts_model
    tts_cfg = tts.synthesizer.tts_config

    # Text → mel
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

    mel_t = extract_mel_from_synthesis(out)
    mel = mel_t.detach().cpu().float().numpy()

    if mel.ndim == 3:
        mel = mel[0]
    if mel.shape[0] != n_mels:
        mel = mel.T

    # Bridge normalization (CRITICAL)
    mel = tts_model.ap.denormalize(mel)      # [80, T_real]
    mel_tc = voc_ap.normalize(mel.T)         # [T_real, 80]

    # Speed control
    mel_tc = time_scale_mel_tc(mel_tc, args.speed)

    # --- SAVE REAL MEL LENGTH (IMPORTANT) ---
    real_mel_frames = mel_tc.shape[0]

    # Pad / trim to fixed ONNX length
    mel_tc = pad_or_trim_mel_tc(mel_tc, args.mel_frames)

    # Run HiFiGAN ONNX
    sess = ort.InferenceSession(args.vocoder_onnx, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    mel_in = mel_tc.T[np.newaxis, :, :].astype(np.float32)  # [1, 80, T]
    wav = sess.run(None, {input_name: mel_in})[0].squeeze()

    # --- EXACT TRIMMING ---
    trim_samples = real_mel_frames * hop_length
    if args.tail_ms > 0:
        trim_samples += int(voc_sr * args.tail_ms / 1000.0)

    wav = wav[:trim_samples]
    wav = np.clip(wav, -1.0, 1.0)

    sf.write(args.out_wav, wav, voc_sr)

    print(f"> Wrote: {args.out_wav}")
    print(f"> Duration: {len(wav) / voc_sr:.2f}s (trimmed)")


if __name__ == "__main__":
    main()
