import argparse
import torch

from TTS.utils.manage import ModelManager
from TTS.config import load_config
from TTS.vocoder.models.gan import GAN

class VocoderWrapper(torch.nn.Module):
    def __init__(self, gan):
        super().__init__()
        self.gan = gan

    def forward(self, mel):
        return self.gan.inference(mel)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="hifigan_v2_T200.onnx")
    ap.add_argument("--mel_frames", type=int, default=200)
    ap.add_argument("--opset", type=int, default=15)
    args = ap.parse_args()

    mm = ModelManager()
    ckpt, cfg_path, _ = mm.download_model("vocoder_models/en/ljspeech/hifigan_v2")

    cfg = load_config(cfg_path)
    model = GAN.init_from_config(cfg, verbose=False)
    model.load_checkpoint(cfg, ckpt, eval=True)
    model.eval()

    n_mels = cfg.audio.num_mels
    dummy_mel = torch.randn(1, n_mels, args.mel_frames, dtype=torch.float32)

    wrapper = VocoderWrapper(model).eval()

    torch.onnx.export(
        wrapper,
        (dummy_mel,),
        args.out,
        opset_version=args.opset,
        input_names=["mel"],
        output_names=["wav"],
        dynamic_axes=None,
        do_constant_folding=True,
    )

    print(f"Exported: {args.out}")
    print(f"Input shape: [1, {n_mels}, {args.mel_frames}] -> wav")

if __name__ == "__main__":
    main()
