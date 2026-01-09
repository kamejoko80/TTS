import argparse
from rknn.api import RKNN

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--platform", default="rk3588")
    ap.add_argument("--n_mels", type=int, default=80)
    ap.add_argument("--mel_frames", type=int, default=200)
    ap.add_argument("--opt_level", type=int, default=3)
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()

    rknn = RKNN(verbose=True)
    cfg = {"target_platform": args.platform, "optimization_level": args.opt_level}
    if args.fp16:
        cfg["float_dtype"] = "float16"
    rknn.config(**cfg)

    ret = rknn.load_onnx(
        model=args.onnx,
        input_size_list=[
            [1, args.n_mels, args.mel_frames],
        ],
    )
    if ret != 0:
        raise SystemExit("load_onnx failed")

    ret = rknn.build(do_quantization=False)
    if ret != 0:
        raise SystemExit("build failed")

    ret = rknn.export_rknn(args.out)
    if ret != 0:
        raise SystemExit("export_rknn failed")

    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()
