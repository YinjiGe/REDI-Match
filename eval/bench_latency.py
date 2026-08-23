"""Inference latency test (plain + rot)."""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys

_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _repo)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for k in list(sys.modules.keys()):
    if "redimatch" in k.lower():
        del sys.modules[k]

import torch

from _common import (
    _build_eval_model,
    install_gp_intrinsic_rotation_router,
)

WEIGHTS = os.path.join(_repo, "models", "outdoor.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _apply_fh_hybrid_decoder(model):
    dec = model.decoder
    dec.c4_hybrid = True
    dec.c4_hybrid_alpha = 0.35
    dec.c4_hybrid_temp = 0.05
    dec.c4_hybrid_margin_keep = 0.03
    dec.c4_hybrid_disagree_margin = 0.03
    dec.c4_hybrid_ot_pool_size = 12
    dec.c4_hybrid_ot_epsilon = 0.07
    dec.c4_hybrid_ot_iters = 10


def _cuda_ms(fn, warmup=10, iterations=100):
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
    torch.cuda.synchronize()
    timings = []
    with torch.inference_mode():
        for _ in range(iterations):
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            timings.append(start.elapsed_time(end))
    return sum(timings) / len(timings)


def measure_latency(model, img_A, img_B, warmup=20, iterations=100):
    batch = {"im_A": img_A, "im_B": img_B}
    _, _, h, w = img_A.shape
    scale_factor = math.sqrt((h * w) / (560.0**2))
    model.eval()
    return _cuda_ms(lambda: model(batch, batched=True, scale_factor=scale_factor), warmup, iterations)


def build_model(args, weights_ckpt, resolution, enable_rotation, fh_hybrid):
    model = _build_eval_model(
        args, str(device), weights_ckpt, symmetric=False, upsample_preds=False,
    )
    dec = model.decoder
    if enable_rotation:
        dec.c4_rotation_matching = True
        dec.c4_perm_reverse = True
        install_gp_intrinsic_rotation_router(model)
        if fh_hybrid:
            _apply_fh_hybrid_decoder(model)
    else:
        dec.c4_rotation_matching = False
    model.eval()
    model.h_resized = resolution
    model.w_resized = resolution
    return model


def _prepare(args):
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    os.makedirs(os.path.join(_repo, "results"), exist_ok=True)
    weights_ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)

    img_A = torch.randn(1, 3, args.resolution, args.resolution, device=device)
    img_B = torch.randn(1, 3, args.resolution, args.resolution, device=device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    img_A = (img_A - mean) / std
    img_B = (img_B - mean) / std
    return weights_ckpt, img_A, img_B


def run_single(args, mode_name):
    """Measure a single mode. Runs inside its own subprocess so that the
    CUDA Graph memory allocated by reduce-overhead is fully reclaimed when
    the process exits (torch.cuda.empty_cache() cannot free CUDA Graphs)."""
    weights_ckpt, img_A, img_B = _prepare(args)

    enable_rot = mode_name == "rot_fh_hybrid"
    fh_hybrid = mode_name == "rot_fh_hybrid"
    model = build_model(args, weights_ckpt, args.resolution, enable_rot, fh_hybrid)
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    compiled = torch.compile(
        model, mode="reduce-overhead", fullgraph=False, dynamic=False
    )
    ms = measure_latency(compiled, img_A, img_B, warmup=args.warmup, iterations=args.iterations)
    print(f"{mode_name:>16}  {args.resolution}p  {params_m:.1f}M  {ms:.1f}ms", flush=True)


MODES = ["plain", "rot_fh_hybrid"]


def main():
    parser = argparse.ArgumentParser(description="Inference latency test")
    parser.add_argument("--weights", default=WEIGHTS)
    parser.add_argument("--resolution", type=int, default=576)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--upsample-res", type=int, default=800, dest="upsample_res")
    parser.add_argument("--no-custom-corr", action="store_true", dest="no_custom_corr")
    parser.add_argument("--single-mode", choices=MODES, default=None, dest="single_mode",
                        help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.freeze_encoder = True
    args.coarse_res = args.resolution
    args.symmetric = False
    args.upsample_preds = False

    if args.single_mode:
        # Child-process entry point: measure one mode, then exit and release GPU memory.
        run_single(args, args.single_mode)
        return

    # Parent process: use one child process per mode to avoid CUDA Graph memory
    # accumulation across modes and possible out-of-memory errors.
    script = os.path.abspath(__file__)
    for mode_name in MODES:
        cmd = [sys.executable, script, *sys.argv[1:], "--single-mode", mode_name]
        print(f"--- {mode_name} (subprocess) ---", flush=True)
        ret = subprocess.run(cmd, cwd=_repo)
        if ret.returncode != 0:
            print(f"[error] {mode_name} exited with code {ret.returncode}", file=sys.stderr)
            sys.exit(ret.returncode)


if __name__ == "__main__":
    main()
