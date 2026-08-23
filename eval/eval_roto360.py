#!/usr/bin/env python3
"""Roto360 benchmark (MMA@3/5/10)."""
import sys
import os

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_local_redimatch_path = _repo_root
sys.path.insert(0, _local_redimatch_path)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for k in list(sys.modules.keys()):
    if "redimatch" in k.lower():
        del sys.modules[k]

import argparse
import json
import torch
from argparse import ArgumentParser
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.path.join(_REPO, "models", "outdoor.pth")

from _common import install_gp_intrinsic_rotation_router
from _common import prepare_run_seed

from redimatch.benchmarks import Roto360HomogBenchmark
from redimatch.benchmarks.roto360 import prefix_metrics_for_eval_mode

_DEFAULT_ROTO360_DIR = os.path.join(_REPO, "data", "roto360")

from _common import (
    _build_eval_model,
)

def _set_c4(
    model,
    enabled: bool,
    *,
    perm_reverse: bool = True,
    symmetric_reverse: str = "align_to_query",
):
    dec = getattr(model, "decoder", None)
    if dec is None:
        return
    dec.c4_rotation_matching = bool(enabled)
    if enabled:
        dec.c4_perm_reverse = bool(perm_reverse)
        dec.c4_symmetric_reverse = str(symmetric_reverse)

def _local_corr_extension_available() -> bool:
    try:
        import local_corr  # noqa: F401
        return True
    except ImportError:
        return False

def _build_model_from_args(args, device: str, weights_ckpt):
    model = _build_eval_model(
        args,
        device,
        weights_ckpt,
        symmetric=bool(args.symmetric),
        upsample_preds=bool(args.upsample),
    )
    # C4 rotation matching (can disable with --no-c4; enable hybrid mode with --c4-hybrid)
    c4_enabled = not getattr(args, "no_c4", False)
    c4_hybrid = getattr(args, "c4_hybrid", False)
    _set_c4(
        model,
        c4_enabled,
        perm_reverse=bool(getattr(args, "c4_perm_reverse", True)),
        symmetric_reverse="align_to_query",
    )
    dec = getattr(model, "decoder", None)
    if dec is not None and c4_enabled and c4_hybrid:
        dec.c4_hybrid = True
        dec.c4_hybrid_alpha = 0.35
        dec.c4_hybrid_temp = 0.05
        dec.c4_hybrid_margin_keep = 0.03
        dec.c4_hybrid_disagree_margin = 0.03
        dec.c4_hybrid_ot_pool_size = 12
        dec.c4_hybrid_ot_epsilon = 0.07
        dec.c4_hybrid_ot_iters = 10
    if dec is not None:
        print(
            f"[C4] c4_rotation_matching={c4_enabled} hybrid={c4_hybrid}"
            + (
                f" perm_reverse={bool(dec.c4_perm_reverse)}"
                f" c4_symmetric_reverse={dec.c4_symmetric_reverse}"
                if c4_enabled
                else ""
            ),
            flush=True,
        )
    return model

def _maybe_enable_gp_intrinsic_router(model):
    if not getattr(getattr(model, "decoder", None), "c4_rotation_matching", False):
        return False
    ok, _ = install_gp_intrinsic_rotation_router(model)
    print(f"[C4] gp_intrinsic rotation router {'enabled' if ok else 'not available, fallback to default'}")
    return ok

def main():
    parser = ArgumentParser(description="0403 MaxPool Roto360 evaluation script")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=_DEFAULT_ROTO360_DIR,
        help="Roto360 dataset root directory",
    )
    parser.add_argument(
        "--eval_modes",
        type=str,
        default="all",
        help="Evaluation mode, comma-separated: i(illumination), v(viewpoint), all(full)",
    )
    parser.add_argument("--weights", default=WEIGHTS, type=str)
    parser.add_argument("--eval_resolutions", nargs="+", type=int, default=[576],
                        help="Evaluation resolution list, default 576")
    parser.add_argument("--symmetric", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--upsample", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--upsample_res", type=int, default=800)
    parser.add_argument("--c4-hybrid", dest="c4_hybrid", action="store_true", help="Enable C4 hybrid mode (disabled by default)")
    parser.add_argument(
        "--c4-perm-reverse",
        dest="c4_perm_reverse",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="C4 channel permutation direction, default True",
    )
    parser.add_argument(
        "--no_custom_corr",
        action="store_true",
        help="Disable local_corr CUDA extension, use PyTorch implementation (slower, None requires compiling extension)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed (default: random, not fixed)",
    )

    args = parser.parse_args()
    prepare_run_seed(args)

    os.makedirs("results", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.weights:
        try:
            weights = torch.load(args.weights, map_location=device, weights_only=False)
        except TypeError:
            weights = torch.load(args.weights, map_location=device)
        print(f"[Weights] {args.weights}")
    else:
        weights = None

    # Parse evaluation mode
    eval_modes = [m.strip() for m in args.eval_modes.split(",") if m.strip() in ["i", "v", "all"]]
    if not eval_modes:
        eval_modes = ["i", "v", "all"]

    res_list = list(args.eval_resolutions)

    for res in res_list:
        print(f"\n[Eval] resolution={res}")
        args.coarse_res = res
        print(
            f"[Eval] symmetric={args.symmetric} upsample={args.upsample} "
            f"c4_symmetric_reverse={"align_to_query"} "
            f"c4_rotation_matching=True rot_method=fundamental_hybrid_gp_intrinsic",
        )
        model = _build_model_from_args(args, device, weights)
        if args.weights and os.path.isfile(args.weights):
            pass  # weights already merged
        _maybe_enable_gp_intrinsic_router(model)
        model.h_resized = res
        model.w_resized = res
        model.train(False)

        dataset_path = os.path.abspath(args.dataset_path)
        if not os.path.isdir(dataset_path):
            print(f"[Error] Dataset directory not found: {dataset_path}")
            return

        for mode in eval_modes:
            suffix = {"i": "illumination", "v": "viewpoint", "all": "all"}[mode]
            out_name = f"roto360_{suffix}_r{res}.json"
            out_path = os.path.join("results", out_name)
            
            print(f"\n[Roto360] mode={mode} ({suffix}) -> {out_path}")
            
            # Create benchmark (eval_mode="i" / "v" / "all" for illumination / viewpoint / all)
            bench = Roto360HomogBenchmark(dataset_path, eval_mode=mode)
            
            print(f"[Info] Total {bench.n_target_pairs} pairs")
            
            metrics = bench.run_romav2(model, num_samples=5000, max_th=20)
            metrics = prefix_metrics_for_eval_mode(metrics, mode)
            if "mma@3" in metrics:
                print(
                    f"[MMA] mode={suffix} | MMA@3={float(metrics['mma@3']):.2f}  "
                    f"MMA@5={float(metrics['mma@5']):.2f}  "
                    f"MMA@10={float(metrics['mma@10']):.2f}  "
                    f"MMA@20={float(metrics.get('mma@20', 0.0)):.2f}",
                    flush=True,
                )
            print(metrics)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
