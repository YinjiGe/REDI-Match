"""WxBS evaluation."""

from __future__ import annotations

import sys
import os

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_local_redimatch_path = _repo_root

sys.path.insert(0, _local_redimatch_path)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for k in list(sys.modules.keys()):
    if "redimatch" in k.lower():
        del sys.modules[k]

import json
from typing import Optional

import argparse
import numpy as np
import torch
from argparse import ArgumentParser

from _common import prepare_run_seed

import redimatch

from redimatch.benchmarks import WxBSBenchmark

from _common import install_gp_intrinsic_rotation_router
from _common import (
    _build_eval_model,
)

WEIGHTS = os.path.join(_repo_root, "models", "outdoor.pth")
DEFAULT_WXBS_ROOT = os.path.join(_repo_root, "data", "WxBS", "v1.1")

def _wxbs_dataset_root_dir(path: str) -> str:
    """WxBSDataset(root) requires root/v1.1 to exist. If .../v1.1 is passed, use parent directory."""
    p = os.path.abspath(os.path.expanduser(path))
    if os.path.basename(p.rstrip(os.sep)) == "v1.1":
        return os.path.dirname(p)
    return p

def _json_safe(x):
    if x is None or isinstance(x, (bool, str)):
        return x
    if isinstance(x, (int, float)):
        return x
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            nk = str(k) if not isinstance(k, str) else k
            out[nk] = _json_safe(v)
        return out
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if isinstance(x, complex):
        return {"real": float(x.real), "imag": float(x.imag)}
    return x

def _pck_curve_index_for_px(thresholds, *, px: int, curve_len: int) -> Optional[int]:
    px_i = int(px)
    if thresholds is None:
        return px_i if 0 <= px_i < curve_len else None
    th = np.asarray(thresholds, dtype=np.float64).ravel()
    if th.size != curve_len or th.size == 0:
        return px_i if 0 <= px_i < curve_len else None
    match = np.nonzero(np.abs(th - float(px_i)) < 1e-6)[0]
    if match.size > 0:
        return int(match[0])
    if np.allclose(th, np.arange(th.size)):
        return px_i if 0 <= px_i < curve_len else None
    return None

def _wxbs_maa_at_standard_px(result_dict, thresholds, pixels=(3, 5, 10)):
    """mAA at fixed 3/5/10 px on PCK curve (raw 0..1, consistent with ``average[k]``)."""
    out = {px: None for px in pixels}
    if not isinstance(result_dict, dict) or "average" not in result_dict:
        return out
    arr = np.asarray(result_dict["average"], dtype=np.float64).ravel()
    if arr.size == 0:
        return out
    for px in pixels:
        idx = _pck_curve_index_for_px(thresholds, px=px, curve_len=len(arr))
        if idx is not None:
            out[px] = float(arr[idx])
    return out

def _extract_maa_from_wxbs(result_dict, thresholds, *, maa_threshold_px: int = 10):
    if isinstance(result_dict, dict):
        for key in ("mAA", "maa", "MAA", "mean_average_accuracy"):
            if key in result_dict:
                try:
                    return float(result_dict[key]), f"direct:{key}"
                except Exception:
                    pass
        if "average" in result_dict:
            arr = np.asarray(result_dict["average"], dtype=np.float64).ravel()
            if arr.size > 0:
                mean_curve = float(np.mean(arr))
                idx = _pck_curve_index_for_px(thresholds, px=maa_threshold_px, curve_len=len(arr))
                if idx is not None:
                    return float(arr[idx]), f"PCK@average_curve[{maa_threshold_px}px],idx={idx}"
                return mean_curve, "mean(PCK_curve[average])_fallback_bad_thresholds"
    vals = []
    if isinstance(result_dict, dict):
        for v in result_dict.values():
            if isinstance(v, (list, tuple)):
                for x in v:
                    try:
                        vals.append(float(x))
                    except Exception:
                        pass
            else:
                try:
                    vals.append(float(v))
                except Exception:
                    pass
    elif isinstance(result_dict, (list, tuple)):
        for x in result_dict:
            try:
                vals.append(float(x))
            except Exception:
                pass
    if vals:
        return float(sum(vals) / len(vals)), "fallback:mean(all_numeric_values)"
    return None, "unavailable"

def test_wxbs(
    model,
    name: str,
    *,
    wxbs_root: str,
    subset: str,
    download: bool,
    maa_threshold_px: int = 10,
):
    cfg = WxBSBenchmark.Cfg(subset=subset, dataset_path=wxbs_root, download=download)
    bench = WxBSBenchmark(cfg)
    results, thresholds = bench(model, step=0)
    maa, maa_source = _extract_maa_from_wxbs(results, thresholds, maa_threshold_px=maa_threshold_px)
    maa_std = _wxbs_maa_at_standard_px(results, thresholds, pixels=(3, 5, 10))
    mean_curve = None
    if isinstance(results, dict) and "average" in results:
        _a = np.asarray(results["average"], dtype=np.float64).ravel()
        if _a.size > 0:
            mean_curve = float(np.mean(_a))
    wrapped = {
        "name": name,
        "subset": subset,
        "mAA": maa,
        "mAA_3px": maa_std.get(3),
        "mAA_5px": maa_std.get(5),
        "mAA_10px": maa_std.get(10),
        "mAA_threshold_px": int(maa_threshold_px),
        "mAA_source": maa_source,
        "mAA_mean_over_curve": mean_curve,
        "thresholds": thresholds,
        "raw_result_dict": results,
    }
    print(f"[wxbs] {name}", flush=True)
    print(f"  mAA@{maa_threshold_px}px: {maa} ({maa_source})", flush=True)
    print(
        f"  mAA@3px: {maa_std.get(3)}  mAA@5px: {maa_std.get(5)}  mAA@10px: {maa_std.get(10)}",
        flush=True,
    )
    if mean_curve is not None:
        print(f"  PCK_curve_mean(0..19px): {mean_curve:.6f}", flush=True)
    return wrapped

def ensure_results_dir():
    os.makedirs("results", exist_ok=True)

def _set_c4(
    model,
    enabled: bool,
    *,
    perm_reverse: bool,
    symmetric_reverse: str = "align_to_query",
):
    dec = getattr(model, "decoder", None)
    if dec is None:
        return
    dec.c4_rotation_matching = bool(enabled)
    if enabled:
        dec.c4_perm_reverse = bool(perm_reverse)
        dec.c4_symmetric_reverse = str(symmetric_reverse)

def _maybe_enable_router(model, enabled: bool) -> bool:
    if not enabled:
        return False
    ok, _ = install_gp_intrinsic_rotation_router(model)
    print(f"[C4] gp_intrinsic rotation router {'enabled' if ok else 'not available, fallback to default'}")
    return bool(ok)

def _eval_one(
    *,
    args,
    device: str,
    weights,
    res: int,
    tag: str,
    enable_c4: bool,
):
    args.coarse_res = int(res)
    model = _build_eval_model(
        args,
        device,
        weights,
        symmetric=bool(args.symmetric),
        upsample_preds=bool(args.upsample),
    )
    if args.weights and os.path.isfile(args.weights):
        # weights already merged
        base = "wxbs"
    else:
        base = "wxbs"

    _set_c4(
        model,
        bool(enable_c4),
        perm_reverse=bool(True),
        symmetric_reverse="align_to_query",
    )
    _maybe_enable_router(model, bool(enable_c4))

    model.h_resized = int(res)
    model.w_resized = int(res)
    model.train(False)

    name = f"{base}_{tag}_r{res}"
    metrics = test_wxbs(
        model,
        name,
        wxbs_root=args.wxbs_root,
        subset=args.wxbs_subset,
        download=not args.no_download,
        maa_threshold_px=args.wxbs_maa_threshold_px,
    )
    del model
    torch.cuda.empty_cache()
    return {
        "resolution": int(res),
        "tag": tag,
        "c4_rotation_matching": bool(enable_c4),
        "name": name,
        "mAA": metrics.get("mAA"),
        "mAA_source": metrics.get("mAA_source"),
        "metrics": metrics,
    }

if __name__ == "__main__":
    parser = ArgumentParser(description="WxBS multires eval (C4 off/on)")
    parser.add_argument("--weights", default=WEIGHTS, type=str)
    parser.add_argument(
        "--wxbs_root",
        type=str,
        default=DEFAULT_WXBS_ROOT,
        help="WxBS dataset root (downloaded separately)",
    )
    parser.add_argument("--eval_resolutions", nargs="+", type=int, default=[576],
                        help="Evaluation resolution list, default 576")
    parser.add_argument("--symmetric", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--upsample", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--no_custom_corr", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    args.wxbs_root = os.path.abspath(os.path.expanduser(args.wxbs_root))
    args.wxbs_subset = getattr(args, "wxbs_subset", None) or "all"
    args.wxbs_maa_threshold_px = getattr(args, "wxbs_maa_threshold_px", None) or 3
    args.no_download = getattr(args, "no_download", True)
    prepare_run_seed(args)
    args.freeze_encoder = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = torch.load(args.weights, map_location=device, weights_only=False)

    ensure_results_dir()
    summary = []
    for res in args.eval_resolutions:
        summary.append(_eval_one(args=args, device=device, weights=weights, res=int(res), tag="wxbs", enable_c4=False))

    print("\n[Summary]", flush=True)
    for s in summary:
        print(s, flush=True)
