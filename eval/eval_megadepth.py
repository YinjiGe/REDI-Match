#!/usr/bin/env python3
"""MegaDepth multi-resolution evaluation (plain / rot / dense)."""

from __future__ import annotations

import os
import sys
import json
import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

_exp = os.path.abspath(os.path.dirname(__file__))
_local_redimatch_path = os.path.abspath(os.path.join(_exp, ".."))

sys.path.insert(0, _local_redimatch_path)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _exp)

for k in list(sys.modules.keys()):
    kl = k.lower()
    if "redimatch" in kl or "megadepth_c4_hybrid" in kl or "mega1500_gp_intrinsic" in kl:
        del sys.modules[k]

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

import torch
import tqdm
import redimatch
from torch.utils.data import DataLoader, WeightedRandomSampler
from redimatch.benchmarks import (
    MegaDepthPoseEstimationBenchmark,
    MegaDepthPoseEstimationRotBenchmark,
    MegadepthDenseBenchmark,
)
from redimatch.utils.utils import tensor_to_pil

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.path.join(_REPO, "models", "outdoor.pth")

from _common import get_rng_seed_info
from _common import prepare_run_seed
from _common import (
    install_gp_intrinsic_rotation_router,
    eval_mega1500,
)
from _common import ensure_results_dir

from _common import (
    _build_eval_model,
)


@dataclass(frozen=True)
class MegadepthDenseLayout:
    """Resolved image/depth root and scene-info directory for dense evaluation."""

    data_root: str
    scene_info_root: str
    asset_fallback_root: Optional[str] = None
    symlinks_created: tuple[str, ...] = ()


ROT2_DENSE_PSI_DIR = "prep_scene_info_dense"
ROT2_DENSE_MARKER = ".rot2_dense_ready"
ROT_PREP_SCENE_INFO_MARKER = ".rot_prep_scene_info"


def _is_usable_scene_info_dir(path: str) -> bool:
    return os.path.isdir(path) and any(
        name.endswith(".npy") for name in os.listdir(path)
    )


def _resolve_prep_scene_info_root(data_root: str, scene_local: str) -> str:
    return scene_local


def is_megadepth_rot2_fixed_a_root(data_root: str) -> bool:
    """Return whether a root uses the optional fixed-A rot2 dense layout."""
    return os.path.basename(os.path.normpath(data_root)).lower() in {
        "megadepth_rot2",
        "megadepth_rot_2",
    }


def _try_symlink(source: str, target: str) -> bool:
    try:
        os.symlink(source, target, target_is_directory=True)
        return True
    except FileExistsError:
        return False


def _make_layout(
    *,
    data_root: str,
    scene_info_root: str,
    asset_fallback_root: Optional[str] = None,
    symlinks_created: tuple[str, ...] = (),
) -> MegadepthDenseLayout:
    return MegadepthDenseLayout(
        data_root=os.path.abspath(data_root),
        scene_info_root=os.path.abspath(scene_info_root),
        asset_fallback_root=asset_fallback_root,
        symlinks_created=symlinks_created,
    )


# Inlined dense_layout:
def resolve_megadepth_dense_layout(
    data_root: str,
    *,
    asset_fallback_roots: Optional[Sequence[str]] = None,
    auto_symlink: bool = True,
) -> MegadepthDenseLayout:
    """
    Resolve ``data_root`` (base for image/depth relative paths) and ``scene_info_root`` for dense evaluation.

    - If ``{data_root}/prep_scene_info`` already exists → use directly;
    - Otherwise search ``asset_fallback_roots`` for reusable ``prep_scene_info`` and ``phoenix``,
      and create symlinks under ``data_root`` (only when ``auto_symlink=True``).
    """
    data_root = os.path.abspath(os.path.expanduser(str(data_root)))

    if is_megadepth_rot2_fixed_a_root(data_root):
        dense_psi = os.path.join(data_root, ROT2_DENSE_PSI_DIR)
        dense_marker = os.path.join(dense_psi, ROT2_DENSE_MARKER)
        if _is_usable_scene_info_dir(dense_psi) and os.path.isfile(dense_marker):
            return _make_layout(
                data_root=data_root,
                scene_info_root=dense_psi,
            )
        raise FileNotFoundError(
            f"megadepth_rot_2 dense needs dedicated {ROT2_DENSE_PSI_DIR}/(does not read prep_scene_info symlink, nor modify original megadepth).\n"
            f"  data_root={data_root}\n"
            f"  Please prepare the rot2 dense scene-info files first, then rerun this evaluation."
        )

    scene_local = os.path.join(data_root, "prep_scene_info")
    scene_info_root = _resolve_prep_scene_info_root(data_root, scene_local)
    rot_marker = os.path.join(scene_info_root, ROT_PREP_SCENE_INFO_MARKER)

    phoenix_local = os.path.join(data_root, "phoenix")
    if os.path.isfile(rot_marker) and os.path.islink(phoenix_local):
        raise FileNotFoundError(
            f"{phoenix_local} must not be a symlink to plain (would mix rot and plain depths)."
            f"Please delete that symlink and run prepare_megadepth_rot_prep_scene_info.py to generate rot depths."
        )

    if _is_usable_scene_info_dir(scene_info_root) and os.path.isfile(rot_marker):
        return _make_layout(
            data_root=data_root,
            scene_info_root=scene_info_root,
        )

    if _is_usable_scene_info_dir(scene_info_root):
        return _make_layout(
            data_root=data_root,
            scene_info_root=scene_info_root,
        )

    scene_local = scene_info_root

    fallbacks: List[str] = []
    if asset_fallback_roots:
        fallbacks.extend(os.path.abspath(os.path.expanduser(str(p))) for p in asset_fallback_roots)
    if DEFAULT_MEGADEPTH_FULL not in fallbacks:
        fallbacks.append(DEFAULT_MEGADEPTH_FULL)

    symlinks: List[str] = []
    chosen_fallback: Optional[str] = None

    for fb in fallbacks:
        if not fb or not os.path.isdir(fb):
            continue
        psi_fb = os.path.join(fb, "prep_scene_info")
        if not _is_usable_scene_info_dir(psi_fb):
            continue
        chosen_fallback = fb
        if not auto_symlink:
            return _make_layout(
                data_root=data_root,
                scene_info_root=psi_fb,
                asset_fallback_root=fb,
            )
        if not _is_usable_scene_info_dir(scene_local):
            if _try_symlink(psi_fb, scene_local):
                symlinks.append(f"{scene_local} -> {psi_fb}")
            scene_local = os.path.join(data_root, "prep_scene_info")
        phoenix_fb = os.path.join(fb, "phoenix")
        if not os.path.isfile(rot_marker):
            phoenix_local = os.path.join(data_root, "phoenix")
            if os.path.isdir(phoenix_fb) and not os.path.lexists(phoenix_local):
                if _try_symlink(phoenix_fb, phoenix_local):
                    symlinks.append(f"{phoenix_local} -> {phoenix_fb}")
        break

    if not _is_usable_scene_info_dir(scene_local):
        fb_hint = ", ".join(fallbacks[:3])
        raise FileNotFoundError(
            f"Megadepth dense needs usable prep_scene_info (containing 0015.npy, 0022.npy, etc.).\n"
            f"  data_root={data_root}\n"
            f"  not found {scene_local}/，and not completed via fallback: {fb_hint}\n"
            f"Can specify full MegaDepth root, or manually symlink:\n"
            f"  ln -s {DEFAULT_MEGADEPTH_FULL}/prep_scene_info {scene_local}\n"
            f"  ln -s {DEFAULT_MEGADEPTH_FULL}/phoenix {os.path.join(data_root, 'phoenix')}"
        )

    return _make_layout(
        data_root=data_root,
        scene_info_root=scene_local,
        symlinks_created=tuple(symlinks),
        asset_fallback_root=chosen_fallback,
    )

# _DENSE_ASSET_FALLBACK defined below

DEFAULT_RESOLUTIONS = [576]
DEFAULT_MEGADEPTH_ROOT_PLAIN = os.path.join(_REPO, "data", "megadepth")
DEFAULT_MEGADEPTH_ROOT_ROT = os.path.join(_REPO, "data", "megadepth_rot")
DEFAULT_MEGADEPTH_FULL = os.path.join(_REPO, "data", "megadepth")
_DENSE_ASSET_FALLBACK = DEFAULT_MEGADEPTH_FULL

def _set_c4(model, enabled: bool, *, perm_reverse: bool = True, symmetric_reverse: str = "align_to_query"):
    dec = getattr(model, "decoder", None)
    if dec is None:
        return
    dec.c4_rotation_matching = bool(enabled)
    if enabled:
        dec.c4_perm_reverse = bool(perm_reverse)
        dec.c4_symmetric_reverse = str(symmetric_reverse)

def _build_model_for_eval(args, device: str):
    weights = torch.load(args.weights, map_location=device, weights_only=False)
    model = _build_eval_model(
        args,
        device,
        weights,
        symmetric=bool(args.symmetric),
        upsample_preds=bool(args.upsample),
    )
    # weights already merged
    model.eval()
    return model

def _dense_use_rot_variant(args, data_root: str) -> bool:
    variant = str(getattr(args, "dense_variant", "auto"))
    if variant == "rot":
        return True
    if variant == "plain":
        return False
    abs_root = os.path.abspath(data_root)
    rot_root = os.path.abspath(os.path.expanduser(str(args.mega_data_root_rot)))
    if abs_root == rot_root:
        return True
    base = os.path.basename(os.path.normpath(abs_root)).lower()
    return "megadepth_rot" in base

def _resolve_dense_data_root(args) -> str:
    if args.dense_data_root:
        return os.path.abspath(os.path.expanduser(str(args.dense_data_root)))
    if args.mode == "plain":
        return os.path.abspath(os.path.expanduser(str(args.mega_data_root_plain)))
    return os.path.abspath(os.path.expanduser(str(args.mega_data_root_rot)))

def _load_rotation_gt_map(gt_json: str) -> Dict[str, int]:
    """Load rotation_gt.json -> {rel_path: physical_k}.
    rotation_gt.json stores physical k (B must rotate clockwise kx90 to align with A frame).
    Note: when used for C4 alignment, need to convert to c4_k = (4 - physical_k) % 4.
    """
    data = json.loads(Path(gt_json).read_text(encoding="utf-8"))
    records = data.get("records") or data.get("pairs") or []
    out: Dict[str, int] = {}
    for rec in records:
        rel = str(rec.get("rel_path", "")).replace("\\", "/")
        if rel.startswith("Undistorted_SfM/"):
            rel = rel.split("Undistorted_SfM/", 1)[1]
        if rel:
            out[rel] = int(rec["k"]) % 4
    return out

def _resolve_rotation_gt_json(args, data_root: str) -> Optional[str]:
    p = str(getattr(args, "rotation_gt_json", "auto"))
    if p == "auto":
        local = os.path.join(os.path.abspath(data_root), "rotation_gt.json")
        return local if os.path.isfile(local) else None
    return p if os.path.isfile(p) else None

def _dense_scene_kwargs(args, *, use_rot: bool, data_root: str) -> dict:
    resize_mode = str(getattr(args, "dense_resize", "stretch")).lower()
    tensor_rot = bool(getattr(args, "dense_rot_tensor_equivariant", False)) and use_rot
    if bool(getattr(args, "dense_rot_equivariant", False)) and use_rot:
        tensor_rot = True
    kw: dict = {"resize_mode": resize_mode}
    if tensor_rot:
        gt_path = _resolve_rotation_gt_json(args, data_root)
        if not gt_path:
            raise FileNotFoundError(
                f"dense rot tensor equivariant requires rotation_gt.json (data_root={data_root}）"
            )
        kw["rotation_gt_map"] = _load_rotation_gt_map(gt_path)
        kw["rot_b_tensor_equivariant"] = True
    return kw

def _dense_scene_kwargs_with_layout(args, *, use_rot: bool, layout) -> dict:
    kw = _dense_scene_kwargs(args, use_rot=use_rot, data_root=layout.data_root)
    if kw.get("rot_b_tensor_equivariant"):
        plain_root = os.path.abspath(
            os.path.expanduser(str(getattr(args, "mega_data_root_plain", "") or ""))
        )
        if not os.path.isdir(plain_root):
            plain_root = ""
        fb = getattr(layout, "asset_fallback_root", None) or _DENSE_ASSET_FALLBACK
        kw["plain_asset_root"] = plain_root or os.path.abspath(os.path.expanduser(str(fb)))
    return kw

def _dense_match_batch(model, im_a: torch.Tensor, im_b: torch.Tensor):
    return model.match(im_a, im_b, batched=True)

def _sanitize_pair_dirname(name: str, *, max_len: int = 80) -> str:
    s = re.sub(r"[^\w.\-]+", "_", str(name)).strip("_")
    return s[:max_len] if s else "pair"

def _per_item_dense_metrics(
    bench: MegadepthDenseBenchmark,
    depth1: torch.Tensor,
    depth2: torch.Tensor,
    t12: torch.Tensor,
    k1: torch.Tensor,
    k2: torch.Tensor,
    matches: torch.Tensor,
) -> List[dict]:
    """Per-sample EPE / PCK (each pair in batch computed separately)."""
    b = int(matches.shape[0])
    rows: List[dict] = []
    for i in range(b):
        gd, p1, p3, p5, prob = bench.geometric_dist(
            depth1[i : i + 1],
            depth2[i : i + 1],
            t12[i : i + 1],
            k1[i : i + 1],
            k2[i : i + 1],
            matches[i : i + 1],
        )
        rows.append(
            {
                "epe": float(gd.mean().item()) if gd.numel() else None,
                "pck1": float(p1.item()) if gd.numel() else None,
                "pck3": float(p3.item()) if gd.numel() else None,
                "pck5": float(p5.item()) if gd.numel() else None,
                "n_valid": int(prob.sum().item()),
            }
        )
    return rows

def _batch_field(data: dict, key: str, idx: int):
    val = data.get(key)
    if val is None:
        return None
    if isinstance(val, (list, tuple)):
        return val[idx]
    if isinstance(val, torch.Tensor):
        return val[idx].item() if val.ndim == 0 else val[idx]
    return val

def _save_big_epe_pair(
    *,
    save_root: str,
    pair_dir: str,
    im_a: torch.Tensor,
    im_b: torch.Tensor,
    meta: dict,
) -> str:
    out_dir = os.path.join(save_root, pair_dir)
    os.makedirs(out_dir, exist_ok=True)
    tensor_to_pil(im_a.detach().cpu(), unnormalize=True).save(
        os.path.join(out_dir, "im_A.jpg"), quality=95
    )
    tensor_to_pil(im_b.detach().cpu(), unnormalize=True).save(
        os.path.join(out_dir, "im_B.jpg"), quality=95
    )
    return out_dir

def _benchmark_dense_with_big_epe_save(
    bench: MegadepthDenseBenchmark,
    model,
    *,
    batch_size: int,
    big_epe_threshold: float,
    big_epe_dir: str,
    tag: str,
    coarse_res: int,
    meta_base: dict,
) -> tuple[dict, List[dict]]:
    """Megadepth dense evaluation; pairs with EPE above threshold written to ``big_epe_dir/{tag}_r{res}/``."""
    model.train(False)
    gd_tot = 0.0
    pck_1_tot = 0.0
    pck_3_tot = 0.0
    pck_5_tot = 0.0
    save_root = os.path.join(
        os.path.abspath(big_epe_dir), f"{tag}_r{int(coarse_res)}"
    )
    os.makedirs(save_root, exist_ok=True)

    sampler = WeightedRandomSampler(
        torch.ones(len(bench.dataset)),
        replacement=False,
        num_samples=int(bench.num_samples),
    )
    dataloader = DataLoader(
        bench.dataset,
        batch_size=int(batch_size),
        num_workers=int(batch_size),
        sampler=sampler,
    )

    big_epe_rows: List[dict] = []
    pair_counter = 0

    for batch_idx, data in enumerate(
        tqdm.tqdm(dataloader, disable=redimatch.RANK > 0)
    ):
        im_a = data["im_A"].cuda()
        im_b = data["im_B"].cuda()
        depth1 = data["im_A_depth"].cuda()
        depth2 = data["im_B_depth"].cuda()
        t12 = data["T_1to2"].cuda()
        k1 = data["K1"].cuda()
        k2 = data["K2"].cuda()

        matches, _cert = _dense_match_batch(model, im_a, im_b)
        if getattr(model, "symmetric", False):
            matches = MegadepthDenseBenchmark._symmetric_crop_a2b(matches)

        # Summary metrics fully consistent with MegadepthDenseBenchmark.benchmark (weighted by valid pixels)
        gd, pck_1, pck_3, pck_5, prob = bench.geometric_dist(
            depth1, depth2, t12, k1, k2, matches
        )
        gd_tot += gd.mean()
        pck_1_tot += pck_1
        pck_3_tot += pck_3
        pck_5_tot += pck_5

        # Per-pair EPE only used for big_epe save decision
        item_metrics = _per_item_dense_metrics(
            bench, depth1, depth2, t12, k1, k2, matches
        )
        c4_info = getattr(model.decoder, "_last_c4_rotation", None)
        c4_rot_idx = None
        c4_margin = None
        if c4_info is not None:
            if c4_info.get("rot_idx") is not None:
                c4_rot_idx = [
                    int(x) % 4
                    for x in torch.as_tensor(c4_info["rot_idx"]).reshape(-1).tolist()
                ]
            if c4_info.get("rot_margin") is not None:
                c4_margin = [
                    float(x)
                    for x in torch.as_tensor(c4_info["rot_margin"]).reshape(-1).tolist()
                ]

        for i, m in enumerate(item_metrics):
            epe = m.get("epe")
            if epe is None or epe <= float(big_epe_threshold):
                continue
            id_a = _batch_field(data, "im_A_identifier", i) or f"A{i}"
            id_b = _batch_field(data, "im_B_identifier", i) or f"B{i}"
            pair_dir = (
                f"{pair_counter:06d}_{_sanitize_pair_dirname(id_a)}"
                f"__{_sanitize_pair_dirname(id_b)}"
            )
            pair_counter += 1
            meta = {
                **meta_base,
                "batch_idx": int(batch_idx),
                "item_idx": int(i),
                "im_A_identifier": id_a,
                "im_B_identifier": id_b,
                "im_A_path": _batch_field(data, "im_A_path", i),
                "im_B_path": _batch_field(data, "im_B_path", i),
                "epe": epe,
                "pck1": m.get("pck1"),
                "pck3": m.get("pck3"),
                "pck5": m.get("pck5"),
                "n_valid": m.get("n_valid"),
                "big_epe_threshold": float(big_epe_threshold),
                "c4_rot_idx": c4_rot_idx[i] if c4_rot_idx and i < len(c4_rot_idx) else None,
                "c4_margin": c4_margin[i] if c4_margin and i < len(c4_margin) else None,
            }
            out_dir = _save_big_epe_pair(
                save_root=save_root,
                pair_dir=pair_dir,
                im_a=im_a[i],
                im_b=im_b[i],
                meta=meta,
            )
            big_epe_rows.append({**meta, "pair_dir": pair_dir, "saved_to": out_dir})

    n_batches = max(len(dataloader), 1)
    _tot = lambda x: float(x.item()) if torch.is_tensor(x) else float(x)
    metrics = {
        "epe": _tot(gd_tot) / n_batches,
        "mega_pck_1": _tot(pck_1_tot) / n_batches,
        "mega_pck_3": _tot(pck_3_tot) / n_batches,
        "mega_pck_5": _tot(pck_5_tot) / n_batches,
        "big_epe_saved": len(big_epe_rows),
        "big_epe_threshold": float(big_epe_threshold),
        "big_epe_dir": save_root,
    }
    if big_epe_rows:
        print(
            f"[Dense] EPE>{big_epe_threshold}: saved {len(big_epe_rows)} pairs -> {save_root}",
            flush=True,
        )
    return metrics, big_epe_rows

def _evaluate_dense(
    *,
    args,
    device: str,
    coarse_res: int,
    data_root: str,
    layout,
    use_rot: bool,
    tag_suffix: str = "",
) -> dict:
    """MegadepthDenseBenchmark: dense warp geometric error (EPE / mega_pck_*)."""
    args.coarse_res = coarse_res
    model = _build_model_for_eval(args, device)
    enable_c4 = bool(use_rot)
    _set_c4(
        model,
        enable_c4,
        perm_reverse=bool(True),
        symmetric_reverse="align_to_query",
    )
    scene_kw = _dense_scene_kwargs_with_layout(args, use_rot=use_rot, layout=layout)
    if (
        use_rot
        and scene_kw.get("resize_mode") == "letterbox"
    ):
        print(
            "[Dense] Warning: letterbox mismatches with prep_scene_info_dense / training stretch 576,"
            "dense EPE will be systematically higher; for equivariant dense use default stretch + --dense-rot-equivariant",
            flush=True,
        )
    print(
        f"[Dense] data_root={layout.data_root} scene_info={layout.scene_info_root} "
        f"c4_rotation_matching={enable_c4} symmetric={bool(args.symmetric)} "
        f"resize={scene_kw.get('resize_mode', 'stretch')} "
        f"rot_b_tensor={scene_kw.get('rot_b_tensor_equivariant', False)} "
        f"(symmetric mode: benchmark auto-crops A->B left half)",
        flush=True,
    )
    if layout.symlinks_created:
        for ln in layout.symlinks_created:
            print(f"[Dense] symlink: {ln}", flush=True)
    model.h_resized = int(coarse_res)
    model.w_resized = int(coarse_res)
    model.train(False)
    if enable_c4:
        install_gp_intrinsic_rotation_router(
            model,
            gp_entropy_pool_size=int(12),
            gp_entropy_temp=float(0.05),
        )

    num_samples = int(args.dense_num_samples)
    bench = MegadepthDenseBenchmark(
        data_root=layout.data_root,
        h=int(coarse_res),
        w=int(coarse_res),
        num_samples=num_samples,
        scene_info_root=layout.scene_info_root,
        scene_kwargs=scene_kw,
    )
    tag = ("rot" if use_rot else "plain") + str(tag_suffix)
    experiment_name = f"{args.experiment_prefix}_dense_{tag}"
    meta_base = {
        "weights_path": args.weights,
                "coarse_res": int(coarse_res),
        "data_root": layout.data_root,
        "scene_info_root": layout.scene_info_root,
        "dense_variant": tag,
        "symmetric": bool(args.symmetric),
        "dense_c4_rotation_matching": enable_c4,
        "dense_resize": scene_kw.get("resize_mode", "stretch"),
        "dense_rot_b_tensor_equivariant": scene_kw.get("rot_b_tensor_equivariant", False),
    }
    metrics, _big_rows = _benchmark_dense_with_big_epe_save(
        bench,
        model,
        batch_size=int(args.dense_batch_size),
        big_epe_threshold=float(args.dense_big_epe_threshold),
        big_epe_dir=str(args.dense_big_epe_dir),
        tag=tag,
        coarse_res=int(coarse_res),
        meta_base=meta_base,
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "dataset_tag": "dense",
        "mega_data_root": layout.data_root,
        "dense_variant": tag,
        "resolution": coarse_res,
        "experiment_name": experiment_name,
        "results": dict(metrics),
    }

def _evaluate_plain(
    *,
    args,
    device: str,
    coarse_res: int,
    mega_data_root: str,
) -> dict:
    args.coarse_res = coarse_res
    model = _build_model_for_eval(args, device)
    _set_c4(
        model,
        False,
        perm_reverse=bool(True),
        symmetric_reverse="align_to_query",
    )
    experiment_name = f"{args.experiment_prefix}_plain"
    bench = MegaDepthPoseEstimationBenchmark(mega_data_root)
    results = bench.benchmark(model, model_name=experiment_name)
    print(f"[MegaDepth] {experiment_name}")
    print(results, flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "dataset_tag": "plain",
        "c4_mode": "disabled",
        "mega_data_root": mega_data_root,
        "resolution": coarse_res,
        "experiment_name": experiment_name,
        "results": results,
    }

def _evaluate_rot_c4_mode(
    *,
    args,
    device: str,
    coarse_res: int,
    mega_data_root: str,
    c4_mode: str,
) -> dict:
    args.coarse_res = coarse_res

    # ── rot_no_c4: completely disable C4 rotation matching on rot data ──
    if c4_mode == "rot_no_c4":
        model = _build_model_for_eval(args, device)
        _set_c4(model, False)
        model.h_resized = int(coarse_res)
        model.w_resized = int(coarse_res)
        model.train(False)
        experiment_name = f"{args.experiment_prefix}_rot_noc4"
        bench = MegaDepthPoseEstimationRotBenchmark(mega_data_root)
        results = bench.benchmark(model, model_name=experiment_name)
        print(f"[MegaDepth] {experiment_name}")
        print(results, flush=True)
        del model
        torch.cuda.empty_cache()
        return {
            "dataset_tag": "rot_no_c4",
            "c4_mode": "rot_no_c4",
            "mega_data_root": mega_data_root,
            "resolution": coarse_res,
            "experiment_name": experiment_name,
            "results": results,
        }

    model = _build_model_for_eval(args, device)
    _set_c4(
        model,
        True,
        perm_reverse=bool(True),
        symmetric_reverse="align_to_query",
    )
    experiment_name = f"{args.experiment_prefix}_rot"
    results = eval_mega1500(
        model,
        experiment_name,
        data_root=mega_data_root,
        rot=True,
        fh_hybrid_alpha=float(args.fh_hybrid_alpha),
        fh_hybrid_temp=float(args.fh_hybrid_temp),
        fh_margin_keep=float(args.fh_margin_keep),
        fh_hybrid_disagree_margin=float(args.fh_hybrid_disagree_margin),
        fh_ot_pool_size=int(args.fh_ot_pool_size),
        fh_ot_epsilon=float(args.fh_ot_epsilon),
        fh_ot_iters=int(args.fh_ot_iters),
        gp_entropy_pool_size=int(12),
        gp_entropy_temp=float(0.05),
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "dataset_tag": "rot",
        "c4_mode": c4_mode,
        "mega_data_root": mega_data_root,
        "resolution": coarse_res,
        "experiment_name": experiment_name,
        "results": results,
    }

def _apply_fh_hybrid_decoder(model, args) -> None:
    dec = model.decoder
    dec.c4_hybrid = True
    dec.c4_hybrid_alpha = float(args.fh_hybrid_alpha)
    dec.c4_hybrid_temp = float(args.fh_hybrid_temp)
    dec.c4_hybrid_margin_keep = float(args.fh_margin_keep)
    dec.c4_hybrid_disagree_margin = float(args.fh_hybrid_disagree_margin)
    dec.c4_hybrid_ot_pool_size = int(args.fh_ot_pool_size)
    dec.c4_hybrid_ot_epsilon = float(args.fh_ot_epsilon)
    dec.c4_hybrid_ot_iters = int(args.fh_ot_iters)

_DEFAULT_LATENCY_IM_A = os.path.join(_local_redimatch_path, "assets", "toronto_A.jpg")
_DEFAULT_LATENCY_IM_B = os.path.join(_local_redimatch_path, "assets", "toronto_B_rot180.jpg")

def _get_latency_tensors(args, resolution: int, device: str):
    from redimatch.utils import get_tuple_transform_ops
    from PIL import Image
    tf = get_tuple_transform_ops(resize=(resolution,resolution), normalize=True)
    def _make_pair(a,b):
        return {"im_A": tf(Image.open(a).convert("RGB")).unsqueeze(0).to(device), "im_B": tf(Image.open(b).convert("RGB")).unsqueeze(0).to(device)}

    ua = getattr(args, "latency_im_A", None)
    ub = getattr(args, "latency_im_B", None)
    if ua and ub:
        batch = make_image_pair_batch(im_a_path=ua, im_b_path=ub, resolution=resolution, device=device)
        return batch["im_A"], batch["im_B"]
    if os.path.isfile(_DEFAULT_LATENCY_IM_A) and os.path.isfile(_DEFAULT_LATENCY_IM_B):
        batch = make_image_pair_batch(
            im_a_path=_DEFAULT_LATENCY_IM_A,
            im_b_path=_DEFAULT_LATENCY_IM_B,
            resolution=resolution,
            device=device,
        )
        return batch["im_A"], batch["im_B"]
    img_A = torch.randn(1, 3, resolution, resolution, device=device)
    img_B = torch.randn(1, 3, resolution, resolution, device=device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return (img_A - mean) / std, (img_B - mean) / std

def _measure_rot_latency_ms(
    model,
    img_A: torch.Tensor,
    img_B: torch.Tensor,
    args,
    *,
    c4_mode: str,
    warmup: int,
    iterations: int,
) -> float:
    def measure_cuda_ms(fn, warmup=10, iters=100):
        import torch, time
        for _ in range(warmup): fn()
        torch.cuda.synchronize(); start = time.perf_counter()
        for _ in range(iters): fn()
    torch.cuda.synchronize()
    return {"mean_ms": (time.perf_counter()-start)/iters*1000}

    _, _, h, w = img_A.shape
    scale_factor = math.sqrt((h * w) / (560.0**2))
    batch = {"im_A": img_A, "im_B": img_B}
    model.eval()

    if c4_mode == "gp":
        _apply_fh_hybrid_decoder(model, args)

        def _fn():
            model(batch, batched=True, scale_factor=scale_factor)

    else:
        raise ValueError(c4_mode)

    stats = measure_cuda_ms(_fn, warmup=warmup, iters=iterations)
    return float(stats["mean_ms"])

def _run_latency_compare(
    args, device: str, resolution: int, *, rot_c4_modes: List[str]
) -> List[dict]:
    args.coarse_res = resolution
    model = _build_model_for_eval(args, device)
    _set_c4(model, True, perm_reverse=bool(True), symmetric_reverse="align_to_query")
    install_gp_intrinsic_rotation_router(
        model,
        gp_entropy_pool_size=int(12),
        gp_entropy_temp=float(0.05),
    )

    img_A, img_B = _get_latency_tensors(args, resolution, device)

    rows = []
    for c4_mode in rot_c4_modes:
        print(f"[Latency] {c4_mode} @ r={resolution}", flush=True)
        ms = _measure_rot_latency_ms(
            model,
            img_A,
            img_B,
            args,
            c4_mode=c4_mode,
            warmup=int(args.latency_warmup),
            iterations=int(args.latency_iterations),
        )
        rows.append(
            {
                "resolution": resolution,
                "c4_mode": c4_mode,
                "latency_ms": ms,
                "latency_warmup": int(args.latency_warmup),
                "latency_iterations": int(args.latency_iterations),
            }
        )
        print(f"  -> {ms:.3f} ms", flush=True)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MegaDepth plain/rot evaluation."
    )
    parser.add_argument("--weights", type=str, default=WEIGHTS)
    parser.add_argument("--eval_resolutions", nargs="+", type=int, default=DEFAULT_RESOLUTIONS)
    parser.add_argument("--upsample_res", type=int, default=800)
    parser.add_argument("--upsample", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--experiment_prefix", type=str, default="megadepth")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no_scale", action="store_true", default=True)
    parser.add_argument("--symmetric", action="store_true", default=True)
    parser.add_argument("--no_custom_corr", action="store_true")
    parser.add_argument("--c4_perm_reverse", action=argparse.BooleanOptionalAction, default=True)
    # c4_symmetric_reverse = "align_to_query" (default, not configurable)
    parser.add_argument("--mega_data_root_plain", type=str, default=DEFAULT_MEGADEPTH_ROOT_PLAIN)
    parser.add_argument("--mega_data_root_rot", type=str, default=DEFAULT_MEGADEPTH_ROOT_ROT)
    parser.add_argument(
        "--mode",
        type=str,
        choices=("plain", "rot"),
        default="plain",
        help="plain=megadepth only (default); rot=megadepth_rot only.",
    )
    parser.add_argument("--rotation-gt-json", type=str, default="auto")
    parser.add_argument("--mega1500-tta-cert-thresh", type=float, default=0.1)
    parser.add_argument("--mega1500-hybrid-multi-margin-thresh", type=float, default=0.02)
    parser.add_argument("--fh-hybrid-alpha", type=float, default=0.35)
    parser.add_argument("--fh-hybrid-temp", type=float, default=0.05)
    parser.add_argument("--fh-margin-keep", type=float, default=0.03)
    parser.add_argument("--fh-hybrid-disagree-margin", type=float, default=0.03)
    parser.add_argument("--fh-ot-pool-size", type=int, default=12)
    parser.add_argument("--fh-ot-epsilon", type=float, default=0.07)
    parser.add_argument("--fh-ot-iters", type=int, default=10)
    parser.add_argument("--gp-entropy-pool-size", type=int, default=12)
    parser.add_argument("--gp-entropy-temp", type=float, default=0.05)
    parser.add_argument("--run-latency", action="store_true", help="After evaluation, measure inference latency for both rot C4 methods.")
    parser.add_argument("--latency-only", action="store_true", help="measure latency only, skip Mega1500 benchmark.")
    parser.add_argument("--latency-warmup", type=int, default=10)
    parser.add_argument("--latency-iterations", type=int, default=100)
    parser.add_argument("--latency-im-A", type=str, default=None)
    parser.add_argument("--latency-im-B", type=str, default=None)
    parser.add_argument(
        "--run-dense",
        action="store_true",
        help="run MegadepthDenseBenchmark (EPE / PCK@1,3,5); default no longer runs Mega1500.",
    )
    parser.add_argument(
        "--with-mega1500",
        action="store_true",
        help="When combined with --run-dense, also run Mega1500 plain/rot (old behavior: pose first then dense).",
    )
    parser.add_argument(
        "--dense-data-root",
        type=str,
        default=None,
        help=(
            "Root directory for dense image/depth relative paths (e.g., Megadepth_rot)."
            "when missing prep_scene_info/phoenix, complete via --dense-asset-fallback symlinks."
        ),
    )
    parser.add_argument(
        "--dense-asset-fallback",
        type=str,
        default=DEFAULT_MEGADEPTH_FULL,
        help="Full MegaDepth root (contains prep_scene_info, phoenix), for completing rot/simplified roots.",
    )
    parser.add_argument(
        "--dense-variant",
        type=str,
        choices=("auto", "plain", "rot"),
        default="auto",
        help="use plain or rot data for dense; auto: rot/all->megadepth_rot, plain->megadepth_plain.",
    )
    parser.add_argument(
        "--dense-no-auto-symlink",
        action="store_true",
        help="Disable auto symlink for prep_scene_info/phoenix (only use existing dirs or fallback read-only paths).",
    )
    parser.add_argument("--dense-num-samples", type=int, default=2000, dest="dense_num_samples")
    parser.add_argument("--dense-batch-size", type=int, default=4, dest="dense_batch_size")
    parser.add_argument(
        "--dense-big-epe-threshold",
        type=float,
        default=20.0,
        help="When a single pair's mean EPE exceeds this, save to --dense-big-epe-dir (<=0 disables).",
    )
    parser.add_argument(
        "--dense-big-epe-dir",
        type=str,
        default="results/big_epe",
        help="Root directory for saving EPE above-threshold pairs; actual path is {dir}/{plain|rot}_r{res}/{idx}_A__B/.",
    )
    parser.add_argument(
        "--dense-resize",
        type=str,
        choices=("stretch", "letterbox"),
        default="stretch",
        help="dense dataset resize: stretch=original 576 stretch; letterbox=keep aspect ratio pad.",
    )
    parser.add_argument(
        "--dense-rot-equivariant",
        action="store_true",
        help="rot dense equivariant: stretch + rot90(preprocess(Bplain)) tensor + C4 (requires rotation_gt.json).",
    )
    parser.add_argument(
        "--dense-rot-tensor-equivariant",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="B side reads plain image, tensor rot90(k_gt) after resize; typically enabled together with --dense-rot-equivariant.",
    )
    parser.add_argument(
        "--dense-with-plain-baseline",
        action="store_true",
        help="When evaluating rot dense, additionally run plain megadepth (same resize setting) for EPE comparison.",
    )
    args = parser.parse_args()

    prepare_run_seed(args)

    device = "cuda"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    if not os.path.isfile(args.weights):
        raise FileNotFoundError(f"weights: {args.weights}")

    ensure_results_dir()

    dense_layout = None
    dense_use_rot = False
    if args.run_dense:
        dense_root = _resolve_dense_data_root(args)
        dense_use_rot = _dense_use_rot_variant(args, dense_root)
        fallbacks = [args.dense_asset_fallback, _DENSE_ASSET_FALLBACK]
        dense_layout = resolve_megadepth_dense_layout(
            dense_root,
            asset_fallback_roots=fallbacks,
            auto_symlink=not bool(args.dense_no_auto_symlink),
        )

    run_mega1500 = (not args.latency_only) and (
        not args.run_dense or bool(args.with_mega1500)
    )
    run_plain = run_mega1500 and args.mode == "plain"
    run_rot = run_mega1500 and args.mode == "rot"
    

    print(f"[Weights] {args.weights}", flush=True)
    print(
        f"[Eval] MegaDepth plain_root={args.mega_data_root_plain} (C4 off) | rot_root={args.mega_data_root_rot} (C4 on) | "
        f"symmetric={args.symmetric} upsample={args.upsample} seed={args.seed} | "
        f"mode={args.mode}",
        flush=True,
    )
    if args.run_dense and dense_layout is not None:
        print(
            f"  dense: {dense_layout.data_root} ({'rot' if dense_use_rot else 'plain'}, "
            f"scene_info={dense_layout.scene_info_root}, "
            f"samples={args.dense_num_samples}, batch={args.dense_batch_size}, "
            f"C4={'on' if dense_use_rot else 'off'})",
            flush=True,
        )
    if args.run_dense and not run_mega1500:
        print("  (Skipped Mega1500; add --with-mega1500 to run both)", flush=True)

    summary: List[dict] = []
    latency_rows: List[dict] = []
    dense_runs: List[dict] = []

    if run_mega1500:
        for res in args.eval_resolutions:
            res = int(res)
            if run_plain:
                print("")
                print(
                    f"[Eval] MegaDepth | coarse_res={res} | "
                    f"symmetric={args.symmetric} | upsample={args.upsample} | split=plain | "
                    f"c4_mode=disabled"
                )
                print("-" * 40)
                summary.append(
                    _evaluate_plain(
                        args=args,
                        device=device,
                        coarse_res=res,
                        mega_data_root=args.mega_data_root_plain,
                    )
                )
            if run_rot:
                rot_c4_modes = ["fundamental_hybrid_gp_intrinsic"]
                rot_runs = []
                for c4_mode in rot_c4_modes:
                    print("")
                    print(
                        f"[Eval] MegaDepth | coarse_res={res} | "
                        f"symmetric={args.symmetric} | upsample={args.upsample} | split=rot | "
                        f"c4_mode={c4_mode}"
                    )
                    print("-" * 40)
                    rot_runs.append(
                        _evaluate_rot_c4_mode(
                            args=args,
                            device=device,
                            coarse_res=res,
                            mega_data_root=args.mega_data_root_rot,
                            c4_mode=c4_mode,
                        )
                    )
                    summary.append(rot_runs[-1])

    if args.run_dense and not args.latency_only:
        for res in args.eval_resolutions:
            res = int(res)
            print(f"\n[Eval] resolution={res} | Megadepth dense", flush=True)
            runs_this_res: List[dict] = []

            if bool(getattr(args, "dense_with_plain_baseline", False)) and dense_use_rot:
                plain_root = os.path.abspath(os.path.expanduser(str(args.mega_data_root_plain)))
                plain_layout = resolve_megadepth_dense_layout(
                    plain_root,
                    asset_fallback_roots=fallbacks,
                    auto_symlink=not bool(args.dense_no_auto_symlink),
                )
                print(f"[Dense] plain baseline (same resize as rot)", flush=True)
                row_plain = _evaluate_dense(
                    args=args,
                    device=device,
                    coarse_res=res,
                    data_root=plain_layout.data_root,
                    layout=plain_layout,
                    use_rot=False,
                    tag_suffix="_baseline",
                )
                runs_this_res.append(row_plain)
                m0 = row_plain["results"]
                print(
                    f"[Dense plain baseline | r={res}] epe={m0.get('epe')} "
                    f"pck5={m0.get('mega_pck_5')}",
                    flush=True,
                )

            row = _evaluate_dense(
                args=args,
                device=device,
                coarse_res=res,
                data_root=dense_layout.data_root,
                layout=dense_layout,
                use_rot=dense_use_rot,
                tag_suffix="_equiv" if bool(getattr(args, "dense_rot_equivariant", False)) else "",
            )
            runs_this_res.append(row)
            dense_runs.extend(runs_this_res)
            m = row["results"]
            print(
                f"[Dense | r={res}] epe={m.get('epe')} "
                f"pck1={m.get('mega_pck_1')} pck3={m.get('mega_pck_3')} pck5={m.get('mega_pck_5')}",
                flush=True,
            )
            if len(runs_this_res) == 2:
                epe_p = runs_this_res[0]["results"].get("epe")
                epe_r = runs_this_res[1]["results"].get("epe")
                if epe_p is not None and epe_r is not None:
                    print(
                        f"[Dense equivariance check] delta_epe_rot_minus_plain={float(epe_r) - float(epe_p):.4f}",
                        flush=True,
                    )

    if args.run_latency or args.latency_only:
        if not run_rot and args.mode == "plain":
            print("[Latency] rot branch needed, or use --mode rot", flush=True)
        else:
            for res in args.eval_resolutions:
                latency_rows.extend(
                    _run_latency_compare(args, device, int(res), rot_c4_modes=rot_c4_modes)
                )

    payload = {
        "seed_info": get_rng_seed_info(args),
        "weights_path": args.weights,
                "mode": args.mode,
        "rot_c4_modes": rot_c4_modes if run_rot else [],
        "runs": summary,
        "dense": dense_runs,
        "latency": latency_rows,
    }
    print("\n[Summary]", flush=True)
    print(payload, flush=True)

    if latency_rows:
        print(f"[Latency] {latency_rows}", flush=True)
