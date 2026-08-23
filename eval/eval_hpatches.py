"""HPatches homography evaluation (plain / rot)."""

from __future__ import annotations

import sys
import os

# 🔧 Force use of the local redimatch package (relative to the repo root of this script)
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_local_redimatch_path = _repo_root

sys.path.insert(0, _local_redimatch_path)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

modules_to_delete = [k for k in list(sys.modules.keys()) if "redimatch" in k.lower()]
for k in modules_to_delete:
    del sys.modules[k]

import argparse
import json
import torch
import cv2
import numpy as np
from argparse import ArgumentParser
from PIL import Image
from tqdm import tqdm

from _common import prepare_run_seed
from _common import install_gp_intrinsic_rotation_router
from _common import (
    hpatches_H_path,
    hpatches_im_a_path,
    hpatches_im_b_path,
    is_fixed_a_rot_release,
)

from redimatch.utils import pose_auc

import redimatch

from _common import (
    _build_eval_model,
)

WEIGHTS = os.path.join(_repo_root, "models", "outdoor.pth")

DEFAULT_SEQS_SUBDIRS_PLAIN = [
    "hpatches-sequences-release",
]
DEFAULT_SEQS_SUBDIRS_ROT = [
    "hpatches-sequences-release_rot",
]

def ensure_results_dir():
    os.makedirs("results", exist_ok=True)

def _benchmark_one_hpatches_subdir(
    model,
    hpatches_root: str,
    seqs_dir: str,
):
    seqs_path = os.path.join(hpatches_root, seqs_dir)
    if not os.path.isdir(seqs_path):
        raise FileNotFoundError(f"HPatches subdirectory not found: {seqs_path}")

    fixed_a = is_fixed_a_rot_release(seqs_path)

    seq_names = sorted(os.listdir(seqs_path))
    homog_dists = []

    for seq_name in tqdm(seq_names, total=len(seq_names), leave=False, desc=seqs_dir):
        if not os.path.isdir(os.path.join(seqs_path, seq_name)):
            continue
        seq_dir = os.path.join(seqs_path, seq_name)
        im_A_path = hpatches_im_a_path(seq_dir)
        im_A = Image.open(im_A_path)
        w1, h1 = im_A.size
        for im_idx in range(2, 7):
            im_B_path = hpatches_im_b_path(seq_dir, im_idx, fixed_a_only_rotate_b=fixed_a)
            if not os.path.isfile(im_B_path):
                raise FileNotFoundError(im_B_path)
            im_B = Image.open(im_B_path)
            w2, h2 = im_B.size
            H = np.loadtxt(hpatches_H_path(seq_dir, im_idx))
            dense_matches, dense_certainty = model.match(im_A_path, im_B_path)

            good_matches, _ = model.sample(dense_matches, dense_certainty, 5000)
            if torch.is_tensor(good_matches):
                good_matches = good_matches.detach().cpu().numpy()

            offset = 0.5
            pos_a = np.stack(
                (
                    w1 * (good_matches[:, 0] + 1) / 2,
                    h1 * (good_matches[:, 1] + 1) / 2,
                ),
                axis=-1,
            ) - offset
            pos_b = np.stack(
                (
                    w2 * (good_matches[:, 2] + 1) / 2,
                    h2 * (good_matches[:, 3] + 1) / 2,
                ),
                axis=-1,
            ) - offset

            try:
                H_pred, _ = cv2.findHomography(
                    pos_a,
                    pos_b,
                    method=cv2.RANSAC,
                    confidence=0.99999,
                    ransacReprojThreshold=3 * min(w2, h2) / 480,
                )
            except Exception:
                H_pred = None
            if H_pred is None:
                H_pred = np.zeros((3, 3))
                H_pred[2, 2] = 1.0
            corners = np.array([[0, 0, 1], [0, h1 - 1, 1], [w1 - 1, 0, 1], [w1 - 1, h1 - 1, 1]])
            real_warped_corners = np.dot(corners, np.transpose(H))
            real_warped_corners = real_warped_corners[:, :2] / real_warped_corners[:, 2:]
            warped_corners = np.dot(corners, np.transpose(H_pred))
            warped_corners = warped_corners[:, :2] / warped_corners[:, 2:]
            mean_dist = np.mean(np.linalg.norm(real_warped_corners - warped_corners, axis=1)) / (min(w2, h2) / 480.0)
            homog_dists.append(mean_dist)

    thresholds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    auc = pose_auc(np.array(homog_dists), thresholds)
    out = {
        "hpatches_homog_auc_3": auc[2],
        "hpatches_homog_auc_5": auc[4],
        "hpatches_homog_auc_10": auc[9],
        "n_pairs": len(homog_dists),
        "homog_dists": homog_dists,
    }
    return out

def test_hpatches(
    model,
    name,
    hpatches_root: str,
    seqs_subdirs,
):
    subdir_results = {}
    all_dists = []
    for seqs_dir in seqs_subdirs:
        res = _benchmark_one_hpatches_subdir(
            model, hpatches_root, seqs_dir
        )
        all_dists.extend(res.pop("homog_dists", []))
        subdir_results[seqs_dir] = res

    if all_dists:
        auc = pose_auc(np.array(all_dists), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        auc3, auc5, auc10 = auc[2], auc[4], auc[9]
    else:
        auc3 = auc5 = auc10 = 0.0

    hpatches_results = {
        "hpatches_homog_auc_3": float(auc3),
        "hpatches_homog_auc_5": float(auc5),
        "hpatches_homog_auc_10": float(auc10),
    }

    print(f"[hpatches] {name}", flush=True)
    print(hpatches_results, flush=True)
    return hpatches_results

def _set_c4(model, enabled: bool, *, perm_reverse: bool = True):
    dec = getattr(model, "decoder", None)
    if dec is None:
        return
    dec.c4_rotation_matching = bool(enabled)
    if enabled:
        dec.c4_perm_reverse = bool(perm_reverse)
        # fundamental_hybrid 判向：必须用 OT(fundamental) 判向，不用单独余弦相似度
        dec.c4_hybrid = True
        dec.c4_hybrid_alpha = 0.35
        dec.c4_hybrid_temp = 0.05
        dec.c4_hybrid_margin_keep = 0.03
        dec.c4_hybrid_disagree_margin = 0.03
        dec.c4_hybrid_ot_pool_size = 12
        dec.c4_hybrid_ot_epsilon = 0.07
        dec.c4_hybrid_ot_iters = 10

def _maybe_enable_router(model, enabled: bool):
    if not enabled:
        return False
    ok, _ = install_gp_intrinsic_rotation_router(model)
    print(f"[C4] gp_intrinsic rotation router {'enabled' if ok else 'not available, fallback to default'}")
    return bool(ok)

if __name__ == "__main__":
    parser = ArgumentParser(description="HPatches multi-resolution evaluation (plain+rot)")
    parser.add_argument("--weights", default=WEIGHTS, type=str)
    parser.add_argument("--eval_resolutions", nargs="+", type=int, default=[576],
                        help="Evaluation resolution list, default 576")
    parser.add_argument("--no_custom_corr", action="store_true")
    parser.add_argument("--symmetric", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mode", type=str, default="plain", choices=["plain", "rot"],
                        help="plain (default) or rot")
    parser.add_argument(
        "--hpatches_root",
        type=str,
        default=os.path.join(_repo_root, "data", "hpatches"),
        help="HPatches dataset root (downloaded separately)",
    )
    parser.add_argument("--cycle_cert_filter", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable decoder cycle consistency to filter certainty of symmetric halves")
    parser.add_argument("--upsample", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--c4_perm_reverse", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", nargs="?", const=None, default=None, type=int)
    args = parser.parse_args()
    prepare_run_seed(args)
    device = "cuda"
    if args.weights:
        try:
            weights = torch.load(args.weights, map_location=device, weights_only=False)
        except TypeError:
            weights = torch.load(args.weights, map_location=device)
        print(f"[Weights] Loading weights from {args.weights}")
    else:
        weights = None

    ensure_results_dir()
    args.hpatches_root = os.path.abspath(os.path.expanduser(args.hpatches_root))

    seqs_plain = [x.strip() for x in str(getattr(args, "seqs_subdirs_plain", ",".join(DEFAULT_SEQS_SUBDIRS_PLAIN))).split(",") if x.strip()]
    seqs_rot = [x.strip() for x in str(getattr(args, "seqs_subdirs_rot", ",".join(DEFAULT_SEQS_SUBDIRS_ROT))).split(",") if x.strip()]

    summary = []
    for res in args.eval_resolutions:
        res = int(res)
        print("")
        print(f"[HPatches] coarse_res = h_resized = w_resized = {res} | seed={args.seed}")
        print("-" * 40)

        args.coarse_res = res

        # ====== plain (no C4) ======
        if args.mode == "plain":
            print(f"[HPatches/plain] C4 disabled | seqs_subdirs = {seqs_plain}", flush=True)
            model_plain = _build_eval_model(
                args,
                device,
                weights,
                symmetric=bool(args.symmetric),
                upsample_preds=bool(args.upsample),
            )
            if args.weights and os.path.isfile(args.weights):
                # weights already merged
                tag0 = "hpatches"
            else:
                tag0 = "hpatches"
            _set_c4(model_plain, False, perm_reverse=bool(True))
            _maybe_enable_router(model_plain, False)
            if bool(args.cycle_cert_filter) and hasattr(model_plain, "decoder"):
                model_plain.decoder.cycle_cert_filter = True
                print("[cycle_cert] enabled on plain model", flush=True)
            model_plain.h_resized = res
            model_plain.w_resized = res
            model_plain.train(False)
            metrics_plain = test_hpatches(
                model_plain,
                f"{tag0}_plain_r{res}",
                hpatches_root=args.hpatches_root,
                seqs_subdirs=seqs_plain,
            )
            summary.append({"resolution": res, "mode": "plain", "name": f"{tag0}_plain_r{res}", "metrics": metrics_plain})
            del model_plain
            torch.cuda.empty_cache()

        # ====== rot (C4 on) ======
        if args.mode == "rot":
            print(f"[HPatches/rot] C4 enabled | seqs_subdirs = {seqs_rot}", flush=True)
            model_rot = _build_eval_model(
                args,
                device,
                weights,
                symmetric=bool(args.symmetric),
                upsample_preds=bool(args.upsample),
            )
            if args.weights and os.path.isfile(args.weights):
                # weights already merged
                tag1 = "hpatches"
            else:
                tag1 = "hpatches"
            _set_c4(
                model_rot,
                True,
                perm_reverse=bool(True),
            )
            _maybe_enable_router(model_rot, True)
            if bool(args.cycle_cert_filter) and hasattr(model_rot, "decoder"):
                model_rot.decoder.cycle_cert_filter = True
                print("[cycle_cert] enabled on rot model", flush=True)
            model_rot.h_resized = res
            model_rot.w_resized = res
            model_rot.train(False)
            metrics_rot = test_hpatches(
                model_rot,
                f"{tag1}_rot_r{res}",
                hpatches_root=args.hpatches_root,
                seqs_subdirs=seqs_rot,
            )
            summary.append({"resolution": res, "mode": "rot", "name": f"{tag1}_rot_r{res}", "metrics": metrics_rot})
            del model_rot
            torch.cuda.empty_cache()

    print("\n[Summary]")
    for s in summary:
        print(s, flush=True)
