"""SatAst evaluation."""

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
import shutil
import tempfile
import torch
import cv2
import matplotlib
import numpy as np
from argparse import ArgumentParser
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

from _common import prepare_run_seed

from redimatch.benchmarks.satast import SatAst
from redimatch.utils import pose_auc

import redimatch

# keep C4+gp_intrinsic routing consistent with other eval scripts
from _common import install_gp_intrinsic_rotation_router

from _common import (
    _build_eval_model,
)

matplotlib.use("Agg")

WEIGHTS = os.path.join(_repo_root, "models", "outdoor.pth")
ALL_C4_MODES = ["gp"]  # List of all modes

# SATAST is downloaded separately because of its size and distribution terms.
DEFAULT_SATAST_ANNOTATIONS = os.path.join(_repo_root, "data", "satast", "satast_annotations_with_rot")
DEFAULT_SATAST_IMAGE_ROOT = os.path.join(_repo_root, "data", "satast")
OFFSET = 0.5

# SATAST matches.jpg — align with vggt/docs/satast_match_visualization_spec.md (and REDI_Match/benchmarks/satast.py)
MATCH_LINE_COLOR = (1.0, 0.55, 0.0, 1.0)
MATCH_POINT_COLOR = MATCH_LINE_COLOR
MATCH_LINE_WIDTH = 1.0
MATCH_LINE_ALPHA = 0.30
MATCH_LINE_MARKERSIZE = 0.01
MATCH_SCATTER_SIZE = 4
MATCH_TEXT_FONTSIZE = 50
MATCH_TEXT_COLOR = "white"
MATCH_TEXT_POS_AXES = (0.01, 0.98)
MATCH_TEXT_VA = "top"
MATCH_TEXT_HA = "left"
MATCH_FIG_DPI = 250
MATCH_SUBPLOTS_ADJUST = dict(left=0, right=1, bottom=0, top=0.96)
MATCH_SAVE_KWARGS = dict(bbox_inches="tight", pad_inches=0)

def ensure_results_dir():
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/satast", exist_ok=True)

def test_satast(
    model,
    name: str,
    json_folder: str,
    image_root: str,
    use_warp: bool = False,
    *,
    max_save_pairs: int | None = None,
    save_vis: bool = True,
    c4_mode: str = "gp",
):
    if use_warp:
        bench = SatAst(json_folder_path=json_folder, image_dataset_root=image_root)
        results = bench.benchmark_warp(model, model_name=name)
        print(f"[satast] {name}")
        print("[satast] warp mode temporarily does not export top inlier visualizations")
        print(results)
        return results

    results, vis_summary = benchmark_satast_with_top_inlier_visuals(
        model=model,
        model_name=name,
        json_folder=json_folder,
        image_root=image_root,
        out_root=os.path.join("results", "satast", name),
        max_save_pairs=max_save_pairs,
        save_vis=save_vis,
        c4_mode=c4_mode,
    )
    print(f"[satast] {name}")
    print(results)
    return results

def _pixel_to_normalized(pts_pix, w, h, offset=OFFSET):
    pts_norm = np.zeros_like(pts_pix, dtype=np.float64)
    pts_norm[..., 0] = (2.0 * (pts_pix[..., 0] + offset) / w) - 1.0
    pts_norm[..., 1] = (2.0 * (pts_pix[..., 1] + offset) / h) - 1.0
    return pts_norm

def _normalized_to_pixel(pts_norm, w, h, offset=OFFSET):
    pts_pix = np.zeros_like(pts_norm, dtype=np.float64)
    pts_pix[..., 0] = (w * (pts_norm[..., 0] + 1.0) / 2.0) - offset
    pts_pix[..., 1] = (h * (pts_norm[..., 1] + 1.0) / 2.0) - offset
    return pts_pix

def _draw_match_visualization(
    im_a_path,
    im_b_path,
    pts_a,
    pts_b,
    out_path,
    sampled_count: int,
    inlier_count: int,
    *,
    title: str | None = None,
):
    img_a_bgr = cv2.imread(im_a_path)
    img_b_bgr = cv2.imread(im_b_path)
    if img_a_bgr is None or img_b_bgr is None:
        raise FileNotFoundError(f"Failed to read images for visualization: {im_a_path}, {im_b_path}")

    img_a = cv2.cvtColor(img_a_bgr, cv2.COLOR_BGR2RGB)
    img_b = cv2.cvtColor(img_b_bgr, cv2.COLOR_BGR2RGB)
    pts_a = np.asarray(pts_a, dtype=np.float64)
    pts_b = np.asarray(pts_b, dtype=np.float64)

    h_a, w_a = img_a.shape[:2]
    h_b, w_b = img_b.shape[:2]
    h_max = max(h_a, h_b)
    total_w = w_a + w_b
    fig_h = min(22.0, max(3.5, h_max / 120.0))
    fig_w = fig_h * (total_w / h_max)
    if fig_w > 22.0:
        scale = 22.0 / fig_w
        fig_w *= scale
        fig_h *= scale

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=MATCH_FIG_DPI)
    gs = fig.add_gridspec(1, 2, width_ratios=[w_a, w_b], wspace=0)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    for ax, image in ((ax_a, img_a), (ax_b, img_b)):
        ax.imshow(image)
        ax.set_axis_off()
    fig.subplots_adjust(**MATCH_SUBPLOTS_ADJUST)

    if len(pts_a) and len(pts_b):
        fig.canvas.draw()
        trans_fig = fig.transFigure.inverted()
        fig_pts_a = trans_fig.transform(ax_a.transData.transform(pts_a))
        fig_pts_b = trans_fig.transform(ax_b.transData.transform(pts_b))
        # Lines (low zorder, lines below text)
        fig.lines = [
            matplotlib.lines.Line2D(
                (fig_pts_a[i, 0], fig_pts_b[i, 0]),
                (fig_pts_a[i, 1], fig_pts_b[i, 1]),
                transform=fig.transFigure,
                c=MATCH_LINE_COLOR,
                linewidth=MATCH_LINE_WIDTH,
                markersize=MATCH_LINE_MARKERSIZE,
                alpha=MATCH_LINE_ALPHA,
                zorder=1,
            )
            for i in range(len(pts_a))
        ]
        # scatter points (zorder slightly above lines)
        ax_a.scatter(
            pts_a[:, 0],
            pts_a[:, 1],
            c=[MATCH_POINT_COLOR],
            s=MATCH_SCATTER_SIZE,
            zorder=2,
        )
        ax_b.scatter(
            pts_b[:, 0],
            pts_b[:, 1],
            c=[MATCH_POINT_COLOR],
            s=MATCH_SCATTER_SIZE,
            zorder=2,
        )

    head = title or "REDI-match"
    # ── Text: topmost layer, high zorder + black semi-transparent background to avoid being covered by lines ──
    ax_a.text(
        MATCH_TEXT_POS_AXES[0],
        MATCH_TEXT_POS_AXES[1],
        f"{head}\nsampled={sampled_count}\ninliers={inlier_count}",
        transform=ax_a.transAxes,
        fontsize=MATCH_TEXT_FONTSIZE,
        va=MATCH_TEXT_VA,
        ha=MATCH_TEXT_HA,
        color=MATCH_TEXT_COLOR,
        zorder=999,
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="black",
            edgecolor="none",
            alpha=0.45,
        ),
    )
    fig.savefig(out_path, **MATCH_SAVE_KWARGS)
    plt.close(fig)

def _satast_match_with_c4_mode(model, im_a_path, im_b_path, c4_mode):
    if c4_mode == "disabled":
        # C4 fully off, direct matching
        setattr(model, "_satast_pred_rotation", None)
        dm, dc = model.match(im_a_path, im_b_path)
        return dm, dc
    if c4_mode == "gp":
        dm, dc = model.match(im_a_path, im_b_path)
        # try to read last c4 rotation info from decoder
        pred_k = None
        dec = getattr(model, "decoder", None)
        if dec is not None:
            rot = getattr(dec, "_last_c4_rotation", None)
            if isinstance(rot, dict) and rot.get("rot_idx") is not None:
                try:
                    pred_k = int(torch.as_tensor(rot.get("rot_idx")).reshape(-1)[0].item()) % 4
                except Exception:
                    pred_k = None
            else:
                try:
                    pred_k = int(rot) % 4
                except Exception:
                    pred_k = None
        setattr(model, "_satast_pred_rotation", pred_k)
        return dm, dc
    raise ValueError(f"Unknown SATAST C4 mode: {c4_mode}")

def benchmark_satast_with_top_inlier_visuals(
    model,
    model_name: str,
    json_folder: str,
    image_root: str,
    out_root: str,
    max_save_pairs: int | None = None,
    save_vis: bool = True,
    c4_mode: str = "gp",
):
    json_files = sorted([f for f in os.listdir(json_folder) if f.endswith(".json")])
    all_reprojection_errors = []
    temp_file = tempfile.NamedTemporaryFile(suffix=".jpeg", delete=False)
    rotated_b_path = temp_file.name
    pair_records = []

    for json_name in tqdm(json_files, desc="Running SATAST Benchmark"):
        json_path = os.path.join(json_folder, json_name)
        with open(json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        im_a_rel = json_data["query_path"].replace("\\", "/")
        im_b_rel = json_data["pred_path"].replace("\\", "/")
        im_a_path = os.path.join(image_root, im_a_rel)
        im_b_path0 = os.path.join(image_root, im_b_rel)

        im_a = Image.open(im_a_path)
        w1, h1 = im_a.size
        im_b0 = Image.open(im_b_path0)
        w2, h2 = im_b0.size
        im_b0.save(rotated_b_path, format="JPEG")

        dense_preds = _satast_match_with_c4_mode(
            model,
            im_a_path,
            rotated_b_path,
            c4_mode,
        )
        if isinstance(dense_preds, dict):
            sparse_preds = model.sample(dense_preds, 10_000)
        else:
            sparse_preds = model.sample(*dense_preds, 10_000)
        good_matches = sparse_preds[0]
        if torch.is_tensor(good_matches):
            good_matches_np = good_matches.detach().cpu().numpy()
        else:
            good_matches_np = np.asarray(good_matches)
        pos_a_norm = good_matches_np[:, :2]
        pos_b_norm = good_matches_np[:, 2:]

        # Final match uses original image B + C4 alignment; decoder already converted flow back to B's original space via c4_rotate_flow_back,
        # None needs further coordinate unrotation

        H_pred_norm = None
        inliers = None
        if len(pos_a_norm) >= 4:
            try:
                H_pred_norm, inliers = cv2.findHomography(
                    pos_a_norm,
                    pos_b_norm,
                    method=cv2.USAC_DEFAULT,
                    confidence=0.99999999,
                    maxIters=100_000,
                    ransacReprojThreshold=0.001,
                )
            except cv2.error:
                H_pred_norm = None
                inliers = None

        last_iteration_corrs = json_data["correspondences"][-1]
        pts_src_pix = np.array(last_iteration_corrs["pts_src"])
        pts_dst_pix = np.array(last_iteration_corrs["pts_dst"])
        if pts_src_pix.shape[0] == 0:
            continue

        if H_pred_norm is None:
            all_reprojection_errors.extend([float("inf")] * pts_src_pix.shape[0])
            pair_records.append(
                {
                    "json_name": json_name,
                    "im_a_path": im_a_path,
                    "im_b_path": im_b_path0,
                    "num_inliers": 0,
                    "num_matches": int(len(pos_a_norm)),
                    "mean_reproj_error": None,
                    "inlier_pts_a": np.zeros((0, 2), dtype=np.float32),
                    "inlier_pts_b": np.zeros((0, 2), dtype=np.float32),
                }
            )
            continue

        pts_src_norm = _pixel_to_normalized(pts_src_pix, w1, h1)
        pts_src_norm_h = np.hstack((pts_src_norm, np.ones((pts_src_norm.shape[0], 1))))
        warped_pts_norm_h = np.dot(pts_src_norm_h, H_pred_norm.T)
        warped_pts_norm = warped_pts_norm_h[:, :2] / (warped_pts_norm_h[:, 2, np.newaxis] + 1e-8)
        warped_pts_pix = _normalized_to_pixel(warped_pts_norm, w2, h2)
        errors = np.linalg.norm(warped_pts_pix - pts_dst_pix, axis=1)
        all_reprojection_errors.extend(errors.tolist())

        inliers_mask = inliers.flatten().astype(bool) if inliers is not None else np.zeros((len(pos_a_norm),), dtype=bool)
        inlier_pts_a = _normalized_to_pixel(pos_a_norm[inliers_mask], w1, h1) if inliers_mask.any() else np.zeros((0, 2), dtype=np.float32)
        inlier_pts_b = _normalized_to_pixel(pos_b_norm[inliers_mask], w2, h2) if inliers_mask.any() else np.zeros((0, 2), dtype=np.float32)
        pair_records.append(
            {
                "json_name": json_name,
                "im_a_path": im_a_path,
                "im_b_path": im_b_path0,
                "num_inliers": int(inliers_mask.sum()),
                "num_matches": int(len(pos_a_norm)),
                "mean_reproj_error": float(np.mean(errors)) if len(errors) else None,
                "inlier_pts_a": inlier_pts_a,
                "inlier_pts_b": inlier_pts_b,
            }
        )

    temp_file.close()
    try:
        os.remove(rotated_b_path)
    except OSError:
        pass

    thresholds = np.arange(1, 31)
    auc = pose_auc(np.array(all_reprojection_errors), thresholds)
    results = {
        "reprojection_auc_5px": auc[4],
        "reprojection_auc_10px": auc[9],
        "reprojection_auc_20px": auc[19],
        "reprojection_auc_30px": auc[29],
    }

    os.makedirs(out_root, exist_ok=True)
    if save_vis:
        sorted_pairs = sorted(pair_records, key=lambda x: (x["num_inliers"], x["num_matches"]), reverse=True)
        if max_save_pairs is not None:
            sorted_pairs = sorted_pairs[: int(max_save_pairs)]
    pair_summary = []
    if save_vis:
        for rank, rec in enumerate(sorted_pairs, start=1):
            pair_dir = os.path.join(out_root, f"rank{rank:04d}_{os.path.splitext(rec['json_name'])[0]}")
            os.makedirs(pair_dir, exist_ok=True)
            shutil.copy2(rec["im_a_path"], os.path.join(pair_dir, "image_A.jpg"))
            shutil.copy2(rec["im_b_path"], os.path.join(pair_dir, "image_B.jpg"))
            _draw_match_visualization(
                rec["im_a_path"],
                rec["im_b_path"],
                rec["inlier_pts_a"],
                rec["inlier_pts_b"],
                os.path.join(pair_dir, "matches.jpg"),
                sampled_count=rec["num_matches"],
                inlier_count=rec["num_inliers"],
            )
            pair_summary.append(
                {
                    "rank": rank,
                    "json_name": rec["json_name"],
                    "pair_dir": pair_dir,
                    "num_inliers": rec["num_inliers"],
                    "num_matches": rec["num_matches"],
                    "mean_reproj_error": rec["mean_reproj_error"],
                }
            )

    return results, {"model_name": model_name, "all_pairs": pair_summary}

def _maybe_enable_gp_intrinsic_router(model) -> bool:
    dec = getattr(model, "decoder", None)
    if dec is None or not getattr(dec, "c4_rotation_matching", False):
        return False
    ok, _ = install_gp_intrinsic_rotation_router(model)
    print(f"[C4] gp_intrinsic rotation router {'enabled' if ok else 'not available, fallback to default'}")
    return bool(ok)

if __name__ == "__main__":
    parser = ArgumentParser(description="SATAST multi-resolution evaluation")
    parser.add_argument("--weights", default=WEIGHTS, type=str)
    parser.add_argument("--eval_resolutions", nargs="+", type=int, default=[576],
                        help="Evaluation resolution list, default 576")
    parser.add_argument("--save_vis", action=argparse.BooleanOptionalAction, default=False, help="whether to save visualization match plots")
    parser.add_argument("--no_custom_corr", action="store_true")
    parser.add_argument("--symmetric", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--upsample", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--satast_annotations",
        type=str,
        default=DEFAULT_SATAST_ANNOTATIONS,
        help="SATAST annotation directory (downloaded separately)",
    )
    parser.add_argument(
        "--satast_image_root",
        type=str,
        default=DEFAULT_SATAST_IMAGE_ROOT,
        help="SATAST image root (downloaded separately)",
    )
    parser.add_argument(
        "--c4_rotation_matching",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to enable decoder C4 rotation matching (default on, aligned with eval_roma_outdoor_gp0215_satast_multires)",
    )
    parser.add_argument(
        "--no-c4",
        action="store_true",
        help="Completely disable C4 rotation detection (equivalent to --no-c4_rotation_matching, and no entropy router installed)",
    )
    parser.add_argument(
        "--c4_perm_reverse",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="C4 group channel permutation direction (default True, consistent with common configs)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: random, not fixed; can override with --seed)",
    )
    args = parser.parse_args()
    args.c4_inference_mode = "gp"
    args.satast_annotations = os.path.abspath(os.path.expanduser(args.satast_annotations))
    args.satast_image_root = os.path.abspath(os.path.expanduser(args.satast_image_root))
    args.satast_use_warp = getattr(args, "satast_use_warp", False)
    args.c4_rotation_matching = getattr(args, "c4_rotation_matching", True)
    args.max_save_pairs = getattr(args, "max_save_pairs", 20)
    if args.no_c4:
        args.c4_rotation_matching = False
        args.c4_inference_mode = "disabled"
        print("[C4] --no-c4: C4 rotation detection fully disabled", flush=True)

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

    summary = []
    for res in args.eval_resolutions:
        print("")
        print(
            f"[SATAST] coarse_res={res} | "
            f"symmetric={args.symmetric} upsample={args.upsample} seed={args.seed} "
            f"c4_rotation_matching={bool(args.c4_rotation_matching)} c4_inference_mode={args.c4_inference_mode} "
            f"c4_perm_reverse={bool(True)}"
        )
        print("-" * 40)
        args.coarse_res = int(res)
        model = _build_eval_model(
            args,
            device,
            weights,
            symmetric=bool(args.symmetric),
            upsample_preds=bool(args.upsample),
        )
        dec = getattr(model, "decoder", None)
        if dec is not None:
            use_c4 = bool(args.c4_rotation_matching) or args.c4_inference_mode in ALL_C4_MODES or args.c4_inference_mode == "all"
            dec.c4_rotation_matching = bool(use_c4)
            if use_c4:
                dec.c4_perm_reverse = bool(True)
                dec.c4_symmetric_reverse = "align_to_query"
                # Use OT (fundamental) for direction selection; do not use cosine similarity alone.
                dec.c4_hybrid = True
                dec.c4_hybrid_alpha = 0.35
                dec.c4_hybrid_temp = 0.05
                dec.c4_hybrid_margin_keep = 0.03
                dec.c4_hybrid_disagree_margin = 0.03
                dec.c4_hybrid_ot_pool_size = 12
                dec.c4_hybrid_ot_epsilon = 0.07
                dec.c4_hybrid_ot_iters = 10
        if args.c4_inference_mode == "gp" or args.c4_inference_mode == "all":
            _maybe_enable_gp_intrinsic_router(model)
        if args.weights and os.path.isfile(args.weights):
            # weights already merged
            tag = "satast"
        else:
            tag = "satast"
        model.h_resized = int(res)
        model.w_resized = int(res)
        model.train(False)

        # Determine list of modes to run
        if args.c4_inference_mode == "all":
            modes_to_run = list(ALL_C4_MODES)
        else:
            modes_to_run = [args.c4_inference_mode]

        for c4m in modes_to_run:
            print(f"\n[SATAST] mode={c4m}", flush=True)
            metrics = test_satast(
                model,
                f"{tag}_r{res}_{c4m}",
                json_folder=args.satast_annotations,
                image_root=args.satast_image_root,
                use_warp=args.satast_use_warp,
                max_save_pairs=args.max_save_pairs if c4m == modes_to_run[0] else None,
                save_vis=args.save_vis if c4m == modes_to_run[0] else False,
                c4_mode=c4m,
            )
            summary.append({"resolution": int(res), "mode": c4m, "name": f"{tag}_r{res}_{c4m}", "metrics": metrics})

        del model
        torch.cuda.empty_cache()

    print("\n[Summary]")
    for s in summary:
        print(s)
