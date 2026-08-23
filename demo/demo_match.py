#!/usr/bin/env python3
"""
Inference-only demo for the public REDI-Match release.

Usage:
  cd REDI-Match
  python demo/demo_match.py
  python demo/demo_match.py --im_A assets/toronto_A.jpg --im_B assets/toronto_B_rot180.jpg
"""

from __future__ import annotations

import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _REPO_ROOT)

import cv2
import numpy as np
import torch
from PIL import Image

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm, gridspec

from eval._common import install_gp_intrinsic_rotation_router


# ═══════════════════════════════════════════════════════════════════
# Path configuration (all relative to REPO_ROOT)
# ═══════════════════════════════════════════════════════════════════

WEIGHTS = os.path.join(_REPO_ROOT, "models", "outdoor.pth")
IM_A = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "remote_satast_A.jpg")
IM_B = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "remote_satast_B.jpg")
SAVE_SYM = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "demo_match_symmetric.jpg")
SAVE_WARP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "demo_match_warp.jpg")


# ═══════════════════════════════════════════════════════════════════
# Utility functions (inlined, no external dependencies)
# ═══════════════════════════════════════════════════════════════════

def _extract_state_dict(ckpt: dict):
    for key in ("model", "state_dict", "model_state_dict"):
        if key in ckpt and isinstance(ckpt[key], dict):
            ckpt = ckpt[key]
            break
    return {k[7:] if k.startswith("module.") else k: v for k, v in ckpt.items()}


def _resolve(p: str) -> str:
    return os.path.join(_REPO_ROOT, p) if not os.path.isabs(p) else p


def _ransac_inliers_and_mask(kpts_src, kpts_dst):
    """Return (inlier_count, mask), where mask is a bool array, True=inlier."""
    n = len(kpts_src)
    if n < 8:
        return 0, np.zeros(n, dtype=bool)
    try:
        _, mask = cv2.findFundamentalMat(
            kpts_src, kpts_dst,
            method=getattr(cv2, "USAC_MAGSAC", cv2.FM_RANSAC),
            confidence=0.999999, maxIters=10000, ransacReprojThreshold=0.5,
        )
        if mask is not None:
            m = mask.ravel().astype(bool)
            return int(m.sum()), m
    except Exception:
        pass
    return 0, np.zeros(n, dtype=bool)


def _draw_bidirectional(
    img_src, img_dst,
    kpts_ab, kpts_ba,
    inliers_ab, inliers_ba,
    mask_ab, mask_ba,
    sampled_ab, sampled_ba,
    out_path,
):
    """Draw bidirectional point-line matching visualization, inlier=orange, outlier=red."""
    h_a, w_a = img_src.shape[:2]
    h_b, w_b = img_dst.shape[:2]
    h_max = max(h_a, h_b)
    total_w = w_a + w_b
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_h = min(22.0, max(3.5, h_max / 120.0))
    fig_w = fig_h * (total_w / h_max)
    if fig_w > 22.0:
        s = 22.0 / fig_w
        fig_w *= s; fig_h *= s

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=200)
    gs = fig.add_gridspec(1, 2, width_ratios=[w_a, w_b], wspace=0)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    for ax, img in ((ax_a, img_src), (ax_b, img_dst)):
        ax.imshow(img); ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.96)

    if len(kpts_ab) and mask_ab is not None:
        n = len(kpts_ab)
        if len(mask_ab) >= n:
            mask = mask_ab[:n]
        else:
            mask = np.zeros(n, dtype=bool)

        # scatter: outliers red, inliers orange
        pts_a = kpts_ab[:, :2]
        pts_b = kpts_ab[:, 2:]
        ax_a.scatter(pts_a[~mask, 0], pts_a[~mask, 1], c='red', s=3, zorder=5, alpha=0.6)
        ax_a.scatter(pts_a[mask, 0], pts_a[mask, 1], c='orange', s=6, zorder=6, alpha=0.9)
        ax_b.scatter(pts_b[~mask, 0], pts_b[~mask, 1], c='red', s=3, zorder=5, alpha=0.6)
        ax_b.scatter(pts_b[mask, 0], pts_b[mask, 1], c='orange', s=6, zorder=6, alpha=0.9)

        # lines only for inliers
        if mask.any():
            fig.canvas.draw()
            t = fig.transFigure.inverted()
            pa = t.transform(ax_a.transData.transform(pts_a[mask]))
            pb = t.transform(ax_b.transData.transform(pts_b[mask]))
            for i in range(len(pa)):
                fig.lines.append(plt.Line2D(
                    (pa[i, 0], pb[i, 0]), (pa[i, 1], pb[i, 1]),
                    transform=fig.transFigure, c=(1, 0.55, 0, 0.35),
                    linewidth=1.0, zorder=1,
                ))

    ax_a.text(0.01, 0.98,
              f"REDI-match\nsampled={sampled_ab}\ninliers={inliers_ab}\n(orange=inlier, red=outlier)",
              transform=ax_a.transAxes, fontsize=18, va="top", ha="left",
              color="white", zorder=999,
              bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.45))
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# Build model
# ═══════════════════════════════════════════════════════════════════

def build_model(resolution: int, symmetric: bool, device: torch.device):
    from redimatch.models.model_builder import (
        get_model,
        install_gp16_cholesky_solver,
    )
    coarse_to_name = {448: "low", 560: "medium", 672: "high"}
    model_res = coarse_to_name.get(resolution, "medium")
    model = get_model(
        pretrained_backbone=True,
        resolution=model_res,
        model_type="gp",
        freeze_encoder=True,
        exported_weights_path=None,
        symmetric=symmetric,
        upsample_preds=False,
        attenuate_cert=False,
    ).to(device)
    install_gp16_cholesky_solver(model)
    return model


# ═══════════════════════════════════════════════════════════════════
# Main function
# ═══════════════════════════════════════════════════════════════════

def main():
    import argparse
    ap = argparse.ArgumentParser(description="REDI-Match inference demo")
    ap.add_argument("--im_A", type=str, default=IM_A)
    ap.add_argument("--im_B", type=str, default=IM_B)
    ap.add_argument("--weights", type=str, default=WEIGHTS, help="Merged model weights")
    ap.add_argument("--save_sym", type=str, default=SAVE_SYM)
    ap.add_argument("--save_warp", type=str, default=SAVE_WARP)
    ap.add_argument("--coarse_res", type=int, default=576)
    ap.add_argument("--num_matches", type=int, default=10000)
    ap.add_argument("--c4", action="store_true", default=True)
    ap.add_argument("--no-c4", action=argparse.BooleanOptionalAction, dest="c4")
    ap.add_argument("--symmetric", action="store_true", default=True)
    args = ap.parse_args()

    im_A = _resolve(args.im_A)
    im_B = _resolve(args.im_B)
    weights_path = _resolve(args.weights)
    save_sym = _resolve(args.save_sym)
    save_warp = _resolve(args.save_warp)
    os.makedirs(os.path.dirname(save_sym), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[Model] Build + load weights", flush=True)
    model = build_model(args.coarse_res, args.symmetric, device)

    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    state = _extract_state_dict(ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[Weights] missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    model.eval()

    if args.c4:
        model.decoder.c4_rotation_matching = True
        model.decoder.c4_perm_reverse = True
        model.decoder.c4_symmetric_reverse = "align_to_query"
        # Use OT (fundamental) for direction selection; do not use cosine similarity alone.
        model.decoder.c4_hybrid = True
        model.decoder.c4_hybrid_alpha = 0.35
        model.decoder.c4_hybrid_temp = 0.05
        model.decoder.c4_hybrid_margin_keep = 0.03
        model.decoder.c4_hybrid_disagree_margin = 0.03
        model.decoder.c4_hybrid_ot_pool_size = 12
        model.decoder.c4_hybrid_ot_epsilon = 0.07
        model.decoder.c4_hybrid_ot_iters = 10
        print("[C4] enabled (fundamental_hybrid direction)", flush=True)
        dec = getattr(model, "decoder", None)
        if dec is not None and getattr(dec, "c4_rotation_matching", False):
            ok, _ = install_gp_intrinsic_rotation_router(model)
            print(f"[C4] gp_intrinsic rotation router {'enabled' if ok else 'not available, fallback to default'}", flush=True)

    print(f"[Match] {im_A} <-> {im_B}", flush=True)
    warp, certainty = model.match(im_A, im_B, device=device)

    # ── Print C4 detected rotation angle ──
    rot = getattr(getattr(model, "decoder", None), "_last_c4_rotation", None)
    if rot is None:
        print("[C4] no rotation record", flush=True)
    else:
        rot_idx = rot.get("rot_idx")
        if torch.is_tensor(rot_idx):
            rot_idx = int(rot_idx.detach().cpu().reshape(-1)[0].item()) % 4
        elif rot_idx is not None:
            rot_idx = int(rot_idx) % 4
        rot_margin = rot.get("rot_margin")
        if torch.is_tensor(rot_margin):
            rot_margin = float(rot_margin.detach().cpu().reshape(-1)[0].item())
        print(
            f"[C4] pred_rot_idx={rot_idx} pred_rot_deg={None if rot_idx is None else rot_idx * 90} "
            f"rot_margin={rot_margin}",
            flush=True,
        )

    im = Image.open(im_A); wA, hA = im.size
    imB = Image.open(im_B); wB, hB = imB.size

    if args.symmetric:
        W2 = warp.shape[2]
        W = W2 // 2
        warp_ab = warp[0, :, :W, :]
        cert_ab = certainty[0, :, :W]
        warp_ba = warp[0, :, W:, :]
        cert_ba = certainty[0, :, W:]
    else:
        warp_ab = warp[0]; cert_ab = certainty[0]
        warp_ba = None; cert_ba = None

    # ── Warp visualization ──
    model.visualize_warp(
        warp[0], certainty[0],
        im_A_path=im_A, im_B_path=im_B,
        device=device, symmetric=args.symmetric,
        save_path=save_warp,
    )
    print(f"[Saved] {save_warp}", flush=True)

    img_src = cv2.cvtColor(cv2.imread(im_A), cv2.COLOR_BGR2RGB)
    img_dst = cv2.cvtColor(cv2.imread(im_B), cv2.COLOR_BGR2RGB)

    # ── Bidirectional point-line diagram ──
    if args.symmetric and warp_ba is not None:
        sparse_ab, _ = model.sample(warp_ab, cert_ab, args.num_matches)
        k1, k2 = model.to_pixel_coordinates(sparse_ab, hA, wA, hB, wB)
        kpts_ab = np.hstack([k1.cpu().numpy(), k2.cpu().numpy()])
        inl_ab, mask_ab = _ransac_inliers_and_mask(k1.cpu().numpy(), k2.cpu().numpy())

        sparse_ba, _ = model.sample(warp_ba, cert_ba, args.num_matches)
        bk1, bk2 = model.to_pixel_coordinates(sparse_ba, hB, wB, hA, wA)
        kpts_ba_full = np.hstack([bk2.cpu().numpy(), bk1.cpu().numpy()])
        inl_ba, mask_ba = _ransac_inliers_and_mask(bk2.cpu().numpy(), bk1.cpu().numpy())
        _draw_bidirectional(
            img_src, img_dst,
            kpts_ab, kpts_ba_full,
            inl_ab, inl_ba,
            mask_ab, mask_ba,
            len(kpts_ab), len(kpts_ba_full),
            save_sym,
        )
        print(f"[Saved] {save_sym}", flush=True)


if __name__ == "__main__":
    main()
