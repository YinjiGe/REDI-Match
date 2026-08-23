"""
Roto-360 MMA evaluation (consistent with evaluate.py --eval_dataset roto360).

After copying to another project, just implement keypoint+descriptor extraction for one image:

    def extract(path: str) -> (kpts, desc)
        # kpts: (N, 2) pixel coords; desc: (N, D) consistent with MNN input in evaluate (torch or numpy)

    from benchmarks.roto360 import Roto360Benchmark, match_descriptors_mnn

    bench = Roto360Benchmark("/path/to/data/roto360", extract_fn=extract, split="full")
    print(bench.run())   # mma@3, mma@5, mma@10, pred_matches, total_points

Data: roto360_root/HPatches_rot_image.txt + splits.json; reference image is *_rot0.jpg, GT H in target image dir as 1_rot{angle}.txt.
"""
from __future__ import annotations

import json
import math
import os
from typing import Callable, List, Sequence, Tuple

import numpy as np

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(x, **kwargs):
        return x


ExtractFn = Callable[[str], Tuple[np.ndarray, np.ndarray]]

# ``eval_modes`` CLI (i/v/all) -> split name in ``splits.json``
EVAL_MODE_TO_SPLIT = {"i": "illum", "v": "view", "all": "full"}


def resolve_eval_split(eval_mode: str) -> str:
    """``i`` / ``v`` / ``all`` or use split name directly from splits.json."""
    key = str(eval_mode).strip().lower()
    if key in EVAL_MODE_TO_SPLIT:
        return EVAL_MODE_TO_SPLIT[key]
    return key


def count_target_pairs(roto360_root: str, split: str) -> int:
    """Number of non-``_rot0`` target images (i.e., MMA evaluation pairs)."""
    rel_paths = _load_image_list(roto360_root, resolve_eval_split(split))
    n = 0
    for rel in rel_paths:
        if "_rot0.jpg" in rel:
            continue
        n += 1
    return n


def prefix_metrics_for_eval_mode(metrics: dict, eval_mode: str) -> dict:
    """Add ``roto360_{i|v|all}_MMA@*`` keys alongside original ``mma@*``, compatible with old result JSON."""
    mode = str(eval_mode).strip().lower()
    pfx = {"i": "i", "v": "v", "all": "all"}.get(mode, mode)
    out = dict(metrics)
    for thr in (3, 5, 10, 20):
        k = f"mma@{thr}"
        if k in metrics:
            out[f"roto360_{pfx}_MMA@{thr}"] = metrics[k]
    return out


def load_gt_homography(target_image_path: str) -> np.ndarray:
    folder = os.path.dirname(os.path.abspath(target_image_path))
    stem = os.path.splitext(os.path.basename(target_image_path))[0]
    h = np.fromfile(os.path.join(folder, stem + ".txt"), sep=" ")
    return h.reshape(3, 3)


def warp_points(src: np.ndarray, h: np.ndarray) -> np.ndarray:
    h = h.astype(np.float64).copy()
    h /= h[2, 2]
    ones = np.ones((len(src), 1))
    pts = np.hstack([src[:, :2], ones])
    warped = (h @ pts.T).T
    return warped[:, :2] / warped[:, 2:3]


def mnn_matcher(desc_a: np.ndarray, desc_b: np.ndarray) -> np.ndarray:
    a = np.asarray(desc_a, dtype=np.float32)
    b = np.asarray(desc_b, dtype=np.float32)
    sim = a @ b.T
    nn12 = np.argmax(sim, axis=1)
    nn21 = np.argmax(sim, axis=0)
    ids = np.arange(sim.shape[0])
    mask = ids == nn21[nn12]
    return np.stack([ids[mask], nn12[mask]], axis=1)


def match_descriptors_mnn(
    kpts_ref: np.ndarray,
    kpts_tgt: np.ndarray,
    desc_ref: np.ndarray,
    desc_tgt: np.ndarray,
    h_gt: np.ndarray,
) -> Tuple[List[float], int, float]:
    """MNN + GT H reprojection error → (per-match pixel error, match count, total_points)."""
    matches = mnn_matcher(desc_ref, desc_tgt)
    npt = (len(desc_ref) + len(desc_tgt)) / 2.0
    if matches.shape[0] == 0:
        return [], 0, npt
    k1 = kpts_ref[matches[:, 0], :2]
    k2 = kpts_tgt[matches[:, 1], :2]
    gt = warp_points(k1, h_gt)
    dists = [math.hypot(x2 - gx, y2 - gy) for (x2, y2), (gx, gy) in zip(k2, gt)]
    return dists, int(matches.shape[0]), float(npt)


def _resolve_image_path(roto360_root: str, rel: str) -> str:
    """``HPatches_rot_image.txt`` lines are ``roto360/<scene>/...``, data root is typically ``.../data/roto360``."""
    rel = rel.strip().replace("\\", "/")
    prefix = "roto360/"
    if rel.startswith(prefix):
        rel = rel[len(prefix) :]
    return os.path.join(os.path.abspath(roto360_root), rel)


def _load_image_list(roto360_root: str, split: str) -> List[str]:
    root = os.path.abspath(roto360_root)
    with open(os.path.join(root, "splits.json"), encoding="utf-8") as f:
        scenes = set(json.load(f)[split]["test"])
    paths = []
    with open(os.path.join(root, "HPatches_rot_image.txt"), encoding="utf-8") as f:
        for line in sorted(f):
            line = line.strip()
            if line and line.split("/")[-2] in scenes:
                paths.append(line)
    return paths


def _aggregate_mma(
    n_match: Sequence[int],
    dists: Sequence[Sequence[float]],
    n_pts: Sequence[float],
    *,
    max_th: int = 10,
) -> dict:
    th = max_th
    precs = []
    for nm, ds in zip(n_match, dists):
        ok = np.zeros(th)
        for d in ds:
            for t in range(th):
                if d <= t + 1:
                    ok[t] += 1
        precs.append(0.0 if nm == 0 else ok / nm * 100.0)
    p = np.mean(np.stack(precs), axis=0)
    out = {
        "mma@3": float(p[2]),
        "mma@5": float(p[4]),
        "mma@10": float(p[9]),
        "pred_matches": float(np.mean(n_match)),
        "total_points": float(np.mean(n_pts)),
    }
    if th >= 20:
        out["mma@20"] = float(p[19])
    return out


class Roto360Benchmark:
    """Roto-360 MMA benchmark (HPatches_rot_image.txt + splits.json + GT ``.txt`` homography)."""

    def __init__(
        self,
        roto360_root: str,
        extract_fn: ExtractFn | None = None,
        split: str = "full",
        *,
        eval_mode: str | None = None,
    ) -> None:
        self.root = os.path.abspath(roto360_root)
        self.extract_fn = extract_fn
        if eval_mode is not None:
            split = resolve_eval_split(eval_mode)
        self.split = resolve_eval_split(split)
        self.eval_mode = eval_mode

    @classmethod
    def for_eval_mode(cls, roto360_root: str, eval_mode: str) -> "Roto360Benchmark":
        return cls(roto360_root, eval_mode=eval_mode)

    @property
    def n_target_pairs(self) -> int:
        return count_target_pairs(self.root, self.split)

    def run(self, show_progress: bool = True) -> dict:
        if self.extract_fn is None:
            raise ValueError("extract_fn is required for run(); use run_romav2(model) for RoMaV2.")
        rel_paths = _load_image_list(self.root, self.split)
        n_match, all_dists, n_pts = [], [], []
        src_path = None
        it = tqdm(rel_paths, desc="Roto360 MMA") if show_progress else rel_paths

        for rel in it:
            path = _resolve_image_path(self.root, rel)
            if "_rot0.jpg" in rel:
                src_path = path
                continue
            if src_path is None:
                continue
            h = load_gt_homography(path)
            k1, d1 = self.extract_fn(src_path)
            k2, d2 = self.extract_fn(path)
            ds, nm, tp = match_descriptors_mnn(k1, k2, d1, d2, h)
            n_match.append(nm)
            all_dists.append(ds)
            n_pts.append(tp)

        return _aggregate_mma(n_match, all_dists, n_pts)

    def run_romav2(
        self,
        model,
        *,
        num_samples: int = 5000,
        show_progress: bool = True,
        max_th: int = 20,
    ) -> dict:
        """Same data list and MMA summary as ``run()``, matching done by RoMaV2."""
        rel_paths = _load_image_list(self.root, self.split)
        n_match, all_dists, n_pts = [], [], []
        src_path = None
        it = tqdm(rel_paths, desc="Roto360 RoMaV2") if show_progress else rel_paths

        for rel in it:
            path = _resolve_image_path(self.root, rel)
            if "_rot0.jpg" in rel:
                src_path = path
                continue
            if src_path is None:
                continue
            h = load_gt_homography(path)
            ds, nm, tp = match_romav2_pair(
                model, src_path, path, h, num_samples=num_samples
            )
            n_match.append(nm)
            all_dists.append(ds)
            n_pts.append(tp)

        return _aggregate_mma(n_match, all_dists, n_pts, max_th=max_th)

    def benchmark(
        self,
        model,
        *,
        num_samples: int = 5000,
        show_progress: bool = True,
        max_th: int = 10,
    ) -> dict:
        """RoMa / RoMaV2 dense matching + ``sample`` sparse points, MMA@3/5/10 (same as ``run_romav2``)."""
        return self.run_romav2(
            model,
            num_samples=num_samples,
            show_progress=show_progress,
            max_th=max_th,
        )


def _norm_to_pixel(pts_norm: np.ndarray, w: int, h: int, offset: float = 0.5) -> np.ndarray:
    pts_norm = np.asarray(pts_norm, dtype=np.float64)
    pts_pix = np.zeros_like(pts_norm, dtype=np.float64)
    pts_pix[..., 0] = (w * (pts_norm[..., 0] + 1.0) / 2.0) - offset
    pts_pix[..., 1] = (h * (pts_norm[..., 1] + 1.0) / 2.0) - offset
    return pts_pix


def _symmetric_crop_a2b_matches(warp, certainty):
    """symmetric match is (B,H,2W,4); evaluation only takes left half A→B (aligned with RoMaV2 sample when bidirectional=False)."""
    w_full = int(warp.shape[2])
    half = w_full // 2
    warp = warp[:, :, :half, :]
    if certainty.dim() == 3:
        certainty = certainty[:, :, :half]
    elif certainty.dim() == 2:
        certainty = certainty[:, :half]
    else:
        certainty = certainty[..., :half]
    return warp, certainty


def match_romav2_pair(
    model,
    ref_path: str,
    tgt_path: str,
    h_gt: np.ndarray,
    *,
    num_samples: int = 5000,
) -> Tuple[List[float], int, float]:
    """RoMaV2 dense matching + sparse sampling, use GT homography to map ref points to tgt then compute pixel error."""
    from PIL import Image

    im_a = Image.open(ref_path)
    im_b = Image.open(tgt_path)
    w1, h1 = im_a.size
    w2, h2 = im_b.size

    dense_preds = model.match(ref_path, tgt_path)
    if isinstance(dense_preds, dict):
        warp = dense_preds.get("warp") or dense_preds.get("warp_AB")
        certainty = dense_preds.get("certainty") or dense_preds.get("overlap_AB")
        if warp is None or certainty is None:
            raise ValueError(f"Unsupported match dict keys: {list(dense_preds.keys())}")
    else:
        warp, certainty = dense_preds

    if getattr(model, "symmetric", False):
        warp, certainty = _symmetric_crop_a2b_matches(warp, certainty)

    sparse = model.sample(warp, certainty, num_samples)

    good_matches = sparse[0]
    if hasattr(good_matches, "detach"):
        good_matches = good_matches.detach().cpu().numpy()
    pos_a = _norm_to_pixel(good_matches[:, :2], w1, h1)
    pos_b = _norm_to_pixel(good_matches[:, 2:], w2, h2)

    if len(pos_a) == 0:
        return [], 0, 0.0

    gt = warp_points(pos_a, h_gt)
    dists = [
        float(math.hypot(x2 - gx, y2 - gy))
        for (x2, y2), (gx, gy) in zip(pos_b, gt)
    ]
    npt = float(len(pos_a))
    return dists, len(dists), npt
