from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from wxbs_benchmark.dataset import WxBSDataset
from wxbs_benchmark.metrics import PCK

# Note: wxbs_benchmark.evaluation.evaluate_corrs hardcodes
# ``WxBSDataset('.WxBS', download=True)`` would re-download dataset under cwd.
# Here we use the same PCK formula as the official one, but GT and inference share self.dataset to avoid re-download.


def _eval_device(model: Any) -> torch.device:
    if isinstance(model, torch.nn.Module):
        try:
            return next(model.parameters()).device
        except StopIteration:
            pass
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_normalized_pixel_coords(pts: torch.Tensor, *, H: int, W: int) -> torch.Tensor:
    """Pixel (x, y) -> grid_sample uses [-1, 1], consistent with RegressionMatcher.to_normalized_coordinates."""
    x = pts[..., 0]
    y = pts[..., 1]
    return torch.stack((2.0 / float(W) * x - 1.0, 2.0 / float(H) * y - 1.0), dim=-1)


def _to_pixel_normalized_coords(coords: torch.Tensor, *, H: int, W: int) -> torch.Tensor:
    """[-1, 1] coordinates -> pixel (x, y), consistent with RegressionMatcher._to_pixel_coordinates."""
    return torch.stack(
        (W / 2.0 * (coords[..., 0] + 1.0), H / 2.0 * (coords[..., 1] + 1.0)),
        dim=-1,
    )


def _bhwc_grid_sample(
    warp_bhwc: torch.Tensor,
    grid_xy: torch.Tensor,
    *,
    mode: str = "bilinear",
    align_corners: bool = False,
) -> torch.Tensor:
    """
    warp_bhwc: (B, H, W, C)
    grid_xy: (B, Hg, Wg, 2), last dim is (x, y), range [-1, 1]
    Returns: (B, Hg, Wg, C)
    """
    if warp_bhwc.dim() != 4:
        raise ValueError(f"expected BHWC warp, got shape {tuple(warp_bhwc.shape)}")
    inp = warp_bhwc.permute(0, 3, 1, 2).contiguous()
    out = F.grid_sample(inp, grid_xy, mode=mode, align_corners=align_corners)
    return out.permute(0, 2, 3, 1).contiguous()


def _tensor_to_pil_rgb(t: torch.Tensor) -> Image.Image:
    """(3, H, W) float [0,1] or uint8 -> PIL RGB."""
    x = t.detach().float().cpu()
    if x.dim() == 4:
        x = x[0]
    if x.shape[0] == 3:
        x = x.permute(1, 2, 0)
    arr = x.numpy()
    if arr.max() <= 1.0:
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    else:
        arr = arr.clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _simple_pair_vis(
    img1: np.ndarray,
    img2: np.ndarray,
    out_path: Path,
) -> None:
    """Left-right concatenation visualization (no dependency on romav2.vis)."""
    a = np.asarray(img1)
    b = np.asarray(img2)
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    if b.ndim == 2:
        b = np.stack([b] * 3, axis=-1)
    h = max(a.shape[0], b.shape[0])

    def _resize_h(im: np.ndarray, target_h: int) -> np.ndarray:
        if im.shape[0] == target_h:
            return im
        scale = target_h / float(im.shape[0])
        tw = max(1, int(round(im.shape[1] * scale)))
        return np.array(Image.fromarray(im).resize((tw, target_h), Image.Bilinear))

    a = _resize_h(a, h)
    b = _resize_h(b, h)
    cat = np.concatenate([a, b], axis=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(cat.astype(np.uint8)).save(out_path)


class WxBSBenchmark:
    @dataclass(frozen=True)
    class Cfg:
        subset: str = "test"
        dataset_path: str = "data/WxBS"
        download: bool = True

    def __init__(self, cfg: Cfg):
        self.subset = cfg.subset
        WxBSDataset.urls["v1.1"][0] = (
            "https://github.com/Parskatt/storage/releases/download/wxbs/WxBS_v1.1.zip"
        )

        def wrap(f):
            def __getitem__(self, idx):
                out = f(self, idx)
                return {
                    **out,
                    "imgfname1": self.pairs[idx][0],
                    "imgfname2": self.pairs[idx][1],
                }

            return __getitem__

        WxBSDataset.__getitem__ = wrap(WxBSDataset.__getitem__)
        self.dataset = WxBSDataset(
            cfg.dataset_path, subset=self.subset, download=cfg.download
        )

    def __call__(self, model: Any, step: int):
        estimated_right = []
        estimated_left = []
        gt_corrs = []
        names = []
        ths = np.arange(20)
        dev = _eval_device(model)
        model_name = str(getattr(model, "name", "model"))

        for pair_dict in tqdm(self.dataset):
            names.append(pair_dict["name"])
            gt_corrs.append(pair_dict["pts"])
            H_A, W_A = pair_dict["img1"].shape[:2]
            H_B, W_B = pair_dict["img2"].shape[:2]
            points_left = torch.from_numpy(pair_dict["pts"][:, :2]).to(dev).float()
            points_right = torch.from_numpy(pair_dict["pts"][:, 2:4]).to(dev).float()
            n_points_left = _to_normalized_pixel_coords(points_left, H=H_A, W=W_A)
            n_points_right = _to_normalized_pixel_coords(points_right, H=H_B, W=W_B)

            use_paths = bool(pair_dict.get("imgfname1") and pair_dict.get("imgfname2"))
            if use_paths or str(getattr(model, "name", "")).lower() == "roma":
                preds: tuple[torch.Tensor, torch.Tensor] = model.match(
                    pair_dict["imgfname1"], pair_dict["imgfname2"]
                )  # type: ignore[assignment]
                warp_bidirectional = preds[0]
                W_pred = warp_bidirectional.shape[2]
                warp_AB = warp_bidirectional[:, :, : W_pred // 2, 2:]
                warp_BA = warp_bidirectional[:, :, W_pred // 2 :, :2]
                overlap_AB = preds[1][:, :, : W_pred // 2]
                overlap_BA = preds[1][:, :, W_pred // 2 :]
            else:
                preds = model.match(pair_dict["img1"], pair_dict["img2"])
                warp_AB = preds["warp_AB"]
                warp_BA = preds["warp_BA"]
                overlap_AB = preds["overlap_AB"]
                overlap_BA = preds["overlap_BA"]

            grid_l = n_points_left[None, None, :, :].to(warp_AB.device, dtype=warp_AB.dtype)
            grid_r = n_points_right[None, None, :, :].to(warp_BA.device, dtype=warp_BA.dtype)

            n_est_points_right = _bhwc_grid_sample(
                warp_AB,
                grid_l,
                mode="bilinear",
                align_corners=False,
            )
            n_est_points_left = _bhwc_grid_sample(
                warp_BA,
                grid_r,
                mode="bilinear",
                align_corners=False,
            )
            est_points_right = _to_pixel_normalized_coords(
                n_est_points_right, H=H_B, W=W_B
            )
            est_points_left = _to_pixel_normalized_coords(
                n_est_points_left, H=H_A, W=W_A
            )

            estimated_right.append(est_points_right.cpu().numpy()[0, 0])
            estimated_left.append(est_points_left.cpu().numpy()[0, 0])

        assert len(estimated_right) == len(gt_corrs)
        assert len(estimated_left) == len(gt_corrs)
        all_res = []
        per_pair_results: dict[str, Any] = {}
        for est_right, est_left, gt_pts, pairname in zip(
            estimated_right, estimated_left, gt_corrs, names
        ):
            res = 0.5 * (
                PCK(est_right, gt_pts[:, 2:4], ths)
                + PCK(est_left, gt_pts[:, :2], ths)
            )
            per_pair_results[pairname] = res
            all_res.append(res)
        per_pair_results["average"] = np.stack(all_res, axis=1).mean(axis=1)
        thresholds = ths.tolist()
        return per_pair_results, thresholds
