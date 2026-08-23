"""ScanNet benchmark intrinsics loading: prefers per-frame ``intrinsic_color_{id}.txt``, otherwise falls back to ``intrinsic_color.txt``."""
from __future__ import annotations

import os.path as osp
from typing import Union

import numpy as np

__all__ = ["load_scannet_intrinsic_color"]


def load_scannet_intrinsic_color(
    data_root: str, scene_name: str, frame_id: Union[int, str]
) -> np.ndarray:
    # Supports integer frame IDs or ``1485_Brot`` (scans_rot_3 fixed-A dataset)
    fid_key = str(frame_id).strip()
    base = osp.join(data_root, "scans_test", scene_name, "intrinsic")
    per = osp.join(base, f"intrinsic_color_{fid_key}.txt")
    path = per if osp.isfile(per) else osp.join(base, "intrinsic_color.txt")
    with open(path, "r") as f:
        lines = [r for r in f.read().split("\n") if r.strip()]
    return np.stack([np.array([float(i) for i in r.split()]) for r in lines])
