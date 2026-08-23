#!/usr/bin/env python3
"""Validate the public runtime, checkpoint compatibility, and release boundary."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FORBIDDEN = (
    ROOT / "escnn_lib",
    ROOT / "REDI",
    ROOT / "eval" / "train_indoor.py",
    ROOT / "eval" / "train_outdoor.py",
    ROOT / "eval" / "timing_profile_utils.py",
    ROOT / "redimatch" / "models" / "dino_vgg.py",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-weights", action="store_true")
    args = parser.parse_args()

    forbidden = [path for path in FORBIDDEN if path.exists()]
    if forbidden:
        raise SystemExit("Forbidden release paths exist:\n" + "\n".join(map(str, forbidden)))

    import redimatch  # noqa: F401
    from redimatch.models.model_builder import get_model

    model = get_model(
        resolution="medium",
        model_type="gp",
        freeze_encoder=True,
        exported_weights_path=None,
        symmetric=True,
        upsample_preds=False,
        attenuate_cert=False,
        use_custom_corr=False,
    )

    if args.skip_weights:
        print("[OK] public runtime and release boundary")
        return

    import torch

    for name in ("indoor", "outdoor"):
        checkpoint_path = ROOT / "models" / f"{name}.pth"
        if not checkpoint_path.is_file():
            raise SystemExit(f"Missing checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
        state = {
            key[7:] if key.startswith("module.") else key: value
            for key, value in state.items()
        }
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise SystemExit(
                f"{name}.pth incompatible: missing={len(missing)} unexpected={len(unexpected)}"
            )
        print(f"[OK] {name}.pth")


if __name__ == "__main__":
    main()
