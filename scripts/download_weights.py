#!/usr/bin/env python3
"""Download public REDI-Match checkpoints from Hugging Face."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="Hugging Face model repository, e.g. USER/REPO")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--token", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("indoor.pth", "outdoor.pth"):
        path = hf_hub_download(
            repo_id=args.repo_id,
            filename=filename,
            revision=args.revision,
            token=args.token,
            local_dir=str(args.output_dir),
        )
        print(f"[Downloaded] {path}")


if __name__ == "__main__":
    main()
