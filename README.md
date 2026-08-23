# REDI-Match Public Release

This is the runnable REDI-Match release for image-matching inference, visualization, and public benchmarks.
It includes only the inference runtime, evaluation scripts, example data, and instructions for downloading pretrained weights. Training, distillation, private data processing, and internal experiments are excluded.

## Repository layout

```text
demo/                         # Matching visualization demo
eval/                         # MegaDepth, ScanNet, HPatches, and other benchmarks
redimatch/                    # Public inference runtime
assets/                       # Demo input images
results/                      # Generated demo results
data/README.md                # External evaluation data layout
scripts/download_weights.py   # Download weights from Hugging Face
scripts/check_release.py      # Check release boundaries and weight compatibility
models/                       # Local weight directory (*.pth is not tracked by Git)
```

## Installation

Python 3.12 and a CUDA environment are recommended:

```bash
conda create -n redimatch python=3.12 -y
conda activate redimatch
pip install -r requirements.txt
```

The release uses native PyTorch operators by default. It does not require `escnn`, a Fortran compiler, DINOv3, or training dependencies.
For optional CUDA extensions, first confirm compatibility with your local PyTorch and CUDA versions:

```bash
pip install -r requirements-optional.txt
```

## Pretrained weights

`indoor.pth` and `outdoor.pth` are large files and should not be committed to GitHub. Download them from the Hugging Face model repository; the code repository keeps only `models/.gitkeep`.
The pretrained weights are hosted at [`YinjiGe/REDI-Match`](https://huggingface.co/YinjiGe/REDI-Match):

```bash
pip install huggingface_hub
python scripts/download_weights.py \
  --repo-id YinjiGe/REDI-Match \
  --output-dir models
```

You can also download the files manually and place them at:

```text
models/indoor.pth
models/outdoor.pth
```

For a private model repository, run `huggingface-cli login` first or pass an access token with `--token`.

## Demo

Run the default example. It uses the `remote_satast` image pair and the outdoor weights:

```bash
python demo/demo_match.py
```

Results are saved to `results/demo_match_symmetric.jpg` and `results/demo_match_warp.jpg`.
You can specify any image pair, weight file, and output path. The default is 10,000 sampled matches:

```bash
python demo/demo_match.py \
  --im_A assets/indoor_scannet_A.jpg \
  --im_B assets/indoor_scannet_B.jpg \
  --weights models/indoor.pth \
  --save_sym results/indoor_symmetric.jpg \
  --save_warp results/indoor_warp.jpg
```

The repository includes four example pairs:

```text
assets/indoor_scannet_A.jpg     assets/indoor_scannet_B.jpg
assets/remote_satast_A.jpg      assets/remote_satast_B.jpg
assets/sacre_coeur_A.jpg        assets/sacre_coeur_B_rot180.jpg
assets/toronto_A.jpg            assets/toronto_B_rot180.jpg
```

The corresponding visualizations are in `results/`. Use `models/indoor.pth` for indoor images and `models/outdoor.pth` for the other examples.

## Evaluation data

Full benchmark datasets are not included because of their size and licensing terms. Follow [data/README.md](data/README.md) to download or mount them under `data/`, or provide external paths through command-line arguments.

Typical directory layout:

```text
data/megadepth/
data/megadepth_rot/
data/scannet/scans/
data/scannet/scans_rot/
data/hpatches/
data/roto360/
data/satast/
data/WxBS/
```

## Evaluation commands

All evaluation scripts read weights from `models/` and data from `data/` by default. Missing data paths are reported at startup.

```bash
# MegaDepth plain / rot
python eval/eval_megadepth.py --mode plain
python eval/eval_megadepth.py --mode rot

# ScanNet plain / rot
python eval/eval_scannet.py --mode plain
python eval/eval_scannet.py --mode rot

# HPatches plain / rot
python eval/eval_hpatches.py --mode plain
python eval/eval_hpatches.py --mode rot

# Other benchmarks
python eval/eval_roto360.py
python eval/eval_satast.py
python eval/eval_wxbs.py

# Examples with external data roots
python eval/eval_hpatches.py --hpatches_root /path/to/hpatches
python eval/eval_satast.py \
  --satast_annotations /path/to/satast/satast_annotations_with_rot \
  --satast_image_root /path/to/satast
python eval/eval_wxbs.py --wxbs_root /path/to/WxBS
```

Evaluation outputs are written to `results/`. CUDA evaluation scripts usually require an NVIDIA GPU. Run `--help` for the full options of each script.

## Latency benchmark

```bash
python eval/bench_latency.py \
  --weights models/outdoor.pth \
  --resolution 576
```

## Release checks

Run before publishing:

```bash
python scripts/check_release.py
```

To check the release boundary without local weights:

```bash
python scripts/check_release.py --skip-weights
```

This public release does not include:

- REDI distillation and training scripts;
- model training, EMA, experiment monitoring, or private data preparation code;
- `escnn_lib`, DINOv3 teacher weights, or other training-only dependencies.

## License

Before publishing to GitHub, add the license used by the project to the repository root. Confirm that the example images, evaluation datasets, and pretrained weights comply with their respective licenses or distribution terms.
