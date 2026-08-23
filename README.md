# REDI-Match public release

这是 REDI-Match 的可运行发布版，面向图像匹配推理、可视化和公开评估。
仓库只包含必要的推理代码、评估脚本、示例数据和预训练权重的下载入口；训练、蒸馏、私有数据处理和内部实验代码不在发布范围内。

## 目录

```text
demo/                         # 匹配可视化 demo
eval/                         # MegaDepth、ScanNet、HPatches 等评估脚本
redimatch/                   # 公开推理运行时
assets/                      # demo 输入图像
results/                     # 已生成的 demo 结果
data/README.md               # 外部评估数据的目录约定
scripts/download_weights.py  # 从 Hugging Face 下载权重
scripts/check_release.py     # 发布边界和权重兼容性检查
models/                       # 本地权重目录（*.pth 不提交 Git）
```

## 环境安装

建议使用 Python 3.12 和 CUDA 环境：

```bash
conda create -n redimatch python=3.12 -y
conda activate redimatch
pip install -r requirements.txt
```

项目默认使用原生 PyTorch 相关算子，不需要安装 `escnn`、Fortran 编译器、DINOv3 或训练依赖。
如果需要 CUDA 加速扩展，可在确认本机 PyTorch/CUDA 版本兼容后额外安装：

```bash
pip install -r requirements-optional.txt
```

## 预训练权重

`indoor.pth` 和 `outdoor.pth` 是大文件，不应提交到 GitHub；请从 Hugging Face 模型仓库下载，代码仓库只保留 `models/.gitkeep`。
当前预训练权重仓库为 [`YinjiGe/REDI-Match`](https://huggingface.co/YinjiGe/REDI-Match)：

```bash
pip install huggingface_hub
python scripts/download_weights.py \
  --repo-id YinjiGe/REDI-Match \
  --output-dir models
```

也可以手动下载并放置为：

```text
models/indoor.pth
models/outdoor.pth
```

如果模型仓库为私有仓库，请先执行 `huggingface-cli login`，或通过 `--token` 传入访问令牌。

## Demo

运行默认示例（默认使用 `remote_satast` 图像对和 outdoor 权重）：

```bash
python demo/demo_match.py
```

结果保存在 `results/demo_match_symmetric.jpg` 和 `results/demo_match_warp.jpg`。
可指定任意图像对、权重和输出位置；默认采样 10,000 个匹配点：

```bash
python demo/demo_match.py \
  --im_A assets/indoor_scannet_A.jpg \
  --im_B assets/indoor_scannet_B.jpg \
  --weights models/indoor.pth \
  --save_sym results/indoor_symmetric.jpg \
  --save_warp results/indoor_warp.jpg
```

仓库中的四组示例输入为：

```text
assets/indoor_scannet_A.jpg     assets/indoor_scannet_B.jpg
assets/remote_satast_A.jpg      assets/remote_satast_B.jpg
assets/sacre_coeur_A.jpg        assets/sacre_coeur_B_rot180.jpg
assets/toronto_A.jpg            assets/toronto_B_rot180.jpg
```

对应的四组可视化结果位于 `results/`。室内图像建议使用 `models/indoor.pth`，其余示例使用 `models/outdoor.pth`。

## 评估数据

完整基准数据集因体积和各自许可证未随仓库发布。请按照 [data/README.md](data/README.md) 下载或挂载到 `data/`，或通过命令行参数指定外部路径。

常用目录约定如下：

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

## 评估命令

所有评估脚本默认从项目内 `models/` 和 `data/` 读取文件；缺少数据时会在启动阶段报出路径错误。

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

# 其他基准
python eval/eval_roto360.py
python eval/eval_satast.py
python eval/eval_wxbs.py

# 显式指定外部数据目录的例子
python eval/eval_hpatches.py --hpatches_root /path/to/hpatches
python eval/eval_satast.py \
  --satast_annotations /path/to/satast/satast_annotations_with_rot \
  --satast_image_root /path/to/satast
python eval/eval_wxbs.py --wxbs_root /path/to/WxBS
```

评估结果写入当前项目的 `results/`。CUDA 评估脚本通常需要 NVIDIA GPU；运行 `--help` 可查看每个脚本的完整参数。

## 延迟测试

```bash
python eval/bench_latency.py \
  --weights models/outdoor.pth \
  --resolution 576
```

## 发布边界检查

发布前可运行：

```bash
python scripts/check_release.py
```

若只检查代码边界而暂时没有权重：

```bash
python scripts/check_release.py --skip-weights
```

本公开版明确不包含：

- REDI 蒸馏实现和训练脚本；
- 匹配模型训练、EMA、实验监控和私有数据准备代码；
- `escnn_lib`、DINOv3 教师权重及其他训练专用依赖。

## License

发布到 GitHub 前，请在根目录补充项目实际采用的许可证文件，并确认示例图像、评估数据和预训练权重分别符合其来源许可证或分发条款。
