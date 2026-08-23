# Evaluation data

The benchmark datasets are intentionally not bundled in this repository because of their size and their individual dataset licenses.

Place or link the datasets using the following layout:

```text
data/
├── megadepth/
├── megadepth_rot/
├── scannet/
│   ├── scans/
│   └── scans_rot/
├── hpatches/
├── roto360/
├── satast/
└── WxBS/
```

The evaluation scripts also accept explicit dataset roots where supported. See the main README for the commands.

This repository does not redistribute copies of the benchmark datasets. Please
download each dataset from its official source, follow its license and terms
of use, and mount or link it under `data/` as shown above. If a dataset is not
redistributable, keep it outside the Git repository and pass its path through
the supported command-line option.
