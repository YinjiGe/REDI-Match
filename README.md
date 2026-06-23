# REDI-Match

### Rotation-Equivariant Distillation for Efficient and Robust Dense Matching

**Yinji Ge**<sup>1*</sup>, **Guixu Zheng**<sup>2*</sup>, **Wulong Guo**<sup>3</sup>, **Qian Feng**<sup>4</sup>, **Xu Wu**<sup>1</sup>, **Kai Zhou**<sup>1†</sup>, **Xinyuan Liu**<sup>1†</sup>, **Fei Xing**<sup>1†</sup>

<sup>1</sup> Tsinghua University &nbsp; <sup>2</sup> Southern University of Science and Technology &nbsp; <sup>3</sup> Beihang University &nbsp; <sup>4</sup> Zhejiang University

<small><sup>*</sup> Equal contribution &nbsp; <sup>†</sup> Corresponding authors</small>

---

## 📄 Abstract

Vision Foundation Models (VFMs) have significantly advanced dense feature matching, yet severe in-plane rotation remains a critical challenge. Existing solutions face a fundamental dilemma: data-driven methods require inefficient parameter scaling to implicitly learn rotations, whereas strictly equivariant networks lack the semantic capacity of modern VFMs. To break this architectural bottleneck, we propose **REDI-Match**, an efficient framework driven by a novel **Rotation-Equivariant Distillation (REDI)** paradigm. Instead of relying on massive data augmentation, REDI distills the powerful semantics of a VFM into a lightweight, strictly rotation-equivariant encoder, structurally capturing geometric variations early at the feature level. To fully exploit these features, we equip the decoder with an entropy-driven spatial alignment module that explicitly locks onto the canonical coordinate system, eliminating global ambiguity before continuous refinement.

---

## 🎯 Key Highlights

| | |
|---|---|
| 🏆 **+13.89%** AUC@5° on SatAst | Outperforms SOTA under extreme rotations |
| ⚡ **1.9× faster** than RoMa v2 | Real-time **~41 FPS** on RTX 4090 |
| 🧱 **85M** parameters | 5× fewer than RoMa v2 (425M) |
| 🎓 **112× compression** | Distills 303M VFM → 2.7M equivariant encoder |
| 🔄 **Strict C₄-equivariant** | Mathematically guaranteed rotation consistency |

---

## 🏗️ Pipeline Overview

![pipeline](assets/pipeline.png)

**Left (Stage 1):** A frozen VFM (DINOv3-L) is distilled into a lightweight G-CNN via MSE loss, projecting non-equivariant semantics onto a strictly rotation-equivariant manifold.  
**Right (Stage 2):** The entropy-driven rotation decoder evaluates discrete hypotheses to lock onto the canonical orientation, enabling robust dense matching without rotation-augmented training data.

---

## 📊 Results

### Radar Comparison

![radar](assets/radar.png)

REDI-Match achieves the best overall accuracy–efficiency trade-off across all benchmarks.

### Qualitative Comparison under Severe Rotations

![qualitative](assets/qualitative.png)

Across outdoor (MegaDepth-C4), indoor (ScanNet-C4), and remote sensing (SatAst) — REDI-Match consistently maintains dense, accurate alignments while SOTA methods suffer catastrophic failure on SatAst.

---

## ⚖️ Equivariance Analysis

![equivariance](assets/equivariance.png)

Our distilled equivariant encoder preserves strict spatial and semantic consistency under rotation, achieving near-zero feature degradation. In contrast, standard non-equivariant foundation models exhibit severe structural collapse.

---

## 🔬 Entropy-Driven Rotation Detection

![entropy](assets/entropy.png)

Correct matching is consistently accompanied by low-entropy distributions across orientation hypotheses. The entropy-driven module identifies the optimal canonical orientation without any training.

---

## 📦 Code & Models

> 🚧 **Coming Soon** — Code, pretrained weights, and inference demo will be released upon publication. Stay tuned!

---

## 📝 Citation

```bibtex
@article{ge2025redimatch,
  title   = {REDI-Match: Rotation-Equivariant Distillation for Efficient and Robust Dense Matching},
  author  = {Ge, Yinji and Zheng, Guixu and Guo, Wulong and Feng, Qian and Wu, Xu and Zhou, Kai and Liu, Xinyuan and Xing, Fei},
  journal = {arXiv preprint},
  year    = {2025}
}
```
