
# ═══ Inlined small helpers ═══
import random as _random, numpy as _np
def prepare_run_seed(args):
    s = getattr(args, "seed", None)
    if s is None:
        return  # 不固定种子：使用系统随机
    s = int(s)
    _random.seed(s); _np.random.seed(s)
    import torch; torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def get_rng_seed_info(args=None):
    return {"seed": getattr(args, "seed", None) if args else None}

def ensure_results_dir():
    import os; os.makedirs("results", exist_ok=True)

# ═══ HPatches path helpers ═══

def hpatches_im_a_path(seq_dir):
    import os
    return os.path.join(seq_dir, "1.ppm")

def hpatches_im_b_path(seq_dir, im_idx, fixed_a_only_rotate_b=False):
    import os
    suffix = "_Brot" if fixed_a_only_rotate_b else ""
    return os.path.join(seq_dir, f"{im_idx}{suffix}.ppm")

def hpatches_H_path(seq_dir, im_idx):
    import os
    return os.path.join(seq_dir, f"H_1_{im_idx}")

def is_fixed_a_rot_release(seqs_path):
    import os
    return "_rot" in os.path.basename(seqs_path)

# ═══ C4 utilities ═══

import types
from typing import Any, Dict, List, Tuple

import torch

# ═══ C4 hybrid decoder helpers ═══

from typing import Any, Dict

import torch

_DECODER_C4_HYBRID_ATTRS = (
    "c4_hybrid",
    "c4_hybrid_alpha",
    "c4_hybrid_temp",
    "c4_hybrid_margin_keep",
    "c4_hybrid_disagree_margin",
    "c4_hybrid_ot_pool_size",
    "c4_hybrid_ot_epsilon",
    "c4_hybrid_ot_iters",
)

def _snapshot_decoder_c4_hybrid(decoder: Any) -> Dict[str, Any]:
    return {k: getattr(decoder, k, None) for k in _DECODER_C4_HYBRID_ATTRS}

def _restore_decoder_c4_hybrid(decoder: Any, snap: Dict[str, Any]) -> None:
    for k, v in snap.items():
        if hasattr(decoder, k):
            setattr(decoder, k, v)

def _apply_fh_kw_to_decoder(decoder: Any, fh_kw: Dict[str, Any]) -> None:
    key_map = {
        "hybrid_alpha": "c4_hybrid_alpha",
        "hybrid_temp": "c4_hybrid_temp",
        "margin_keep": "c4_hybrid_margin_keep",
        "disagree_margin": "c4_hybrid_disagree_margin",
        "ot_pool_size": "c4_hybrid_ot_pool_size",
        "ot_epsilon": "c4_hybrid_ot_epsilon",
        "ot_iters": "c4_hybrid_ot_iters",
    }
    for src, dst in key_map.items():
        if src in fh_kw and hasattr(decoder, dst):
            setattr(decoder, dst, fh_kw[src])
    decoder.c4_hybrid = True

def _sanitize_dense_for_multinomial(
    dense_matches: torch.Tensor, dense_certainty: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    dm = dense_matches.detach().clone()
    dc = torch.nan_to_num(
        dense_certainty.detach().clone(), nan=0.0, posinf=1.0, neginf=0.0
    ).clamp(0.0, 1.0)
    if float(dc.sum().item()) <= 0.0:
        dc = torch.full_like(dc, 1e-8)
    return dm, dc

# ═══ _rot.py ═══
import json
import math
import types
import os

import torch
import torch.nn.functional as F

from redimatch.benchmarks import (
    MegaDepthPoseEstimationBenchmark,
    MegaDepthPoseEstimationRotBenchmark,
)

def _module_dict_get(md, key):
    if key in md:
        return md[key]
    sk = str(key)
    if sk in md:
        return md[sk]
    if isinstance(key, str):
        try:
            ik = int(key)
            if ik in md:
                return md[ik]
        except ValueError:
            pass
    return None

def _first_module_dict_value(md):
    keys = list(md.keys())
    if not keys:
        return None
    return md[keys[0]]

def _gp_pool_tokens(f1_s, f2_s, decoder, gp_module, k, pool_size):
    """Consistent with demo/batch_c4_rotation_eval: permute + spatial unrotate + adaptive pool."""
    yk = decoder._c4_channel_permute(f2_s, k)
    if getattr(decoder, "c4_detect_use_spatial_unrotate", True) and k > 0:
        yk = torch.rot90(yk, k=(-k) % 4, dims=[-2, -1])
        if yk.shape[-2:] != f1_s.shape[-2:]:
            yk = F.interpolate(yk, size=f1_s.shape[-2:], mode="bilinear", align_corners=False)

    if pool_size > 0:
        sz = (pool_size, pool_size)
        x = F.adaptive_avg_pool2d(f1_s.float(), sz)
        y = F.adaptive_avg_pool2d(yk.float(), sz)
    else:
        x = f1_s.float()
        y = yk.float()

    x_tok = gp_module.reshape(x.float())
    y_tok = gp_module.reshape(y.float())
    k_xy = gp_module.K(x_tok, y_tok)
    return k_xy, x_tok, y_tok

def _gp_aggregate_token_metric(
    metric_bn: torch.Tensor,
    *,
    top_k_ratio: float = 0.20,
    disable_top20: bool = False,
) -> torch.Tensor:
    """demo: take mean of top-k%% tokens with smallest cost (B dim); smaller cost is better."""
    if disable_top20:
        return metric_bn.mean(dim=-1)
    n_tokens = metric_bn.shape[-1]
    k_tokens = max(1, int(n_tokens * float(top_k_ratio)))
    topk_val, _ = torch.topk(metric_bn, k=k_tokens, dim=-1, largest=False)
    return topk_val.mean(dim=-1)

def _gp_cycle_cost_for_k(
    decoder,
    gp_module,
    f1_s,
    f2_s,
    k,
    pool_size=12,
    *,
    prob_temp: float = 0.05,
    top_k_ratio: float = 0.20,
    disable_top20: bool = False,
):
    """
    Consistent with use_cycle_consistency branch in demo/batch_c4_rotation_eval.py:
    K_yx = K_xy^T；P_AB=softmax(K_xy/t,dim=-1)；P_BA=softmax(K_yx/t,dim=-1)；
    P_cycle=P_AB@P_BA; metric=-log(diag); top-k mean as cost (smaller is better).
    """
    k_xy, _, _ = _gp_pool_tokens(f1_s, f2_s, decoder, gp_module, k, pool_size)
    k_yx = k_xy.transpose(-1, -2)
    temp = max(float(prob_temp), 1e-6)
    p_ab = F.softmax(k_xy / temp, dim=-1)
    p_ba = F.softmax(k_yx / temp, dim=-1)
    p_cycle = torch.matmul(p_ab, p_ba)
    cycle_diag = torch.diagonal(p_cycle, dim1=-2, dim2=-1)
    score_metric = -torch.log(cycle_diag + 1e-8)
    return _gp_aggregate_token_metric(
        score_metric, top_k_ratio=top_k_ratio, disable_top20=disable_top20
    )

def _gp_entropy_score_for_k(
    decoder,
    gp_module,
    f1_s,
    f2_s,
    k,
    pool_size=12,
):
    """FH original gp_intrinsic entropy base (unchanged when cycle is off)."""
    k_xy, _, _ = _gp_pool_tokens(f1_s, f2_s, decoder, gp_module, k, pool_size)
    n = k_xy.shape[-1]
    norm = max(math.log(max(n, 2)), 1e-6)

    prob = k_xy / (k_xy.sum(dim=-1, keepdim=True) + 1e-8)
    entropy = -(prob * torch.log(prob + 1e-8)).sum(dim=-1)
    entropy_norm = entropy.mean(dim=-1) / norm
    score = -entropy_norm
    return score, entropy_norm

def _install_gp_intrinsic_rotation_router(
    model,
    gp_entropy_pool_size=12,
    gp_entropy_temp=0.05,
    *,
    use_cycle_consistency: bool = False,
    gp_prob_temp: float = 0.05,
    top_k_ratio: float = 0.20,
    disable_top20: bool = False,
):
    decoder = getattr(model, "decoder", None)
    if decoder is None or not hasattr(decoder, "_estimate_c4_rotation"):
        return False, None
    if not hasattr(decoder, "gps") or not hasattr(decoder, "_c4_channel_permute"):
        return False, None

    orig_fn = decoder._estimate_c4_rotation
    decoder._gp_intrinsic_use_cycle_consistency = bool(use_cycle_consistency)
    decoder._gp_intrinsic_prob_temp = float(gp_prob_temp)
    decoder._gp_intrinsic_top_k_ratio = float(top_k_ratio)
    decoder._gp_intrinsic_disable_top20 = bool(disable_top20)

    def _gp_intrinsic_estimate(self, f1_s, f2_s):
        if (not self._is_c4_tensor(f1_s)) or (not self._is_c4_tensor(f2_s)):
            return orig_fn(f1_s, f2_s)
        if f1_s.shape[1] != f2_s.shape[1]:
            return orig_fn(f1_s, f2_s)

        target_scale = getattr(self, "c4_est_scale", None)
        gp_module = _module_dict_get(self.gps, target_scale) if target_scale is not None else None
        if gp_module is None:
            gp_module = _first_module_dict_value(self.gps)
        if gp_module is None or (not hasattr(gp_module, "K")) or (not hasattr(gp_module, "reshape")):
            return orig_fn(f1_s, f2_s)

        try:
            scores = []
            aux = []
            for k in range(4):
                if use_cycle_consistency:
                    cost = _gp_cycle_cost_for_k(
                        self,
                        gp_module,
                        f1_s,
                        f2_s,
                        k=k,
                        pool_size=gp_entropy_pool_size,
                        prob_temp=gp_prob_temp,
                        top_k_ratio=top_k_ratio,
                        disable_top20=disable_top20,
                    )
                    # Same as entropy branch: 'larger is better'; demo uses min cost → score = -cost
                    n = max(gp_entropy_pool_size, 2) ** 2 if gp_entropy_pool_size > 0 else f1_s.shape[-2] * f1_s.shape[-1]
                    norm = max(math.log(max(n, 2)), 1e-6)
                    s_k = -cost / norm
                    aux.append(cost)
                else:
                    s_k, h_k = _gp_entropy_score_for_k(
                        self,
                        gp_module,
                        f1_s,
                        f2_s,
                        k=k,
                        pool_size=gp_entropy_pool_size,
                    )
                    aux.append(h_k)
                scores.append(s_k)

            scores = torch.stack(scores, dim=1)
            metric_by_k = torch.stack(aux, dim=1)
            temp = max(gp_entropy_temp, 1e-6)
            prob = torch.softmax(scores / temp, dim=1)
            idx = scores.argmax(dim=1)
            top2 = torch.topk(scores, k=2, dim=1).values
            margin = top2[:, 0] - top2[:, 1]
            return {
                "rot_idx": idx,
                "rot_prob": prob,
                "rot_margin": margin,
                "rot_scores": scores,
                "rot_entropy": metric_by_k,
                "source": "gp_intrinsic_cycle" if use_cycle_consistency else "gp_intrinsic",
                "gp_use_cycle_consistency": bool(use_cycle_consistency),
                "gp_prob_temp": float(gp_prob_temp),
                "gp_top_k_ratio": float(top_k_ratio),
            }
        except Exception:
            return orig_fn(f1_s, f2_s)

    decoder._estimate_c4_rotation = types.MethodType(_gp_intrinsic_estimate, decoder)
    return True, orig_fn

def gp_intrinsic_router_kwargs_from_args(model, args=None) -> dict:
    """Fetch FH gp_intrinsic hyperparameters from model cache or argparse."""
    a = args if args is not None else getattr(model, "_c4_benchmark_args", None)
    return dict(
        gp_entropy_pool_size=int(getattr(a, "gp_entropy_pool_size", 12)),
        gp_entropy_temp=float(getattr(a, "gp_entropy_temp", 0.05)),
        use_cycle_consistency=bool(
            getattr(model, "_use_cycle_consistency", getattr(a, "use_cycle_consistency", False))
        ),
        gp_prob_temp=float(getattr(a, "gp_prob_temp", 0.05)),
        top_k_ratio=float(getattr(a, "c4_top_k_ratio", 0.20)),
        disable_top20=bool(getattr(a, "c4_disable_top20_filter", False)),
    )

def install_gp_intrinsic_rotation_router(
    model,
    gp_entropy_pool_size=12,
    gp_entropy_temp=0.05,
    *,
    use_cycle_consistency: bool = False,
    gp_prob_temp: float = 0.05,
    top_k_ratio: float = 0.20,
    disable_top20: bool = False,
    force: bool = False,
):
    """Expose for non-MegaDepth benchmark scripts: switch decoder's C4 rotation estimation to gp_intrinsic."""
    decoder = getattr(model, "decoder", None)
    if (
        not force
        and getattr(model, "_gp_intrinsic_router_installed", False)
        and decoder is not None
        and hasattr(decoder, "_estimate_c4_rotation")
    ):
        return True, getattr(model, "_c4_orig_estimate_fn", None)
    saved_orig = getattr(model, "_c4_orig_estimate_fn", None)
    if force and saved_orig is not None and decoder is not None:
        decoder._estimate_c4_rotation = saved_orig
    ok, orig_fn = _install_gp_intrinsic_rotation_router(
        model,
        gp_entropy_pool_size=gp_entropy_pool_size,
        gp_entropy_temp=gp_entropy_temp,
        use_cycle_consistency=use_cycle_consistency,
        gp_prob_temp=gp_prob_temp,
        top_k_ratio=top_k_ratio,
        disable_top20=disable_top20,
    )
    if ok:
        model._gp_intrinsic_router_installed = True
        if getattr(model, "_c4_orig_estimate_fn", None) is None:
            model._c4_orig_estimate_fn = orig_fn
    return ok, orig_fn

def eval_mega1500(
    model,
    name,
    data_root="data/megadepth",
    rot=False,
    fh_hybrid_alpha=0.35,
    fh_hybrid_temp=0.05,
    fh_margin_keep=0.03,
    fh_hybrid_disagree_margin=0.03,
    fh_ot_pool_size=12,
    fh_ot_epsilon=0.07,
    fh_ot_iters=10,
    gp_entropy_pool_size=12,
    gp_entropy_temp=0.05,
):
    """Mega1500 eval with fundamental_hybrid C4 + GP intrinsic rotation router."""
    bench_cls = MegaDepthPoseEstimationRotBenchmark if rot else MegaDepthPoseEstimationBenchmark
    mega1500_benchmark = bench_cls(data_root)

    # Apply fundamental_hybrid kwargs to decoder
    dec = getattr(model, "decoder", None)
    if dec is not None:
        dec.c4_hybrid_alpha = fh_hybrid_alpha
        dec.c4_hybrid_temp = fh_hybrid_temp
        dec.c4_hybrid_margin_keep = fh_margin_keep
        dec.c4_hybrid_disagree_margin = fh_hybrid_disagree_margin
        dec.c4_hybrid_ot_pool_size = fh_ot_pool_size
        dec.c4_hybrid_ot_epsilon = fh_ot_epsilon
        dec.c4_hybrid_ot_iters = fh_ot_iters
        dec.c4_hybrid = True

    # Install GP intrinsic rotation router
    ok, orig_fn = _install_gp_intrinsic_rotation_router(
        model,
        gp_entropy_pool_size=gp_entropy_pool_size,
        gp_entropy_temp=gp_entropy_temp,
    )
    print(f"[C4] gp_intrinsic rotation router {'enabled' if ok else 'not available, fallback to default'}")
    restore_estimator = orig_fn if ok else None
    try:
        mega1500_results = mega1500_benchmark.benchmark(model, model_name=name)
    finally:
        if restore_estimator is not None and hasattr(model.decoder, "_estimate_c4_rotation"):
            model.decoder._estimate_c4_rotation = restore_estimator

    print(f"[MegaDepth] {name}")
    print(mega1500_results)
    return mega1500_results

def _coarse_res_to_model_resolution(coarse_res: int) -> str:
    mapping = {448: "low", 560: "medium", 576: "medium", 672: "high"}
    return mapping.get(int(coarse_res), "medium")


def _extract_model_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must contain a state-dict mapping")
    return {
        key[7:] if isinstance(key, str) and key.startswith("module.") else key: value
        for key, value in checkpoint.items()
    }


def _build_eval_model(args, device, weights_ckpt, *, symmetric: bool, upsample_preds: bool):
    from redimatch.models.model_builder import get_model, install_gp16_cholesky_solver

    model = get_model(
        pretrained_backbone=True,
        resolution=_coarse_res_to_model_resolution(args.coarse_res),
        model_type="gp",
        freeze_encoder=getattr(args, "freeze_encoder", True),
        exported_weights_path=None,
        symmetric=symmetric,
        upsample_preds=upsample_preds,
        upsample_res=(
            getattr(args, "upsample_res", 800),
            getattr(args, "upsample_res", 800),
        ),
        attenuate_cert=False,
        use_custom_corr=not getattr(args, "no_custom_corr", False),
    ).to(device)
    install_gp16_cholesky_solver(model)
    missing, unexpected = model.load_state_dict(
        _extract_model_state_dict(weights_ckpt), strict=False
    )
    if missing or unexpected:
        print(
            f"[WARN] load_state_dict missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )
    return model


def _apply_fast_cuda_runtime(*, enable: bool) -> None:
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = bool(enable)
    torch.backends.cudnn.allow_tf32 = bool(enable)
    torch.backends.cudnn.benchmark = bool(enable)
    try:
        torch.set_float32_matmul_precision("high" if enable else "highest")
    except Exception:
        pass
