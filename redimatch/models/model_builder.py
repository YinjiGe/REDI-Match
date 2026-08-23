"""Model construction for inference / evaluation.

This module contains the model construction path used by the public demo and
evaluation scripts.
"""

import os
import sys
import warnings
from types import MethodType

import torch
import torch.nn as nn

from redimatch.models.matcher import (
    ConvRefiner,
    CosKernel,
    Decoder,
    GP,
    RegressionMatcher,
)
from redimatch.models.transformer import (
    Block,
    MemEffAttention,
    TransformerDecoder,
)
from redimatch.models.inference_encoder import E2CNNEncoderExportedMaxPool

resolutions = {"low": (448, 448), "medium": (14 * 8 * 5, 14 * 8 * 5), "high": (14 * 8 * 6, 14 * 8 * 6)}


def _local_corr_extension_available() -> bool:
    try:
        import local_corr  # noqa: F401
        return True
    except ImportError:
        return False


def install_gp16_cholesky_solver(model: nn.Module) -> bool:
    """Use Cholesky for the s16 GP posterior while keeping checkpoint format unchanged."""
    decoder = getattr(model, "decoder", None)
    gps = getattr(decoder, "gps", None)
    if gps is None or "16" not in gps:
        return False

    gp = gps["16"]

    def _forward_cholesky(self, x, y, **kwargs):
        b, _c, h1, w1 = x.shape
        _b2, _c2, h2, w2 = y.shape
        f = self.get_pos_enc(y)
        self._cached_pos_enc_B = f
        x_flat = self.reshape(x.float())
        y_flat = self.reshape(y.float())
        f_flat = self.reshape(f.float())
        k_yy = self.K(y_flat, y_flat)
        k_xy = self.K(x_flat, y_flat)
        n = h2 * w2
        reg = (self.sigma_noise + self.jitter) * torch.eye(
            n, device=x.device, dtype=x_flat.dtype
        )[None, :, :]
        a = k_yy + reg
        chol, _ = torch.linalg.cholesky_ex(a)
        z = torch.cholesky_solve(f_flat, chol)
        mu_x = k_xy @ z
        return mu_x.reshape(b, h1, w1, -1).permute(0, 3, 1, 2).contiguous()

    gp.forward = MethodType(_forward_cholesky, gp)
    gp._solver_name = "cholesky"
    return True


def get_model(pretrained_backbone=True, resolution="medium", **kwargs):
    """Build the GP matcher (s16 GP + TransformerDecoder + multi-scale ConvRefiner)."""
    warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

    # Kept for backward compatibility with old call sites; only the "gp"
    # architecture is implemented in this package.
    kwargs.pop("model_type", "gp")

    # ========== GP architecture (s16 coarse decoder) ==========
    gp_dim = 512
    feat_dim = 512
    decoder_dim = gp_dim + feat_dim
    cls_to_coord_res = 64
    coordinate_decoder = TransformerDecoder(
        nn.Sequential(*[Block(decoder_dim, 8, attn_class=MemEffAttention) for _ in range(5)]),
        decoder_dim,
        cls_to_coord_res**2 + 1,
        is_classifier=True,
        amp=True,
        pos_enc=False,
    )
    kernel_temperature = 0.2
    gps = nn.ModuleDict({
        "16": GP(
            CosKernel, T=kernel_temperature, learn_temperature=False,
            only_attention=False, gp_dim=gp_dim, basis="fourier", no_cov=True,
        )
    })
    decoder_scales = ["16", "8", "4", "2", "1"]

    dw = True
    hidden_blocks = 8
    kernel_size = 5
    displacement_emb = "linear"
    disable_local_corr_grad = True

    # local_corr CUDA extension: computation path equivalent to pure PyTorch
    # grid_sample; accelerates local correlation inside ConvRefiner when available.
    use_custom_corr = kwargs.pop("use_custom_corr", None)
    if use_custom_corr is None:
        use_custom_corr = sys.platform == "linux" and _local_corr_extension_available()
    elif use_custom_corr and sys.platform != "linux":
        use_custom_corr = False
    elif use_custom_corr and not _local_corr_extension_available():
        use_custom_corr = False

    # ========== ConvRefiner ==========
    conv_refiner_dict = {}
    conv_refiner_dict["16"] = ConvRefiner(
        2 * 512 + 128 + (2 * 7 + 1) ** 2,
        2 * 512 + 128 + (2 * 7 + 1) ** 2,
        2 + 1,
        kernel_size=kernel_size,
        dw=dw,
        hidden_blocks=hidden_blocks,
        displacement_emb=displacement_emb,
        displacement_emb_dim=128,
        local_corr_radius=7,
        corr_in_other=True,
        amp=True,
        disable_local_corr_grad=disable_local_corr_grad,
        bn_momentum=0.01,
        use_custom_corr=use_custom_corr,
    )
    conv_refiner_dict.update({
        "8": ConvRefiner(
            2 * 512 + 64 + (2 * 3 + 1) ** 2,
            2 * 512 + 64 + (2 * 3 + 1) ** 2,
            2 + 1,
            kernel_size=kernel_size,
            dw=dw,
            hidden_blocks=hidden_blocks,
            displacement_emb=displacement_emb,
            displacement_emb_dim=64,
            local_corr_radius=3,
            corr_in_other=True,
            amp=True,
            disable_local_corr_grad=disable_local_corr_grad,
            bn_momentum=0.01,
            use_custom_corr=use_custom_corr,
        ),
        "4": ConvRefiner(
            2 * 256 + 32 + (2 * 2 + 1) ** 2,
            2 * 256 + 32 + (2 * 2 + 1) ** 2,
            2 + 1,
            kernel_size=kernel_size,
            dw=dw,
            hidden_blocks=hidden_blocks,
            displacement_emb=displacement_emb,
            displacement_emb_dim=32,
            local_corr_radius=2,
            corr_in_other=True,
            amp=True,
            disable_local_corr_grad=disable_local_corr_grad,
            bn_momentum=0.01,
            use_custom_corr=use_custom_corr,
        ),
        "2": ConvRefiner(
            2 * 64 + 16,
            128 + 16,
            2 + 1,
            kernel_size=kernel_size,
            dw=dw,
            hidden_blocks=hidden_blocks,
            displacement_emb=displacement_emb,
            displacement_emb_dim=16,
            amp=True,
            disable_local_corr_grad=disable_local_corr_grad,
            bn_momentum=0.01,
            use_custom_corr=use_custom_corr,
        ),
        "1": ConvRefiner(
            2 * 9 + 6,
            24,
            2 + 1,
            kernel_size=kernel_size,
            dw=dw,
            hidden_blocks=hidden_blocks,
            displacement_emb=displacement_emb,
            displacement_emb_dim=6,
            amp=True,
            disable_local_corr_grad=disable_local_corr_grad,
            bn_momentum=0.01,
            use_custom_corr=use_custom_corr,
        ),
    })
    conv_refiner = nn.ModuleDict(conv_refiner_dict)

    # ========== Projection (adapt MaxPool encoder channels to decoder targets) ==========
    proj = nn.ModuleDict({
        "8": nn.Sequential(nn.Conv2d(384, 512, 1, 1), nn.BatchNorm2d(512)),
        "4": nn.Sequential(nn.Conv2d(256, 256, 1, 1), nn.BatchNorm2d(256)),
        "2": nn.Sequential(nn.Conv2d(128, 64, 1, 1), nn.BatchNorm2d(64)),
        "1": nn.Sequential(nn.Conv2d(64, 9, 1, 1), nn.BatchNorm2d(9)),
        "16": nn.Sequential(nn.Conv2d(512, 512, 1, 1), nn.BatchNorm2d(512)),
    })
    displacement_dropout_p = 0.0
    gm_warp_dropout_p = 0.0
    decoder = Decoder(
        coordinate_decoder,
        gps,
        proj,
        conv_refiner,
        detach=True,
        scales=decoder_scales,
        displacement_dropout_p=displacement_dropout_p,
        gm_warp_dropout_p=gm_warp_dropout_p,
    )

    h, w = resolutions[resolution]

    # ========== Encoder: MaxPool only, weights from exported_weights_path ==========
    exported_weights_path = kwargs.pop("exported_weights_path", None)
    for k in ("use_maxpool_encoder", "use_exported_encoder", "maxpool_weights_path", "original_weights_path"):
        kwargs.pop(k, None)

    freeze_encoder = kwargs.pop("freeze_encoder", True)
    base_encoder = E2CNNEncoderExportedMaxPool(
        weights_path=exported_weights_path,
        freeze=freeze_encoder,
        amp=True,
    )
    encoder = base_encoder

    matcher = RegressionMatcher(encoder, decoder, h=h, w=w, **kwargs)
    return matcher
