from typing import Optional

import os
import math
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from warnings import warn
from PIL import Image

from redimatch.utils import get_tuple_transform_ops
from redimatch.utils.local_correlation import local_correlation
from redimatch.utils.utils import (
    check_rgb,
    cls_to_flow_refine,
    get_autocast_params,
    check_not_i16,
)
from redimatch.utils.kde import kde
from redimatch.models.inference_encoder import E2CNNEncoderExportedMaxPool


class ConvRefiner(nn.Module):
    def __init__(
        self,
        in_dim=6,
        hidden_dim=16,
        out_dim=2,
        dw=False,
        kernel_size=5,
        hidden_blocks=3,
        displacement_emb=None,
        displacement_emb_dim=None,
        local_corr_radius=None,
        corr_in_other=None,
        no_im_B_fm=False,
        amp=False,
        concat_logits=False,
        use_bias_block_1=True,
        use_cosine_corr=False,
        disable_local_corr_grad=False,
        is_classifier=False,
        sample_mode="bilinear",
        norm_type=nn.BatchNorm2d,
        bn_momentum=0.1,
        amp_dtype=torch.float16,
        use_custom_corr=False,
        tile_size=None,
    ):
        super().__init__()
        self.tile_size = tile_size  # Tile size for tiled inference, None means no tiling (used to reduce memory for large images)
        if sys.platform != "linux":
            warn("Local correlation is not supported on non-Linux platforms, setting use_custom_corr to False")
            use_custom_corr = False
        self.bn_momentum = bn_momentum
        self.block1 = self.create_block(
            in_dim,
            hidden_dim,
            dw=dw,
            kernel_size=kernel_size,
            bias=use_bias_block_1,
        )
        self.hidden_blocks = nn.Sequential(
            *[
                self.create_block(
                    hidden_dim,
                    hidden_dim,
                    dw=dw,
                    kernel_size=kernel_size,
                    norm_type=norm_type,
                )
                for hb in range(hidden_blocks)
            ]
        )
        self.hidden_blocks = self.hidden_blocks
        self.out_conv = nn.Conv2d(hidden_dim, out_dim, 1, 1, 0)
        if displacement_emb:
            self.has_displacement_emb = True
            self.disp_emb = nn.Conv2d(2, displacement_emb_dim, 1, 1, 0)
        else:
            self.has_displacement_emb = False
        self.local_corr_radius = local_corr_radius
        self.corr_in_other = corr_in_other
        self.no_im_B_fm = no_im_B_fm
        self.amp = amp
        self.concat_logits = concat_logits
        self.use_cosine_corr = use_cosine_corr
        self.disable_local_corr_grad = disable_local_corr_grad
        self.is_classifier = is_classifier
        self.sample_mode = sample_mode
        self.amp_dtype = amp_dtype
        self.use_custom_corr = use_custom_corr

    def create_block(
        self,
        in_dim,
        out_dim,
        dw=False,
        kernel_size=5,
        bias=True,
        norm_type=nn.BatchNorm2d,
    ):
        num_groups = 1 if not dw else in_dim
        if dw:
            assert out_dim % in_dim == 0, (
                "outdim must be divisible by indim for depthwise"
            )
        conv1 = nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=num_groups,
            bias=bias,
        )
        norm = (
            norm_type(out_dim, momentum=self.bn_momentum)
            if norm_type is nn.BatchNorm2d
            else norm_type(num_channels=out_dim)
        )
        relu = nn.ReLU(inplace=True)
        conv2 = nn.Conv2d(out_dim, out_dim, 1, 1, 0)
        return nn.Sequential(conv1, norm, relu, conv2)

    def _forward_core(self, x, y, warp, scale_factor=1, logits=None):
        """Single forward pass (full block or single tile), shared by forward and tiled."""
        b, c, hs, ws = x.shape
        autocast_device, autocast_enabled, autocast_dtype = get_autocast_params(
            x.device, enabled=self.amp, dtype=self.amp_dtype
        )
        with torch.autocast(
            autocast_device, enabled=autocast_enabled, dtype=autocast_dtype
        ):
            x_hat = F.grid_sample(
                y, warp.permute(0, 2, 3, 1), align_corners=False, mode=self.sample_mode
            )
            if self.has_displacement_emb:
                im_A_coords = torch.meshgrid(
                    (
                        torch.linspace(-1 + 1 / hs, 1 - 1 / hs, hs, device=x.device),
                        torch.linspace(-1 + 1 / ws, 1 - 1 / ws, ws, device=x.device),
                    ),
                    indexing="ij",
                )
                im_A_coords = torch.stack((im_A_coords[1], im_A_coords[0]))
                im_A_coords = im_A_coords[None].expand(b, 2, hs, ws)
                in_displacement = warp - im_A_coords
                emb_in_displacement = self.disp_emb(
                    40 / 32 * scale_factor * in_displacement
                )
                if self.local_corr_radius:
                    if self.corr_in_other:
                        # Corr in other means take a kxk grid around the predicted coordinate in other image
                        local_corr = local_correlation(
                            x,
                            y,
                            self.local_corr_radius,
                            warp,
                            sample_mode=self.sample_mode,
                            use_custom_corr=self.use_custom_corr,
                        )
                    else:
                        raise NotImplementedError(
                            "Local corr in own frame should not be used."
                        )
                    if self.no_im_B_fm:
                        x_hat = torch.zeros_like(x)
                    d = torch.cat((x, x_hat, emb_in_displacement, local_corr), dim=1)
                else:
                    d = torch.cat((x, x_hat, emb_in_displacement), dim=1)
            else:
                if self.no_im_B_fm:
                    x_hat = torch.zeros_like(x)
                d = torch.cat((x, x_hat), dim=1)
            if self.concat_logits and logits is not None:
                d = torch.cat((d, logits), dim=1)
            d = self.block1(d)
            d = self.hidden_blocks(d)
        d = self.out_conv(d.float())
        displacement, certainty = d[:, :-1], d[:, -1:]
        return displacement, certainty

    def forward(self, x, y, warp, scale_factor=1, logits=None):
        b, c, hs, ws = x.shape
        tile_size = getattr(self, 'tile_size', None)
        if tile_size is None or (hs <= tile_size and ws <= tile_size):
            return self._forward_core(x, y, warp, scale_factor=scale_factor, logits=logits)
        # Tiled inference: split spatial domain by tile_size, forward block by block then stitch, reducing large-image memory and peak usage
        ts = tile_size
        n_h = (hs + ts - 1) // ts
        n_w = (ws + ts - 1) // ts
        device, dtype = x.device, x.dtype
        displacement = torch.zeros(b, 2, hs, ws, device=device, dtype=torch.float32)
        certainty = torch.zeros(b, 1, hs, ws, device=device, dtype=torch.float32)
        for i in range(n_h):
            for j in range(n_w):
                h0, h1 = i * ts, min((i + 1) * ts, hs)
                w0, w1 = j * ts, min((j + 1) * ts, ws)
                x_t = x[:, :, h0:h1, w0:w1].contiguous()
                y_t = y[:, :, h0:h1, w0:w1].contiguous()
                warp_t = warp[:, :, h0:h1, w0:w1].contiguous()
                logits_t = logits[:, :, h0:h1, w0:w1].contiguous() if logits is not None else None
                disp_t, cert_t = self._forward_core(
                    x_t, y_t, warp_t, scale_factor=scale_factor, logits=logits_t
                )
                displacement[:, :, h0:h1, w0:w1] = disp_t.float()
                certainty[:, :, h0:h1, w0:w1] = cert_t.float()
        return displacement, certainty


class CosKernel(nn.Module):  # similar to softmax kernel
    def __init__(self, T, learn_temperature=False):
        super().__init__()
        self.learn_temperature = learn_temperature
        if self.learn_temperature:
            self.T = nn.Parameter(torch.tensor(T))
        else:
            self.T = T

    def __call__(self, x, y, eps=1e-6):
        c = torch.einsum("bnd,bmd->bnm", x, y) / (
            x.norm(dim=-1)[..., None] * y.norm(dim=-1)[:, None] + eps
        )
        if self.learn_temperature:
            T = self.T.abs() + 0.01
        else:
            T = torch.tensor(self.T, device=c.device)
        K = ((c - 1.0) / T).exp()
        return K


class GP(nn.Module):
    def __init__(
        self,
        kernel,
        T=1,
        learn_temperature=False,
        only_attention=False,
        gp_dim=64,
        basis="fourier",
        covar_size=5,
        only_nearest_neighbour=False,
        sigma_noise=0.1,
        no_cov=False,
        predict_features=False,
        jitter=1e-4,
    ):
        super().__init__()
        self.K = kernel(T=T, learn_temperature=learn_temperature)
        self.sigma_noise = sigma_noise
        # Numerically stable: regularize diagonal with jitter to avoid inv/cholesky failure on ill-conditioned K_yy
        self.jitter = jitter
        self.covar_size = covar_size
        self.pos_conv = torch.nn.Conv2d(2, gp_dim, 1, 1)
        self.only_attention = only_attention
        self.only_nearest_neighbour = only_nearest_neighbour
        self.basis = basis
        self.no_cov = no_cov
        self.dim = gp_dim
        self.predict_features = predict_features

    def get_local_cov(self, cov):
        K = self.covar_size
        b, h, w, h, w = cov.shape
        hw = h * w
        cov = F.pad(cov, 4 * (K // 2,))  # pad v_q
        delta = torch.stack(
            torch.meshgrid(
                torch.arange(-(K // 2), K // 2 + 1),
                torch.arange(-(K // 2), K // 2 + 1),
                indexing="ij",
            ),
            dim=-1,
        )
        positions = torch.stack(
            torch.meshgrid(
                torch.arange(K // 2, h + K // 2),
                torch.arange(K // 2, w + K // 2),
                indexing="ij",
            ),
            dim=-1,
        )
        neighbours = positions[:, :, None, None, :] + delta[None, :, :]
        points = torch.arange(hw)[:, None].expand(hw, K**2)
        local_cov = cov.reshape(b, hw, h + K - 1, w + K - 1)[
            :,
            points.flatten(),
            neighbours[..., 0].flatten(),
            neighbours[..., 1].flatten(),
        ].reshape(b, h, w, K**2)
        return local_cov

    def reshape(self, x):
        return rearrange(x, "b d h w -> b (h w) d")

    def project_to_basis(self, x):
        if self.basis == "fourier":
            return torch.cos(8 * math.pi * self.pos_conv(x))
        elif self.basis == "linear":
            return self.pos_conv(x)
        else:
            raise ValueError(
                "No other bases other than fourier and linear currently im_Bed in public release"
            )

    def get_pos_enc(self, y):
        b, c, h, w = y.shape
        coarse_coords = torch.meshgrid(
            (
                torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device=y.device),
                torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device=y.device),
            ),
            indexing="ij",
        )

        coarse_coords = torch.stack((coarse_coords[1], coarse_coords[0]), dim=-1)[
            None
        ].expand(b, h, w, 2)
        coarse_coords = rearrange(coarse_coords, "b h w d -> b d h w")
        coarse_embedded_coords = self.project_to_basis(coarse_coords)
        return coarse_embedded_coords

    def forward(self, x, y, **kwargs):
        b, c, h1, w1 = x.shape
        b, c, h2, w2 = y.shape
        f = self.get_pos_enc(y)
        # Cache B positional encoding for the inference-time Cholesky solver
        self._cached_pos_enc_B = f
        b, d, h2, w2 = f.shape
        x, y, f = self.reshape(x.float()), self.reshape(y.float()), self.reshape(f.float())
        # K_xx = self.K(x, x)
        K_yy = self.K(y, y)
        K_xy = self.K(x, y)
        K_yx = K_xy.permute(0, 2, 1)
        n = h2 * w2
        reg = (self.sigma_noise + self.jitter) * torch.eye(n, device=x.device, dtype=x.dtype)[None, :, :]
        A = K_yy + reg
        f_flat = f.reshape(b, n, -1)
        # use solve instead of inv: solve A @ z = f, numerically more stable, mathematically equivalent to z = A^{-1} @ f
        z = torch.linalg.solve(A, f_flat)
        mu_x = K_xy @ z
        mu_x = rearrange(mu_x, "b (h w) d -> b d h w", h=h1, w=w1)


        # if not self.no_cov:
        #     cov_x = K_xx - K_xy.matmul(K_yy_inv.matmul(K_yx))
        #     cov_x = rearrange(
        #         cov_x, "b (h w) (r c) -> b h w r c", h=h1, w=w1, r=h1, c=w1
        #     )
        #     local_cov_x = self.get_local_cov(cov_x)
        #     local_cov_x = rearrange(local_cov_x, "b h w K -> b K h w")
        #     gp_feats = torch.cat((mu_x, local_cov_x), dim=1)
        # else:
        gp_feats = mu_x
        return gp_feats


class Decoder(nn.Module):
    def __init__(
        self,
        embedding_decoder,
        gps,
        proj,
        conv_refiner,
        detach=False,
        scales="all",
        pos_embeddings=None,
        num_refinement_steps_per_scale=1,
        warp_noise_std=0.0,
        displacement_dropout_p=0.0,
        gm_warp_dropout_p=0.0,
        flow_upsample_mode="bilinear",
        amp_dtype=torch.float16,
        c4_rotation_matching=False,
        c4_est_scale=None,
        c4_est_spatial_size=16,
        c4_score_temperature=0.05,
        c4_perm_reverse=False,
        c4_detect_use_spatial_unrotate=True,
        c4_spatial_canonicalize=True,
        c4_rotate_flow_back=True,
        c4_hybrid=False,
        c4_hybrid_alpha=0.35,
        c4_hybrid_temp=0.05,
        c4_hybrid_margin_keep=0.03,
        c4_hybrid_disagree_margin=0.03,
        c4_hybrid_ot_pool_size=12,
        c4_hybrid_ot_epsilon=0.07,
        c4_hybrid_ot_iters=10,
        c4_force_rot_idx=None,
        c4_reuse_coarse_rot_on_upsample=True,
        c4_symmetric_reverse="align_to_query",
        cycle_cert_filter=False,
        cycle_cert_filter_tau=0.05,
    ):
        super().__init__()
        self.embedding_decoder = embedding_decoder
        self.num_refinement_steps_per_scale = num_refinement_steps_per_scale
        self.gps = gps
        self.proj = proj
        self.conv_refiner = conv_refiner
        self.detach = detach
        if pos_embeddings is None:
            self.pos_embeddings = {}
        else:
            self.pos_embeddings = pos_embeddings
        if scales == "all":
            self.scales = ["32", "16", "8", "4", "2", "1"]
        else:
            self.scales = scales
        self.warp_noise_std = warp_noise_std
        self.refine_init = 4
        self.displacement_dropout_p = displacement_dropout_p
        self.gm_warp_dropout_p = gm_warp_dropout_p
        self.flow_upsample_mode = flow_upsample_mode
        self.amp_dtype = amp_dtype
        self.c4_rotation_matching = c4_rotation_matching
        self.c4_est_scale = int(c4_est_scale) if c4_est_scale is not None else None
        self.c4_est_spatial_size = int(c4_est_spatial_size)
        self.c4_score_temperature = float(c4_score_temperature)
        self.c4_perm_reverse = c4_perm_reverse
        self.c4_detect_use_spatial_unrotate = c4_detect_use_spatial_unrotate
        self.c4_spatial_canonicalize = c4_spatial_canonicalize
        self.c4_rotate_flow_back = c4_rotate_flow_back
        self.c4_hybrid = bool(c4_hybrid)
        self.c4_hybrid_alpha = float(c4_hybrid_alpha)
        self.c4_hybrid_temp = float(c4_hybrid_temp)
        self.c4_hybrid_margin_keep = float(c4_hybrid_margin_keep)
        self.c4_hybrid_disagree_margin = float(c4_hybrid_disagree_margin)
        self.c4_hybrid_ot_pool_size = int(c4_hybrid_ot_pool_size)
        self.c4_hybrid_ot_epsilon = float(c4_hybrid_ot_epsilon)
        self.c4_hybrid_ot_iters = int(c4_hybrid_ot_iters)
        self.c4_force_rot_idx = (
            int(c4_force_rot_idx) % 4 if c4_force_rot_idx is not None else None
        )
        self.c4_reuse_coarse_rot_on_upsample = bool(c4_reuse_coarse_rot_on_upsample)
        self.c4_symmetric_reverse = str(c4_symmetric_reverse)  # "align_to_query" | "identity" | "original"
        self.cycle_cert_filter = bool(cycle_cert_filter)
        self.cycle_cert_filter_tau = float(cycle_cert_filter_tau)
        self._last_c4_rotation = None

    def get_placeholder_flow(self, b, h, w, device):
        coarse_coords = torch.meshgrid(
            (
                torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device=device),
                torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device=device),
            ),
            indexing="ij",
        )
        coarse_coords = torch.stack((coarse_coords[1], coarse_coords[0]), dim=-1)[
            None
        ].expand(b, h, w, 2)
        coarse_coords = rearrange(coarse_coords, "b h w d -> b d h w")
        return coarse_coords

    def get_positional_embedding(self, b, h, w, device):
        coarse_coords = torch.meshgrid(
            (
                torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device=device),
                torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device=device),
            ),
            indexing="ij",
        )

        coarse_coords = torch.stack((coarse_coords[1], coarse_coords[0]), dim=-1)[
            None
        ].expand(b, h, w, 2)
        coarse_coords = rearrange(coarse_coords, "b h w d -> b d h w")
        coarse_embedded_coords = self.pos_embedding(coarse_coords)
        return coarse_embedded_coords

    def _is_c4_tensor(self, feat: torch.Tensor) -> bool:
        return feat.dim() == 4 and feat.shape[1] >= 4 and feat.shape[1] % 4 == 0

    def _c4_forward_k_from_image_rotation(self, k) -> int | torch.Tensor:
        """Plain feature forward: enc(rot(I)) ≈ c4(enc(I), k_fwd), k_fwd = (-k_image)%4."""
        if isinstance(k, torch.Tensor):
            return ((-k.long()) % 4).to(dtype=torch.long, device=k.device)
        return (-int(k)) % 4

    def _c4_align_rot_idx_from_image_k(self, k) -> int | torch.Tensor:
        """Align f2 (rotated B) back to plain B system: use rotation_gt k itself (inverse transform)."""
        if isinstance(k, torch.Tensor):
            return (k.long() % 4).to(dtype=torch.long, device=k.device)
        return int(k) % 4

    def _c4_rot_idx_from_image_gt_k(self, k) -> int | torch.Tensor:
        """Legacy name compat: default refers to f2 alignment k (not forward)."""
        return self._c4_align_rot_idx_from_image_k(k)

    def _c4_symmetric_reverse_k(self, forward_k: int) -> int:
        """Alignment direction for the reverse branch in symmetric mode (f2[1]=A).
        - "align_to_query": -k，A aligned to B_rot frame (consistent with query, default)
        - "align_b_to_a": reverse: also rotate B_rot(query) to align with A frame, A(support) stays -> both at 0 deg
        - "identity": 0, A stays but query still in B_rot frame -> query != support orientation
        - "original": +k (old bug)
        """
        mode = getattr(self, "c4_symmetric_reverse", "align_to_query")
        fk = int(forward_k) % 4
        if mode == "identity":
            return 0
        elif mode == "original":
            return fk
        elif mode == "align_b_to_a":
            return 0  # f2[half:]=enc(A) stays at 0 deg; B_rot separately aligned in f1[half:]
        else:  # "align_to_query"
            return (4 - fk) % 4

    def _apply_symmetric_rot_idx_split(self, rot_idx: torch.Tensor) -> torch.Tensor:
        """symmetric: f1=[A,Brot], f2=[Brot,A]; second half uniformly uses first pair's reverse_k (consistent with batched symmetric eval)."""
        b = int(rot_idx.shape[0])
        if b <= 1:
            return rot_idx
        half = b // 2
        rot_idx = rot_idx.clone()
        rot_idx[half:] = self._c4_symmetric_reverse_k(int(rot_idx[0].item()))
        return rot_idx

    def _c4_channel_permute(self, feat: torch.Tensor, k) -> torch.Tensor:
        if not self._is_c4_tensor(feat):
            return feat
        b, c, h, w = feat.shape
        feat_g = feat.view(b, c // 4, 4, h, w)
        if isinstance(k, int):
            shift = (-k) if self.c4_perm_reverse else k
            feat_g = torch.roll(feat_g, shifts=shift, dims=2)
            return feat_g.view(b, c, h, w)
        k = (k.long() % 4).to(feat.device)
        if self.c4_perm_reverse:
            k = (-k) % 4
        base = torch.arange(4, device=feat.device)[None, None, :, None, None]
        gather_idx = (base - k[:, None, None, None, None]) % 4
        gather_idx = gather_idx.expand(b, c // 4, 4, h, w)
        feat_g = torch.gather(feat_g, dim=2, index=gather_idx)
        return feat_g.view(b, c, h, w)

    def _estimate_c4_rotation(self, f1_s: torch.Tensor, f2_s: torch.Tensor):
        if (not self._is_c4_tensor(f1_s)) or (not self._is_c4_tensor(f2_s)):
            return None
        if f1_s.shape[1] != f2_s.shape[1]:
            return None
        x = f1_s.float()
        y = f2_s.float()
        if self.c4_est_spatial_size > 0:
            size = (self.c4_est_spatial_size, self.c4_est_spatial_size)
            x = F.adaptive_avg_pool2d(x, size)
            y = F.adaptive_avg_pool2d(y, size)
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        scores = []
        x_flat = x.flatten(1)
        for k in range(4):
            yk = self._c4_channel_permute(y, k)
            # Key: when detecting, spatially unrotate B by (-k*90) before comparing, to avoid confusing spatial rotation error with group channel error
            if self.c4_detect_use_spatial_unrotate and k > 0:
                yk = torch.rot90(yk, k=(-k) % 4, dims=[-2, -1])
                # Non-square features swap H/W at 90/270°, resample to A's size for comparison
                if yk.shape[-2:] != x.shape[-2:]:
                    yk = F.interpolate(yk, size=x.shape[-2:], mode="bilinear", align_corners=False)
            yk_flat = yk.flatten(1)
            scores.append(F.cosine_similarity(x_flat, yk_flat, dim=1))
        scores = torch.stack(scores, dim=1)  # (B, 4)
        rot_idx = scores.argmax(dim=1)
        temp = max(self.c4_score_temperature, 1e-6)
        rot_prob = torch.softmax(scores / temp, dim=1)
        top2 = torch.topk(scores, k=2, dim=1).values
        margin = top2[:, 0] - top2[:, 1]
        return {
            "rot_idx": rot_idx,
            "rot_prob": rot_prob,
            "rot_margin": margin,
            "rot_scores": scores,
        }

    def _sinkhorn_consistency_for_k(self, f1_s: torch.Tensor, f2_s: torch.Tensor, k: int):
        yk = self._c4_channel_permute(f2_s, k)
        if self.c4_detect_use_spatial_unrotate and k > 0:
            yk = torch.rot90(yk, k=(-k) % 4, dims=[-2, -1])
            if yk.shape[-2:] != f1_s.shape[-2:]:
                yk = F.interpolate(
                    yk, size=f1_s.shape[-2:], mode="bilinear", align_corners=False
                )

        pool_size = max(int(self.c4_hybrid_ot_pool_size), 0)
        if pool_size > 0:
            sz = (pool_size, pool_size)
            x = F.adaptive_avg_pool2d(f1_s.float(), sz)
            y = F.adaptive_avg_pool2d(yk.float(), sz)
        else:
            x = f1_s.float()
            y = yk.float()

        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        b, c, h, w = x.shape
        n = h * w
        x = x.view(b, c, n)
        y = y.view(b, c, n)
        sim = torch.einsum("bcn,bcm->bnm", x, y)
        cost = 1.0 - sim

        epsilon = max(float(self.c4_hybrid_ot_epsilon), 1e-6)
        K = torch.exp(-cost / epsilon)
        a = torch.full((b, n), 1.0 / n, device=x.device, dtype=x.dtype)
        b_marg = torch.full((b, n), 1.0 / n, device=x.device, dtype=x.dtype)
        v = torch.ones((b, n), device=x.device, dtype=x.dtype)
        iters = max(int(self.c4_hybrid_ot_iters), 1)
        for _ in range(iters):
            u = a / (torch.bmm(K, v.unsqueeze(2)).squeeze(2) + 1e-8)
            v = b_marg / (torch.bmm(K.transpose(1, 2), u.unsqueeze(2)).squeeze(2) + 1e-8)
        P = u.unsqueeze(2) * K * v.unsqueeze(1)
        return (P * sim).sum(dim=(1, 2))

    def _estimate_c4_rotation_hybrid(self, f1_s: torch.Tensor, f2_s: torch.Tensor):
        base = self._estimate_c4_rotation(f1_s, f2_s)
        if base is None:
            return None
        try:
            ot_scores = torch.stack(
                [self._sinkhorn_consistency_for_k(f1_s, f2_s, k) for k in range(4)],
                dim=1,
            )
            base_scores = base["rot_scores"]
            base_z = (base_scores - base_scores.mean(dim=1, keepdim=True)) / (
                base_scores.std(dim=1, keepdim=True) + 1e-6
            )
            ot_z = (ot_scores - ot_scores.mean(dim=1, keepdim=True)) / (
                ot_scores.std(dim=1, keepdim=True) + 1e-6
            )
            hybrid_scores = base_z + self.c4_hybrid_alpha * ot_z
            temp = max(self.c4_hybrid_temp, 1e-6)
            hybrid_prob = torch.softmax(hybrid_scores / temp, dim=1)
            hybrid_idx = hybrid_scores.argmax(dim=1)
            hybrid_top2 = torch.topk(hybrid_scores, k=2, dim=1).values
            hybrid_margin = hybrid_top2[:, 0] - hybrid_top2[:, 1]

            keep_mask = base["rot_margin"] >= self.c4_hybrid_margin_keep
            final_idx = torch.where(keep_mask, base["rot_idx"], hybrid_idx)
            final_prob = torch.where(keep_mask[:, None], base["rot_prob"], hybrid_prob)
            final_margin = torch.where(keep_mask, base["rot_margin"], hybrid_margin)
            final_scores = torch.where(keep_mask[:, None], base_scores, hybrid_scores)
            return {
                "rot_idx": final_idx,
                "rot_prob": final_prob,
                "rot_margin": final_margin,
                "rot_scores": final_scores,
                "rot_idx_base": base["rot_idx"],
                "rot_idx_hybrid": hybrid_idx,
                "rot_keep_base_mask": keep_mask,
                "rot_scores_base": base_scores,
                "rot_scores_ot": ot_scores,
                "rot_scores_hybrid": hybrid_scores,
            }
        except Exception:
            return base

    def _c4_spatial_align_like_detect(self, feat: torch.Tensor, k) -> torch.Tensor:
        """Spatial alignment fully consistent with _estimate_c4_rotation detection path: pure rot90, no stride offset."""
        if not self._is_c4_tensor(feat):
            return feat
        if isinstance(k, int):
            if k % 4 == 0:
                return feat
            return torch.rot90(feat, k=(-k) % 4, dims=[-2, -1])
        k = (k.long() % 4).to(feat.device)
        cands = [
            feat,
            torch.rot90(feat, k=3, dims=[-2, -1]),
            torch.rot90(feat, k=2, dims=[-2, -1]),
            torch.rot90(feat, k=1, dims=[-2, -1]),
        ]
        cand_stack = torch.stack(cands, dim=1)
        gather_idx = k.view(-1, 1, 1, 1, 1).expand(-1, 1, feat.shape[1], feat.shape[2], feat.shape[3])
        return torch.gather(cand_stack, dim=1, index=gather_idx).squeeze(1)

    def _c4_spatial_unrotate(self, feat: torch.Tensor, k) -> torch.Tensor:
        if not self._is_c4_tensor(feat):
            return feat
        if isinstance(k, int):
            if k % 4 == 0:
                return feat
            out = torch.rot90(feat, k=(-k) % 4, dims=[-2, -1])
            return out

        # Tensorized version: avoid k[i].item() causing torch.compile graph break
        k = (k.long() % 4).to(feat.device)
        cands = [
            feat,  # k=0
            torch.rot90(feat, k=3, dims=[-2, -1]),  # -1 mod 4
            torch.rot90(feat, k=2, dims=[-2, -1]),  # -2 mod 4
            torch.rot90(feat, k=1, dims=[-2, -1]),  # -3 mod 4
        ]
        cand_stack = torch.stack(cands, dim=1)  # (B, 4, C, H, W)
        gather_idx = k.view(-1, 1, 1, 1, 1).expand(-1, 1, feat.shape[1], feat.shape[2], feat.shape[3])
        return torch.gather(cand_stack, dim=1, index=gather_idx).squeeze(1)

    def _rotate_flow_coords(self, flow: torch.Tensor, k, inverse: bool = False) -> torch.Tensor:
        if isinstance(k, int):
            k = torch.full((flow.shape[0],), int(k), device=flow.device, dtype=torch.long)
        else:
            k = (k.long() % 4).to(flow.device)
        if inverse:
            k = (4 - k) % 4
        out = flow.clone()
        x = flow[:, 0:1]
        y = flow[:, 1:2]
        for ki in (1, 2, 3):
            mask = (k == ki).view(-1, 1, 1, 1)
            if not mask.any():
                continue
            if ki == 1:
                cand = torch.cat([y, -x], dim=1)
            elif ki == 2:
                cand = torch.cat([-x, -y], dim=1)
            else:
                cand = torch.cat([-y, x], dim=1)
            out = torch.where(mask, cand, out)
        return out

    def _spatially_rotate_flow_map(self, flow: torch.Tensor, k) -> torch.Tensor:
        """Grid rearrangement: inverse of _c4_spatial_unrotate(·,k), maps canonical query grid back to B_rot native indices."""
        if isinstance(k, int):
            if int(k) % 4 == 0:
                return flow
            return torch.rot90(flow, k=int(k) % 4, dims=[-2, -1])
        k = (k.long() % 4).to(flow.device)
        cands = [
            flow,
            torch.rot90(flow, k=1, dims=[-2, -1]),
            torch.rot90(flow, k=2, dims=[-2, -1]),
            torch.rot90(flow, k=3, dims=[-2, -1]),
        ]
        cand_stack = torch.stack(cands, dim=1)
        gather_idx = k.view(-1, 1, 1, 1, 1).expand(
            -1, 1, flow.shape[1], flow.shape[2], flow.shape[3]
        )
        return torch.gather(cand_stack, dim=1, index=gather_idx).squeeze(1)

    def _align_f2_pyramid_with_c4(
        self,
        f1: dict,
        f2: dict,
        coarse_scales,
        spatial_align=False,
        reuse_rot_idx: Optional[torch.Tensor] = None,
    ):
        if not self.c4_rotation_matching:
            return f2, None
        if self.c4_est_scale is not None and self.c4_est_scale in f1 and self.c4_est_scale in f2:
            est_scale = self.c4_est_scale
        else:
            candidates = [int(s) for s in coarse_scales if int(s) in f1 and int(s) in f2]
            if not candidates:
                return f2, None
            est_scale = max(candidates)
        if self.c4_force_rot_idx is not None:
            k = self._c4_align_rot_idx_from_image_k(int(self.c4_force_rot_idx) % 4)
            b = f1[est_scale].shape[0]
            device = f1[est_scale].device
            rot_idx = torch.full((b,), int(k), device=device, dtype=torch.long)
            # Symmetric mode: f2 = [B_rot, A]. f2[0]=B_rot → +k, f2[1]=A → reverse_k.
            if b > 1 and torch.equal(f1[est_scale][:b//2], f2[est_scale][b//2:]):
                rot_idx = self._apply_symmetric_rot_idx_split(rot_idx)
            rot_prob = torch.zeros((b, 4), device=device, dtype=torch.float32)
            rot_prob[torch.arange(b), rot_idx] = 1.0
            rot_scores = rot_prob.clone()
            rot_margin = torch.ones((b,), device=device, dtype=torch.float32)
            rot_info = {
                "rot_idx": rot_idx,
                "rot_prob": rot_prob,
                "rot_margin": rot_margin,
                "rot_scores": rot_scores,
                "scale": est_scale,
                "is_forced": True,
            }
            aligned_f2 = {}
            for scale, feat in f2.items():
                f2k = self._c4_channel_permute(feat, rot_idx)
                if spatial_align:
                    f2k = self._c4_spatial_unrotate(f2k, rot_idx)
                aligned_f2[scale] = f2k
            return aligned_f2, rot_info
        if reuse_rot_idx is not None:
            b = f1[est_scale].shape[0]
            device = f1[est_scale].device
            rot_idx = reuse_rot_idx.to(device=device, dtype=torch.long).reshape(-1) % 4
            if rot_idx.numel() == 1:
                rot_idx = rot_idx.expand(b)
            elif rot_idx.numel() != b:
                raise ValueError(
                    f"reuse_rot_idx length {int(rot_idx.numel())} != batch {int(b)}"
                )
            # Symmetric mode: f2 = [B_rot, A]. f2[1]=A → reverse_k.
            if b > 1 and torch.equal(f1[est_scale][:b//2], f2[est_scale][b//2:]):
                rot_idx = self._apply_symmetric_rot_idx_split(rot_idx)
            rot_prob = torch.zeros((b, 4), device=device, dtype=torch.float32)
            rot_prob.scatter_(1, rot_idx.unsqueeze(1), 1.0)
            rot_scores = rot_prob.clone()
            rot_margin = torch.ones((b,), device=device, dtype=torch.float32)
            rot_info = {
                "rot_idx": rot_idx,
                "rot_prob": rot_prob,
                "rot_margin": rot_margin,
                "rot_scores": rot_scores,
                "scale": est_scale,
                "is_reused_from_coarse": True,
            }
            aligned_f2 = {}
            for scale, feat in f2.items():
                f2k = self._c4_channel_permute(feat, rot_idx)
                if spatial_align:
                    f2k = self._c4_spatial_unrotate(f2k, rot_idx)
                aligned_f2[scale] = f2k
            return aligned_f2, rot_info
        with torch.no_grad():
            if self.c4_hybrid:
                rot_info = self._estimate_c4_rotation_hybrid(
                    f1[est_scale], f2[est_scale]
                )
            else:
                rot_info = self._estimate_c4_rotation(f1[est_scale], f2[est_scale])
        if rot_info is None:
            return f2, None
        rot_idx = rot_info["rot_idx"]
        # Symmetric mode: f2 = [B_rot, A]. f2[0]=B_rot → +k, f2[1]=A → reverse_k.
        b = f1[est_scale].shape[0]
        if b > 1 and torch.equal(f1[est_scale][:b//2], f2[est_scale][b//2:]):
            rot_idx = self._apply_symmetric_rot_idx_split(rot_idx)
            rot_info["rot_idx"] = rot_idx
            rot_info["rot_prob"] = rot_info["rot_prob"].clone()
            rot_info["rot_prob"][b//2:] = 0
            rot_info["rot_prob"][b//2:, 0] = 1.0
            rot_info["rot_scores"] = rot_info["rot_scores"].clone()
            rot_info["rot_scores"][b//2:] = 0
            rot_info["rot_scores"][b//2:, 0] = 1.0
        aligned_f2 = {}
        for scale, feat in f2.items():
            f2k = self._c4_channel_permute(feat, rot_idx)
            if spatial_align:
                f2k = self._c4_spatial_unrotate(f2k, rot_idx)
            aligned_f2[scale] = f2k
        rot_info["scale"] = est_scale
        return aligned_f2, rot_info

    def forward(
        self,
        f1,
        f2,
        gt_warp=None,
        gt_prob=None,
        upsample=False,
        flow=None,
        certainty=None,
        scale_factor=1,
    ):
        """Run the public inference path and return dense warp/certainty maps."""
        return self._forward_inference(f1, f2, upsample, flow, certainty, scale_factor)
    
    def _forward_inference(self, f1, f2, upsample, flow, certainty, scale_factor):
        """🚀 Optimized inference path - reduced dict ops and autocast overhead"""
        coarse_scales = self.embedding_decoder.scales()
        all_scales = self.scales if not upsample else ["8", "4", "2", "1"]
        sizes = {scale: f1[scale].shape[-2:] for scale in f1}
        h, w = sizes[1]
        b = f1[1].shape[0]
        device = f1[1].device
        coarsest_scale = int(all_scales[0])
        
        if not upsample:
            flow = self.get_placeholder_flow(b, *sizes[coarsest_scale], device)
            certainty = 0.0
        else:
            flow = F.interpolate(flow, size=sizes[coarsest_scale], align_corners=False, mode="bilinear")
            certainty = F.interpolate(certainty, size=sizes[coarsest_scale], align_corners=False, mode="bilinear")
        reuse_rot_idx = None
        if upsample and self.c4_reuse_coarse_rot_on_upsample:
            reuse_rot_idx = getattr(self, "_sticky_c4_rot_idx", None)
        try:
            f2_aligned, rot_info = self._align_f2_pyramid_with_c4(
                f1,
                f2,
                coarse_scales,
                spatial_align=self.c4_spatial_canonicalize,
                reuse_rot_idx=reuse_rot_idx,
            )
        finally:
            if upsample and reuse_rot_idx is not None and hasattr(self, "_sticky_c4_rot_idx"):
                delattr(self, "_sticky_c4_rot_idx")
        self._last_c4_rotation = rot_info
        # align_b_to_a reverse half-batch: f1[half:]=enc(B_rot) rotated to align with A frame
        if (
            rot_info is not None
            and self.c4_rotation_matching
            and getattr(self, "c4_symmetric_reverse", "align_to_query") == "align_b_to_a"
            and b > 1
        ):
            est_scale = rot_info.get("scale")
            if est_scale is None:
                _cand = [int(s) for s in coarse_scales if int(s) in f1 and int(s) in f2_aligned]
                est_scale = max(_cand) if _cand else 16
            half = b // 2
            if torch.equal(f1[est_scale][:half], f2_aligned[est_scale][half:]):
                if not getattr(self, "_align_b_to_a_ablate_f1", False):
                    fwd_idx = rot_info["rot_idx"][:half]
                    f1 = {
                        int(scale): torch.cat(
                            [
                                f1[scale][:half],
                                self._c4_spatial_unrotate(
                                    self._c4_channel_permute(f1[scale][half:], fwd_idx),
                                    fwd_idx,
                                ),
                            ],
                            dim=0,
                        )
                        for scale in f1
                    }
        if (
            upsample
            and self.c4_rotation_matching
            and self.c4_spatial_canonicalize
            and self.c4_rotate_flow_back
            and rot_info is not None
        ):
            flow = self._rotate_flow_coords(flow, rot_info["rot_idx"], inverse=True)
        
        # save scale 16 certainty (for attenuate_cert feature)
        certainty_16 = None
        
        # Single autocast wraps the entire decoding process
        with torch.autocast(device.type, enabled=device.type == "cuda", dtype=self.amp_dtype):
            for new_scale in all_scales:
                ins = int(new_scale)
                f1_s, f2_s = f1[ins], f2_aligned[ins]
                
                # Projection
                if new_scale in self.proj:
                    f1_s = self.proj[new_scale](f1_s)
                    f2_s = self.proj[new_scale](f2_s)
                
                # GPS + Coarse Head
                if ins in coarse_scales:
                    gp_posterior = self.gps[new_scale](f1_s, f2_s)
                    gm_warp_or_cls, certainty, _ = self.embedding_decoder(
                        gp_posterior, f1_s, None, new_scale
                    )
                    if self.embedding_decoder.is_classifier:
                        flow = cls_to_flow_refine(gm_warp_or_cls).permute(0, 3, 1, 2)
                    else:
                        flow = gm_warp_or_cls.detach()
                
                # ConvRefiner
                if new_scale in self.conv_refiner:
                    delta_flow, delta_certainty = self.conv_refiner[new_scale](
                        f1_s, f2_s, flow, scale_factor=scale_factor, logits=certainty
                    )
                    displacement = ins * torch.stack(
                        (delta_flow[:, 0].float() / (self.refine_init * w),
                         delta_flow[:, 1].float() / (self.refine_init * h)),
                        dim=1,
                    )
                    flow = flow + displacement
                    certainty = certainty + delta_certainty
                
                # Save scale-16 results (for attenuate_cert)
                if ins == 16:
                    certainty_16 = certainty.clone()
                
                # Upsample
                if new_scale != "1":
                    flow = F.interpolate(flow, size=sizes[ins // 2], mode=self.flow_upsample_mode)
                    certainty = F.interpolate(certainty, size=sizes[ins // 2], mode=self.flow_upsample_mode)
                    if self.detach:
                        flow = flow.detach()
                        certainty = certainty.detach()
        
        if (
            self.c4_rotation_matching
            and self.c4_spatial_canonicalize
            and self.c4_rotate_flow_back
            and rot_info is not None
        ):
            flow = self._rotate_flow_coords(flow, rot_info["rot_idx"], inverse=False)

        # align_b_to_a reverse: f1 already aligned to 0°, flow needs rearrangement back to B_rot native grid (only second half, only symmetric)
        if (
            rot_info is not None
            and self.c4_rotation_matching
            and getattr(self, "c4_symmetric_reverse", "align_to_query") == "align_b_to_a"
            and b > 1
        ):
            _est = rot_info.get("scale")
            if _est is None:
                _cand = [int(s) for s in coarse_scales if int(s) in f1 and int(s) in f2_aligned]
                _est = max(_cand) if _cand else 16
            _half = b // 2
            if torch.equal(f1[_est][:_half], f2_aligned[_est][_half:]):
                if not getattr(self, "_align_b_to_a_ablate_flow_remap", False):
                    _fwd = rot_info["rot_idx"][:_half]
                    flow = torch.cat(
                        [flow[:_half], self._spatially_rotate_flow_map(flow[_half:], _fwd)],
                        dim=0,
                    )

        # return necessary results (scale 1 and scale 16)
        result = {1: {"flow": flow, "certainty": certainty}}
        if certainty_16 is not None:
            result[16] = {"certainty": certainty_16}
        return result
    
def _check_input(im_input):
    if isinstance(im_input, (str, os.PathLike)):
        im = Image.open(im_input)
        check_not_i16(im)
        im = im.convert("RGB")
    elif isinstance(im_input, Image.Image):
        check_rgb(im_input)
        im = im_input
    else:
        assert isinstance(im_input, torch.Tensor), (
            "im_input must be a string, path, or PIL image"
        )
        B, C, H, W = im_input.shape
        assert C == 3, "im_input must be a RGB image"
        # The exported encoder accepts arbitrary image sizes.
        im = im_input
    return im


class RegressionMatcher(nn.Module):
    def __init__(
        self,
        encoder: E2CNNEncoderExportedMaxPool,
        decoder: Decoder,
        h=448,
        w=448,
        sample_mode="threshold_balanced",
        upsample_preds=False,
        symmetric=False,
        sample_thresh=0.05,
        name=None,
        attenuate_cert=None,
        upsample_res=None,
    ):
        super().__init__()
        self.attenuate_cert = attenuate_cert
        self.encoder = encoder
        self.decoder = decoder
        self.name = name
        self.w_resized = w
        self.h_resized = h
        self.og_transforms = get_tuple_transform_ops(resize=None, normalize=True)
        self.sample_mode = sample_mode
        self.upsample_preds = upsample_preds
        self.upsample_res = upsample_res or (14 * 16 * 6, 14 * 16 * 6)
        self.symmetric = symmetric
        self.sample_thresh = sample_thresh

    def get_output_resolution(self):
        if not self.upsample_preds:
            return self.h_resized, self.w_resized
        else:
            return self.upsample_res

    def extract_backbone_features(self, batch, batched=True, upsample=False):
        x_q = batch["im_A"]
        x_s = batch["im_B"]
        if batched:
            X = torch.cat((x_q, x_s), dim=0)
            feature_pyramid = self.encoder(X, upsample=upsample)
        else:
            feature_pyramid = (
                self.encoder(x_q, upsample=upsample),
                self.encoder(x_s, upsample=upsample),
            )
        return feature_pyramid

    def _grid_sample(self, matches, certainty, num, grid_h=8, grid_w=8):
        cert_map = certainty
        if cert_map.dim() == 3:
            cert_map = cert_map[0] if cert_map.shape[0] == 1 else cert_map.mean(dim=0)
        h, w = cert_map.shape[-2:]
        grid_h = min(grid_h, h)
        grid_w = min(grid_w, w)
        cell_h = math.ceil(h / grid_h)
        cell_w = math.ceil(w / grid_w)
        k_per_cell = max(1, math.ceil(num / (grid_h * grid_w)))

        device = cert_map.device
        flat_indices = []
        for gh in range(grid_h):
            r0 = gh * cell_h
            r1 = min((gh + 1) * cell_h, h)
            if r0 >= r1:
                continue
            for gw in range(grid_w):
                c0 = gw * cell_w
                c1 = min((gw + 1) * cell_w, w)
                if c0 >= c1:
                    continue
                cell_cert = cert_map[r0:r1, c0:c1].reshape(-1)
                if cell_cert.numel() == 0 or cell_cert.max() <= 0:
                    continue
                k = min(k_per_cell, cell_cert.numel())
                topk = torch.topk(cell_cert, k=k, largest=True)
                rows = torch.arange(r0, r1, device=device)[:, None].expand(r1 - r0, c1 - c0)
                cols = torch.arange(c0, c1, device=device)[None, :].expand(r1 - r0, c1 - c0)
                cell_flat = (rows * w + cols).reshape(-1)
                flat_indices.append(cell_flat[topk.indices])

        if not flat_indices:
            return matches.reshape(-1, 4), certainty.reshape(-1)

        flat_indices = torch.cat(flat_indices)
        cert_flat = certainty.reshape(-1)
        if flat_indices.numel() > num:
            topk = torch.topk(cert_flat[flat_indices], k=num, largest=True)
            flat_indices = flat_indices[topk.indices]

        return matches.reshape(-1, 4)[flat_indices], cert_flat[flat_indices]

    def sample(
        self,
        matches,
        certainty,
        num=10000,
    ):
        """Sample matching points (density-guided optimized version)"""
        # [Innovation optimization v3] Density-Guided Deterministic Sampling
        # advantages: 
        # 1. vs NMS: soft suppression, won't kill good points due to sub-pixel shift
        # 2. vs Grid: not bounded by grid edges, naturally adapts to image content
        # 3. vs KDE: extremely fast (AvgPool vs N^2), TopK ensures deterministic results (more stable accuracy)

        if "grid" in self.sample_mode:
            # 1. Prepare data
            if certainty.dim() == 2:
                c_map = certainty.unsqueeze(0).unsqueeze(0)
            else:
                c_map = certainty.unsqueeze(1) # B, 1, H, W
            
            # [Optimization fix] Purified Density estimation
            # Only compute density of 'valid points', avoid background noise interfering with distribution
            # This more accurately simulates KDE (only stats density on good_samples)
            valid_map = c_map.clone()
            if self.sample_thresh > 0:
                valid_map[valid_map < self.sample_thresh] = 0
            
            # 2. Compute local density
            k_size = 33 
            density = F.avg_pool2d(valid_map, kernel_size=k_size, stride=1, padding=k_size//2)
            
            # 3. Density reweighting
            eps = 1e-5
            
            # [Urgent fix] Square root (0.5) was too aggressive, causing high-score points to lose all advantage
            # Symptom: accuracy collapse means we selected scattered "garbage"
            # Fix: 
            # 1. Restore respect for original confidence (1.5 -> 2.0), must ensure selected points are trustworthy first
            # 2. Introduce log-density decay instead of dividing by density directly
            #    Original formula: score = cert / density (extremely sensitive, slightly higher density kills points)
            #    New formula: score = cert^2 / log(density + e) (gentle penalty, only suppresses extreme crowding)
            
            density_log = torch.log(density + 1.0)
            balanced_score = (c_map ** 2.0) / (density_log + eps)
            
            # 4. TopK sampling (deterministic)
            score_flat = balanced_score.reshape(-1)
            matches_flat = matches.reshape(-1, 4)
            certainty_flat = certainty.reshape(-1)
            
            # Pre-filtering (unchanged)
            if self.sample_thresh > 0:
                mask = certainty_flat > self.sample_thresh
                if mask.sum() > num: 
                    score_flat = score_flat[mask]
                    matches_flat = matches_flat[mask]
                    certainty_flat = certainty_flat[mask]

            k_final = min(len(score_flat), num)
            if k_final > 0:
                topk = torch.topk(score_flat, k=k_final)
                return (
                    matches_flat[topk.indices].detach().cpu(),
                    certainty_flat[topk.indices].detach().cpu(),
                )
            else:
                return (
                    matches_flat[:0].detach().cpu(),
                    certainty_flat[:0].detach().cpu(),
                )

        # Original logic (threshold/balanced) preserved as fallback or other mode
        if "threshold" in self.sample_mode:
            upper_thresh = self.sample_thresh
            certainty = certainty.clone()
            certainty[certainty > upper_thresh] = 1
            
        matches, certainty = (
            matches.reshape(-1, 4),
            certainty.reshape(-1),
        )
        certainty = torch.nan_to_num(certainty, nan=0.0, posinf=1.0, neginf=0.0).clamp(min=0.0)
        if float(certainty.sum().item()) <= 0.0:
            certainty = torch.full_like(certainty, 1e-8)

        expansion_factor = 4 if "balanced" in self.sample_mode else 1
        good_samples = torch.multinomial(
            certainty,
            num_samples=min(expansion_factor * num, len(certainty)),
            replacement=False,
        )
        good_matches, good_certainty = matches[good_samples], certainty[good_samples]
        
        if "balanced" not in self.sample_mode:
            return good_matches.detach().cpu(), good_certainty.detach().cpu()
            
        density = kde(good_matches, std=0.1)
        p = 1 / (density + 1)
        p = torch.nan_to_num(p, nan=1e-7, posinf=1.0, neginf=1e-7).clamp(min=1e-7)
        p[density < 10] = 1e-7
        balanced_samples = torch.multinomial(
            p, num_samples=min(num, len(good_certainty)), replacement=False
        )
        return (
            good_matches[balanced_samples].detach().cpu(),
            good_certainty[balanced_samples].detach().cpu(),
        )

    def forward(self, batch, batched=True, upsample=False, scale_factor=1):
        feature_pyramid = self.extract_backbone_features(
            batch, batched=batched, upsample=upsample
        )
        if batched:
            f_q_pyramid = {
                scale: f_scale.chunk(2)[0] for scale, f_scale in feature_pyramid.items()
            }
            f_s_pyramid = {
                scale: f_scale.chunk(2)[1] for scale, f_scale in feature_pyramid.items()
            }
        else:
            f_q_pyramid, f_s_pyramid = feature_pyramid
        corresps = self.decoder(
            f_q_pyramid,
            f_s_pyramid,
            upsample=upsample,
            **(batch["corresps"] if "corresps" in batch else {}),
            scale_factor=scale_factor,
        )

        return corresps

    def forward_symmetric(self, batch, batched=True, upsample=False, scale_factor=1):
        feature_pyramid = self.extract_backbone_features(
            batch, batched=batched, upsample=upsample
        )
        f_q_pyramid = feature_pyramid
        f_s_pyramid = {
            scale: torch.cat((f_scale.chunk(2)[1], f_scale.chunk(2)[0]), dim=0)
            for scale, f_scale in feature_pyramid.items()
        }
        corresps = self.decoder(
            f_q_pyramid,
            f_s_pyramid,
            upsample=upsample,
            **(batch["corresps"] if "corresps" in batch else {}),
            scale_factor=scale_factor,
        )
        return corresps

    def conf_from_fb_consistency(self, flow_forward, flow_backward, th=2):
        # assumes that flow forward is of shape (..., H, W, 2)
        has_batch = False
        if len(flow_forward.shape) == 3:
            flow_forward, flow_backward = flow_forward[None], flow_backward[None]
        else:
            has_batch = True
        H, W = flow_forward.shape[-3:-1]
        th_n = 2 * th / max(H, W)
        coords = torch.stack(
            torch.meshgrid(
                torch.linspace(-1 + 1 / W, 1 - 1 / W, W),
                torch.linspace(-1 + 1 / H, 1 - 1 / H, H),
                indexing="xy",
            ),
            dim=-1,
        ).to(flow_forward.device)
        coords_fb = F.grid_sample(
            flow_backward.permute(0, 3, 1, 2),
            flow_forward,
            align_corners=False,
            mode="bilinear",
        ).permute(0, 2, 3, 1)
        diff = (coords - coords_fb).norm(dim=-1)
        in_th = (diff < th_n).float()
        if not has_batch:
            in_th = in_th[0]
        return in_th

    def to_pixel_coordinates(self, coords, H_A, W_A, H_B=None, W_B=None):
        if coords.shape[-1] == 2:
            return self._to_pixel_coordinates(coords, H_A, W_A)

        if isinstance(coords, (list, tuple)):
            kpts_A, kpts_B = coords[0], coords[1]
        else:
            kpts_A, kpts_B = coords[..., :2], coords[..., 2:]
        return self._to_pixel_coordinates(kpts_A, H_A, W_A), self._to_pixel_coordinates(
            kpts_B, H_B, W_B
        )

    def _to_pixel_coordinates(self, coords, H, W):
        kpts = torch.stack(
            (W / 2 * (coords[..., 0] + 1), H / 2 * (coords[..., 1] + 1)), axis=-1
        )
        return kpts

    def to_normalized_coordinates(self, coords, H_A, W_A, H_B, W_B):
        if isinstance(coords, (list, tuple)):
            kpts_A, kpts_B = coords[0], coords[1]
        else:
            kpts_A, kpts_B = coords[..., :2], coords[..., 2:]
        kpts_A = torch.stack(
            (2 / W_A * kpts_A[..., 0] - 1, 2 / H_A * kpts_A[..., 1] - 1), axis=-1
        )
        kpts_B = torch.stack(
            (2 / W_B * kpts_B[..., 0] - 1, 2 / H_B * kpts_B[..., 1] - 1), axis=-1
        )
        return kpts_A, kpts_B

    def match_keypoints(
        self,
        x_A,
        x_B,
        warp,
        certainty,
        return_tuple=True,
        return_inds=False,
        max_dist=0.005,
        cert_th=0,
    ):
        x_A_to_B = F.grid_sample(
            warp[..., -2:].permute(2, 0, 1)[None],
            x_A[None, None],
            align_corners=False,
            mode="bilinear",
        )[0, :, 0].mT
        cert_A_to_B = F.grid_sample(
            certainty[None, None, ...],
            x_A[None, None],
            align_corners=False,
            mode="bilinear",
        )[0, 0, 0]
        D = torch.cdist(x_A_to_B, x_B)
        inds_A, inds_B = torch.nonzero(
            (D == D.min(dim=-1, keepdim=True).values)
            * (D == D.min(dim=-2, keepdim=True).values)
            * (cert_A_to_B[:, None] > cert_th)
            * (D < max_dist),
            as_tuple=True,
        )

        if return_tuple:
            if return_inds:
                return inds_A, inds_B
            else:
                return x_A[inds_A], x_B[inds_B]
        else:
            if return_inds:
                return torch.cat((inds_A, inds_B), dim=-1)
            else:
                return torch.cat((x_A[inds_A], x_B[inds_B]), dim=-1)
    
    def _get_device(self):
        # let's hope this is same for all weights
        return self.encoder.cnn.layers[0].weight.device

    @torch.inference_mode()
    def match(
        self,
        im_A_input,
        im_B_input,
        *args,
        im_A_high_res=None,
        im_B_high_res=None,
        batched=True,
        device=None,
    ):
        self.train(False)
        if not batched:
            raise ValueError("batched must be True, non-batched inference is no longer supported.")
        dec0 = getattr(self, "decoder", None)
        if dec0 is not None and hasattr(dec0, "_sticky_c4_rot_idx"):
            delattr(dec0, "_sticky_c4_rot_idx")
        if device is None and not isinstance(im_A_input, torch.Tensor):
            device = self._get_device()
        elif device is None and isinstance(im_A_input, torch.Tensor):
            device = im_A_input.device

        # Check if inputs are file paths or already loaded images
        im_A = _check_input(im_A_input)
        im_B = _check_input(im_B_input)
        symmetric = self.symmetric
        ws = self.w_resized
        hs = self.h_resized

        scale_factor = math.sqrt(hs * ws / (560**2)) # divide by training resolution
        if isinstance(im_A, Image.Image) and isinstance(im_B, Image.Image):
            b = 1
            w, h = im_A.size
            w2, h2 = im_B.size
            # Get images in good format

            test_transform = get_tuple_transform_ops(
                resize=(hs, ws), normalize=True, clahe=False
            )
            im_A, im_B = test_transform((im_A, im_B))
            batch = {"im_A": im_A[None].to(device), "im_B": im_B[None].to(device)}
        elif isinstance(im_A, torch.Tensor) and isinstance(im_B, torch.Tensor):
            b, c, h, w = im_A.shape
            b, c, h2, w2 = im_B.shape
            assert w == w2 and h == h2, "For batched images we assume same size"
            batch = {"im_A": im_A.to(device), "im_B": im_B.to(device)}
            if h != self.h_resized or self.w_resized != w:
                warn(
                    "Model resolution and batch resolution differ, may produce unexpected results"
                )
            hs, ws = h, w
        else:
            raise ValueError(f"Unsupported input type: {type(im_A)=} and {type(im_B)=}")
        finest_scale = 1
        # Run matcher
        if symmetric:
            corresps = self.forward_symmetric(batch, scale_factor=scale_factor)
        else:
            corresps = self.forward(batch, batched=True, scale_factor=scale_factor)

        if (
            self.upsample_preds
            and self.decoder is not None
            and getattr(self.decoder, "c4_rotation_matching", False)
            and getattr(self.decoder, "c4_reuse_coarse_rot_on_upsample", True)
        ):
            lr = getattr(self.decoder, "_last_c4_rotation", None)
            if lr is not None and lr.get("rot_idx") is not None:
                self.decoder._sticky_c4_rot_idx = lr["rot_idx"].detach().clone()
            elif hasattr(self.decoder, "_sticky_c4_rot_idx"):
                delattr(self.decoder, "_sticky_c4_rot_idx")

        if self.upsample_preds:
            hs, ws = self.upsample_res

        if self.attenuate_cert:
            low_res_certainty = F.interpolate(
                corresps[16]["certainty"],
                size=(hs, ws),
                align_corners=False,
                mode="bilinear",
            )
            cert_clamp = 0
            factor = 0.5
            low_res_certainty = (
                factor * low_res_certainty * (low_res_certainty < cert_clamp)
            )

        finest_corresps = corresps[finest_scale]
        if self.upsample_preds and im_A_high_res is None and im_B_high_res is None:
            torch.cuda.empty_cache()
            test_transform = get_tuple_transform_ops(resize=(hs, ws), normalize=True)
            if isinstance(im_A_input, (str, os.PathLike)):
                im_A, im_B = test_transform(
                    (
                        Image.open(im_A_input).convert("RGB"),
                        Image.open(im_B_input).convert("RGB"),
                    )
                )
                im_A, im_B = im_A[None].to(device), im_B[None].to(device)
            elif isinstance(im_A_input, Image.Image):
                im_A, im_B = test_transform((im_A_input, im_B_input))
                im_A, im_B = im_A[None].to(device), im_B[None].to(device)
            elif isinstance(im_A_input, torch.Tensor):
                # Resize Tensor directly (assumes already normalized)
                im_A = F.interpolate(im_A_input, size=(hs, ws), mode='bilinear', align_corners=False)
                im_B = F.interpolate(im_B_input, size=(hs, ws), mode='bilinear', align_corners=False)
            else:
                raise ValueError(f"Unsupported input type: {type(im_A_input)=}")
            
            batch = {"im_A": im_A, "im_B": im_B, "corresps": finest_corresps}
        elif self.upsample_preds and im_A_high_res is not None and im_B_high_res is not None:
            batch = {"im_A": im_A_high_res, "im_B": im_B_high_res, "corresps": finest_corresps}
        elif self.upsample_preds:
            raise ValueError(f"Invalid upsample_preds and high_res inputs with {im_A=},{im_A_high_res=},{im_B=} and {im_B_high_res=}")

        if self.upsample_preds:
            scale_factor = math.sqrt(
                self.upsample_res[0]
                * self.upsample_res[1]
                / (560**2) # divide by training resolution
            )
            if symmetric:
                corresps = self.forward_symmetric(
                    batch, upsample=True, batched=True, scale_factor=scale_factor
                )
            else:
                corresps = self.forward(
                    batch, batched=True, upsample=True, scale_factor=scale_factor
                )

        im_A_to_im_B = corresps[finest_scale]["flow"]
        certainty = corresps[finest_scale]["certainty"] - (
            low_res_certainty if self.attenuate_cert else 0
        )
        if finest_scale != 1:
            im_A_to_im_B = F.interpolate(
                im_A_to_im_B, size=(hs, ws), align_corners=False, mode="bilinear"
            )
            certainty = F.interpolate(
                certainty, size=(hs, ws), align_corners=False, mode="bilinear"
            )
        im_A_to_im_B = im_A_to_im_B.permute(0, 2, 3, 1)
        # Create im_A meshgrid
        im_A_coords = torch.meshgrid(
            (
                torch.linspace(-1 + 1 / hs, 1 - 1 / hs, hs, device=device),
                torch.linspace(-1 + 1 / ws, 1 - 1 / ws, ws, device=device),
            ),
            indexing="ij",
        )
        im_A_coords = torch.stack((im_A_coords[1], im_A_coords[0]))
        im_A_coords = im_A_coords[None].expand(b, 2, hs, ws)
        certainty = certainty.sigmoid()  # logits -> probs
        im_A_coords = im_A_coords.permute(0, 2, 3, 1)
        if (im_A_to_im_B.abs() > 1).any() and True:
            wrong = (im_A_to_im_B.abs() > 1).sum(dim=-1) > 0
            certainty[wrong[:, None]] = 0
        im_A_to_im_B = torch.clamp(im_A_to_im_B, -1, 1)
        if symmetric:
            A_to_B, B_to_A = im_A_to_im_B.chunk(2)

            if getattr(self.decoder, "cycle_cert_filter", False):
                cycle_tau = max(getattr(self.decoder, "cycle_cert_filter_tau", 0.05), 1e-6)
                B_to_A_bchw = B_to_A.permute(0, 3, 1, 2)
                cycle_A = F.grid_sample(
                    B_to_A_bchw, A_to_B, mode="bilinear", align_corners=False
                ).permute(0, 2, 3, 1)
                cycle_error_A = (cycle_A - im_A_coords).norm(dim=-1)
                A_to_B_bchw = A_to_B.permute(0, 3, 1, 2)
                cycle_B = F.grid_sample(
                    A_to_B_bchw, B_to_A, mode="bilinear", align_corners=False
                ).permute(0, 2, 3, 1)
                cycle_error_B = (cycle_B - im_A_coords).norm(dim=-1)
                cycle_quality_A = torch.exp(-cycle_error_A / cycle_tau)
                cycle_quality_B = torch.exp(-cycle_error_B / cycle_tau)
                cert_A, cert_B = certainty.chunk(2)
                cert_A = cert_A * cycle_quality_A.unsqueeze(1)
                cert_B = cert_B * cycle_quality_B.unsqueeze(1)
                certainty = torch.cat((cert_A, cert_B), dim=0)

            q_warp = torch.cat((im_A_coords, A_to_B), dim=-1)
            s_warp = torch.cat((B_to_A, im_A_coords), dim=-1)
            warp = torch.cat((q_warp, s_warp), dim=2)
            certainty = torch.cat(certainty.chunk(2), dim=3)  # (b, 1, H, 2W)
        else:
            warp = torch.cat((im_A_coords, im_A_to_im_B), dim=-1)
        if batched:
            return (warp, certainty[:, 0])
        else:
            return (
                warp[0],
                certainty[0, 0],
            )

    def visualize_warp(
        self,
        warp,
        certainty,
        im_A=None,
        im_B=None,
        im_A_path=None,
        im_B_path=None,
        device="cuda",
        symmetric=True,
        save_path=None,
        unnormalize=False,
    ):
        # assert symmetric == True, "Currently assuming bidirectional warp, might update this if someone complains ;)"
        H, W2, _ = warp.shape
        W = W2 // 2 if symmetric else W2
        if im_A is None:
            from PIL import Image

            im_A, im_B = (
                Image.open(im_A_path).convert("RGB"),
                Image.open(im_B_path).convert("RGB"),
            )
        if not isinstance(im_A, torch.Tensor):
            im_A = im_A.resize((W, H))
            im_B = im_B.resize((W, H))
            x_B = (torch.tensor(np.array(im_B)) / 255).to(device).permute(2, 0, 1)
            if symmetric:
                x_A = (torch.tensor(np.array(im_A)) / 255).to(device).permute(2, 0, 1)
        else:
            if symmetric:
                x_A = im_A
            x_B = im_B
        im_A_transfer_rgb = F.grid_sample(
            x_B[None], warp[:, :W, 2:][None], mode="bilinear", align_corners=False
        )[0]
        if symmetric:
            im_B_transfer_rgb = F.grid_sample(
                x_A[None], warp[:, W:, :2][None], mode="bilinear", align_corners=False
            )[0]
            warp_im = torch.cat((im_A_transfer_rgb, im_B_transfer_rgb), dim=2)
            white_im = torch.ones((H, 2 * W), device=device)
        else:
            warp_im = im_A_transfer_rgb
            white_im = torch.ones((H, W), device=device)
        vis_im = certainty * warp_im + (1 - certainty) * white_im
        if save_path is not None:
            from redimatch.utils import tensor_to_pil

            tensor_to_pil(vis_im, unnormalize=unnormalize).save(save_path)
        return vis_im
