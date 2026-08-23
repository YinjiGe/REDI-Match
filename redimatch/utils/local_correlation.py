from typing import Literal
import ctypes
import glob
import os
import torch
import torch.nn.functional as F


def _preload_cuda12_runtime():
    """Preload libcudart.so.12 before `import local_corr`.

    The fused-local-corr wheel links the CUDA 12 runtime (libcudart.so.12), but
    torch may be a cu118 build (CUDA 11.8). Preloading it here lets the dynamic
    linker resolve the dependency without requiring LD_LIBRARY_PATH, so the fused
    op is used instead of silently falling back to the native implementation.
    """
    try:
        import nvidia.cuda_runtime
    except ImportError:
        return
    libdir = os.path.join(os.path.dirname(nvidia.cuda_runtime.__file__), "lib")
    for so in glob.glob(os.path.join(libdir, "libcudart.so.12*")):
        try:
            ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass


_preload_cuda12_runtime()


def local_corr_wrapper(
    feature0: torch.Tensor,
    feature1: torch.Tensor,
    coords: torch.Tensor,
    local_window: torch.Tensor,
    B,
    K,
    c,
    r,
    h,
    w,
    device,
    padding_mode="zeros",
    sample_mode: Literal["bilinear", "nearest"] = "bilinear",
    dtype=torch.float32,
):
    import local_corr
    assert padding_mode == "zeros"
    warp = (coords[..., None, :] + local_window[:, None, None]).reshape(B, h * w, K, 2)
    corr = (
        local_corr.local_corr(
            feature0.reshape(B, c, h * w).permute(0, 2, 1).float() / (c**0.5),
            feature1.permute(0, 2, 3, 1).clone().detach().float(),
            warp.clone().detach(),
            mode=sample_mode,
            normalized_coords=True,
        )
        .permute(0, 2, 1)
        .reshape(B, K, h, w)
    )
    return corr, warp


def shitty_native_torch_local_corr(
    feature0,
    feature1,
    warp,
    local_window,
    B,
    K,
    c,
    r,
    h,
    w,
    device,
    padding_mode="zeros",
    sample_mode="bilinear",
    dtype=torch.float32,
):
    """🚀 Optimized version: batch processing, eliminates for loops"""
    # warp: (B, H, W, 2), local_window: (1, K, 2)
    # Expand all coordinates at once (B, H, W, K, 2) -> (B, H, W*K, 2)
    local_window_coords = (
        warp[:, :, :, None, :] + local_window[None, None, None, :, :]
    ).reshape(B, h, w * K, 2)
    
    # batch grid_sample (B, C, H, W*K)
    with torch.no_grad():
        window_feature = F.grid_sample(
            feature1,
            local_window_coords,
            padding_mode=padding_mode,
            align_corners=False,
            mode=sample_mode,
        )
    
    # Reshape: (B, C, H, W, K)
    window_feature = window_feature.reshape(B, c, h, w, K)
    
    # batch dot product: (B, C, H, W, 1) * (B, C, H, W, K) -> sum -> (B, K, H, W)
    corr = torch.einsum('bchw,bchwk->bkhw', feature0 / (c ** 0.5), window_feature)
    
    return corr, None


def local_correlation(
    feature0: torch.Tensor,  # (B x C x H x W)
    feature1: torch.Tensor,  # (B x C x H x W)
    local_radius: int,
    warp: torch.Tensor,  # (B x 2 x H x W)
    *,
    use_custom_corr: bool,
    padding_mode="zeros",
    sample_mode: Literal["bilinear", "nearest"] = "bilinear",
):
    r = local_radius
    K = (2 * r + 1) ** 2
    B, c, h, w = feature0.size()
    warp = warp.permute(0, 2, 3, 1)
    device = feature0.device
    dtype = feature0.dtype
    local_window = torch.meshgrid(
        [
            torch.linspace(
                -2 * local_radius / h, 2 * local_radius / h, 2 * r + 1, device=device
            ),
            torch.linspace(
                -2 * local_radius / w, 2 * local_radius / w, 2 * r + 1, device=device
            ),
        ],
        indexing="ij",
    )
    local_window = (
        torch.stack((local_window[1], local_window[0]), dim=-1)[None]
        .expand(1, 2 * r + 1, 2 * r + 1, 2)
        .reshape(1, K, 2)
    )
    if not use_custom_corr:
        corr, corr_coords = shitty_native_torch_local_corr(
            feature0,
            feature1,
            warp,
            local_window,
            B,
            K,
            c,
            r,
            h,
            w,
            device,
            padding_mode,
            sample_mode,
            dtype,
        )
    else:
        corr, corr_coords = local_corr_wrapper(
            feature0,
            feature1,
            warp,
            local_window,
            B,
            K,
            c,
            r,
            h,
            w,
            device,
            padding_mode,
            sample_mode,
            dtype,
        )
    return corr