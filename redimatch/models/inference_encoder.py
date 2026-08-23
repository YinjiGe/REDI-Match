#!/usr/bin/env python3
"""Small inference-only encoder used by the public release."""

import os
from typing import Dict, Optional

import torch
from torch import nn

class E2CNNEncoderExportedMaxPool(nn.Module):
    """Encoder using exported (standard PyTorch) E2CNN model with MaxPool downsampling.
    
    This class loads the exported C4 max-pool encoder in standard PyTorch format.
    The model uses PointwiseMaxPool for downsampling instead of strided convolutions,
    which preserves equivariance better.
    
    Feature dimensions: scale 1=64, 2=128, 4=256, 8=384, 16=512
    
    Usage:
        >>> encoder = E2CNNEncoderExportedMaxPool(
        ...     weights_path='encoder_weights/outdoor.pt'
        ... )
    """
    
    def __init__(
        self,
        weights_path: Optional[str] = None,
        freeze: bool = True,
        amp: bool = True,
        amp_dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        
        # Build the exported model structure
        print("[E2CNNEncoderExportedMaxPool] Building MaxPool E2CNN model...")
        self.exported_stages = self._build_model()
        
        # Load the exported weights
        if weights_path is not None:
            if os.path.exists(weights_path):
                state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
                # The state dict keys are like "0.0.weight", "0.0.bias", etc.
                # This matches our ModuleList structure directly.
                
                try:
                    missing, unexpected = self.exported_stages.load_state_dict(state_dict, strict=False)
                    print(f"[E2CNNEncoderExportedMaxPool] Loaded weights from {weights_path}")
                    if missing:
                        print(f"[E2CNNEncoderExportedMaxPool] Missing keys: {len(missing)}")
                        for k in missing[:10]:
                            print(f"    {k}")
                    if unexpected:
                        print(f"[E2CNNEncoderExportedMaxPool] Unexpected keys: {len(unexpected)}")
                        for k in unexpected[:10]:
                            print(f"    {k}")
                except Exception as e:
                    print(f"[E2CNNEncoderExportedMaxPool] Warning: Could not load weights: {e}")
            else:
                print(f"[E2CNNEncoderExportedMaxPool] Warning: weights_path not found: {weights_path}")
        
        # Create compatibility wrapper
        self.cnn = self._create_cnn_compat()
        
        self.amp = amp
        self.amp_dtype = amp_dtype
        self.freeze = freeze
        
        # Freeze if requested
        if freeze:
            self._freeze()
        
        print(f"[E2CNNEncoderExportedMaxPool] Model ready with {len(self.exported_stages)} stages")
    
    def _create_cnn_compat(self) -> nn.Module:
        """Create simple compatibility wrapper for device detection."""
        class _SimpleCNNCompat(nn.Module):
            def __init__(self, first_conv_weight):
                super().__init__()
                self.layers = nn.ModuleList([nn.Conv2d(3, 64, 1)])
                with torch.no_grad():
                    self.layers[0].weight.copy_(first_conv_weight[:64, :3, :1, :1])
        
        first_conv = None
        for module in self.exported_stages[0].modules():
            if isinstance(module, nn.Conv2d):
                first_conv = module
                break
        
        if first_conv is not None:
            return _SimpleCNNCompat(first_conv.weight)
        else:
            return _SimpleCNNCompat(torch.randn(64, 3, 1, 1))
    
    def _build_model(self) -> nn.ModuleList:
        """Build the model structure to match the exported weights.
        
        The structure follows the public exported C4 max-pool encoder:
        Stage 0: 2 convs, 64 channels, stride 1
        Stage 1: 2 convs, 128 channels, stride 2 (with PointwiseMaxPool)
        Stage 2: 4 convs, 256 channels, stride 2
        Stage 3: 4 convs, 384 channels, stride 2
        Stage 4: 4 convs, 512 channels, stride 2
        
        Note: We use ModuleList instead of Sequential to match the flat key structure
        in the exported weights (e.g., "0.0.weight" instead of "exported_stages.0.0.weight").
        
        IMPORTANT: All Conv2d layers use bias=False to match the exported weights.
        """
        stages = nn.ModuleList()
        
        # Stage 0: 64 channels, 2 convs
        stage0 = nn.Sequential()
        stage0.add_module('0', nn.BatchNorm2d(3, affine=True))  # First BN for RGB
        stage0.add_module('1', nn.ReLU(inplace=True))
        stage0.add_module('2', nn.Conv2d(3, 64, 3, padding=1, bias=False))
        stage0.add_module('3', nn.BatchNorm2d(64))
        stage0.add_module('4', nn.ReLU(inplace=True))
        stage0.add_module('5', nn.Conv2d(64, 64, 3, padding=1, bias=False))
        stages.append(stage0)
        
        # Stage 1: 128 channels, 2 convs, with maxpool downsampling
        stage1 = nn.Sequential()
        stage1.add_module('0', nn.BatchNorm2d(64))
        stage1.add_module('1', nn.ReLU(inplace=True))
        stage1.add_module('2', nn.Conv2d(64, 128, 3, padding=1, bias=False))
        stage1.add_module('3', nn.MaxPool2d(kernel_size=2, stride=2))  # PointwiseMaxPool equivalent
        stage1.add_module('4', nn.BatchNorm2d(128))
        stage1.add_module('5', nn.ReLU(inplace=True))
        stage1.add_module('6', nn.Conv2d(128, 128, 3, padding=1, bias=False))
        stages.append(stage1)
        
        # Stage 2: 256 channels, 4 convs, with maxpool downsampling
        stage2 = nn.Sequential()
        stage2.add_module('0', nn.BatchNorm2d(128))
        stage2.add_module('1', nn.ReLU(inplace=True))
        stage2.add_module('2', nn.Conv2d(128, 256, 3, padding=1, bias=False))
        stage2.add_module('3', nn.MaxPool2d(kernel_size=2, stride=2))
        stage2.add_module('4', nn.BatchNorm2d(256))
        stage2.add_module('5', nn.ReLU(inplace=True))
        stage2.add_module('6', nn.Conv2d(256, 256, 3, padding=1, bias=False))
        stage2.add_module('7', nn.BatchNorm2d(256))
        stage2.add_module('8', nn.ReLU(inplace=True))
        stage2.add_module('9', nn.Conv2d(256, 256, 3, padding=1, bias=False))
        stage2.add_module('10', nn.BatchNorm2d(256))
        stage2.add_module('11', nn.ReLU(inplace=True))
        stage2.add_module('12', nn.Conv2d(256, 256, 3, padding=1, bias=False))
        stages.append(stage2)
        
        # Stage 3: 384 channels, 4 convs, with maxpool downsampling
        stage3 = nn.Sequential()
        stage3.add_module('0', nn.BatchNorm2d(256))
        stage3.add_module('1', nn.ReLU(inplace=True))
        stage3.add_module('2', nn.Conv2d(256, 384, 3, padding=1, bias=False))
        stage3.add_module('3', nn.MaxPool2d(kernel_size=2, stride=2))
        stage3.add_module('4', nn.BatchNorm2d(384))
        stage3.add_module('5', nn.ReLU(inplace=True))
        stage3.add_module('6', nn.Conv2d(384, 384, 3, padding=1, bias=False))
        stage3.add_module('7', nn.BatchNorm2d(384))
        stage3.add_module('8', nn.ReLU(inplace=True))
        stage3.add_module('9', nn.Conv2d(384, 384, 3, padding=1, bias=False))
        stage3.add_module('10', nn.BatchNorm2d(384))
        stage3.add_module('11', nn.ReLU(inplace=True))
        stage3.add_module('12', nn.Conv2d(384, 384, 3, padding=1, bias=False))
        stages.append(stage3)
        
        # Stage 4: 512 channels, 4 convs, with maxpool downsampling
        stage4 = nn.Sequential()
        stage4.add_module('0', nn.BatchNorm2d(384))
        stage4.add_module('1', nn.ReLU(inplace=True))
        stage4.add_module('2', nn.Conv2d(384, 512, 3, padding=1, bias=False))
        stage4.add_module('3', nn.MaxPool2d(kernel_size=2, stride=2))
        stage4.add_module('4', nn.BatchNorm2d(512))
        stage4.add_module('5', nn.ReLU(inplace=True))
        stage4.add_module('6', nn.Conv2d(512, 512, 3, padding=1, bias=False))
        stage4.add_module('7', nn.BatchNorm2d(512))
        stage4.add_module('8', nn.ReLU(inplace=True))
        stage4.add_module('9', nn.Conv2d(512, 512, 3, padding=1, bias=False))
        stage4.add_module('10', nn.BatchNorm2d(512))
        stage4.add_module('11', nn.ReLU(inplace=True))
        stage4.add_module('12', nn.Conv2d(512, 512, 3, padding=1, bias=False))
        stages.append(stage4)
        
        return stages
    
    def _freeze(self):
        """Freeze all parameters and set to eval mode."""
        for param in self.parameters():
            param.requires_grad = False
        # Call nn.Module.eval directly to bypass train() override
        nn.Module.train(self, False)
        print("[E2CNNEncoderExportedMaxPool] Model frozen")
    
    def train(self, mode: bool = True):
        """Override train to keep model frozen if freeze=True."""
        if self.freeze and mode:
            # If trying to set to train mode while frozen, keep in eval
            return self
        return super().train(mode)
    
    def forward(self, x, upsample=False):
        """Forward pass returning feature pyramid.
        
        Args:
            x: Input tensor (B, 3, H, W), normalized with ImageNet normalization
            upsample: Ignored (for API compatibility)
            
        Returns:
            Dict[int, Tensor]: Feature pyramid {scale: (B, C, H/scale, W/scale)}
                - scale 1: (B, 64, H, W)
                - scale 2: (B, 128, H/2, W/2)
                - scale 4: (B, 256, H/4, W/4)
                - scale 8: (B, 384, H/8, W/8)
                - scale 16: (B, 512, H/16, W/16)
        """
        with torch.no_grad() if self.freeze else torch.enable_grad():
            feature_pyramid = {}
            scale_map = [1, 2, 4, 8, 16]
            
            out = x
            for i, stage in enumerate(self.exported_stages):
                out = stage(out)
                if i < len(scale_map):
                    scale = scale_map[i]
                    # Normalize features to prevent float16 overflow in decoder
                    # The MaxPool encoder outputs very large values in shallow scales
                    # which causes local_correlation to overflow in float16 AMP mode
                    feat = out.contiguous()
                    # Per-channel normalization: subtract mean, divide by std
                    # Use a small epsilon to avoid division by zero
                    feat_mean = feat.mean(dim=(2, 3), keepdim=True)
                    feat_std = feat.std(dim=(2, 3), keepdim=True) + 1e-6
                    feat = (feat - feat_mean) / feat_std
                    feature_pyramid[scale] = feat
            
            return feature_pyramid
    
    def get_feature_dims(self) -> Dict[int, int]:
        """Return feature dimensions at each scale."""
        return {
            1: 64,
            2: 128,
            4: 256,
            8: 384,
            16: 512,
        }
