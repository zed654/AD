"""
SSR_transformer_pt.py: PyTorch-only version (충청도 사투리 주석 포함)
원본: SSR_transformer.py (MMCV/mmdet3d 기반)
이 버전은 외부 의존성 없이 PyTorch만 사용함
형님, 이 파일은 연구/실험용이니 참고해서 쓰셔유!
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ---- 아래는 순수 PyTorch 버전으로 필요한 유틸/레이어 직접 구현 ----
def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)

# ---- 이하 Transformer 구조 (PyTorch-only) ----
class DummyTransformerLayer(nn.Module):
    def __init__(self, embed_dims):
        super().__init__()
        self.linear = nn.Linear(embed_dims, embed_dims)
    def forward(self, x, *args, **kwargs):
        return self.linear(x)

class MapDetectionTransformerDecoderPT(nn.Module):
    """Implements the decoder in DETR3D transformer (PyTorch-only)."""
    def __init__(self, num_layers=1, embed_dims=256, return_intermediate=False):
        super().__init__()
        self.layers = nn.ModuleList([DummyTransformerLayer(embed_dims) for _ in range(num_layers)])
        self.return_intermediate = return_intermediate
    def forward(self, query, reference_points=None, reg_branches=None, key_padding_mask=None, *args, **kwargs):
        output = query
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            output = layer(output)
            if reg_branches is not None:
                tmp = reg_branches[lid](output)
                new_reference_points = torch.zeros_like(reference_points)
                new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points[..., :2])
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)
        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)
        return output, reference_points

class SSRPerceptionTransformerPT(nn.Module):
    """Implements the Detr3D transformer (PyTorch-only)."""
    def __init__(self, embed_dims=256, num_feature_levels=4, num_cams=6):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.level_embeds = nn.Parameter(torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.cams_embeds = nn.Parameter(torch.Tensor(self.num_cams, self.embed_dims))
        self.reference_points = nn.Linear(self.embed_dims, 3)
        self.map_reference_points = nn.Linear(self.embed_dims, 2)
        self.can_bus_mlp = nn.Sequential(
            nn.Linear(18, self.embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims // 2, self.embed_dims),
            nn.ReLU(inplace=True),
            nn.LayerNorm(self.embed_dims)
        )
    def forward(self, *args, **kwargs):
        # 형님, 여기는 forward 함수여유. 실제 논리 구현은 연구 목적에 맞게 수정해서 쓰셔유!
        return {} 