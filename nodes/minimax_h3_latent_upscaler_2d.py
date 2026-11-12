"""
Minimax H3 Latent Upscaler - ComfyUI Inference Node (2D)
- [架构对齐 v12] 与 Q-latent-2d-v12.py 训练结构逐键一致：EnhancedScaleEmbedding(相位去 π、fp32 计算)
  / ResBlockEmb(无 SE) / TemporalConv(dwconv+pwconv, k=5, 每 2 个 block 插一次) / 无 CoordConv
- [数值对齐 v12] 中段上采样用 F.interpolate(mode='bilinear', align_corners=False)，且 forward 不含
  「输出 + 输入上采样」全局残差支路 —— 两者都跟训练一致。旧版 LocalEnsemble + GlobalSkip 会让权重
  落到分布外（实测表现：画面晃动、清晰度下降）。旧 Ultimate 架构可用 USE_LOCAL_ENSEMBLE 回退。
- [工程增强] Temporal Chunking：按「时间感受野(±24帧)」给上下文，分块结果与整段一次前向
  逐位一致（旧版按 kernel 长度给上下文，每 32 帧接缝会差 0.14 → 闪烁）
- [工程增强] Zero-Copy 加载 + soft_empty_cache + force_unload 显存优化
- [工程增强] 像素级 32 对齐，解决底部光带；支持 ROCm / 子文件夹 / 多模式缩放
- [自动检测] 从权重自动推断架构参数 (embed_dim / CoordConv / SE / LargeKernel / Temporal 键名)
- [加载校验] strict=False 下缺失键直接报错，避免静默加载出一个错误模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import gc
import math
import folder_paths
import re
from einops import rearrange
from enum import Enum
from typing import TypedDict

try:
    import comfy.model_management as mm
    HAS_COMFY_MM = True
except ImportError:
    HAS_COMFY_MM = False

try:
    from comfy_api.latest import ComfyExtension, io
    from comfy.nested_tensor import NestedTensor
    from typing_extensions import override
    USE_NEW_API = True
except ImportError:
    USE_NEW_API = False
    class io:
        class ComfyNode: pass
        class Schema: pass
        class NodeOutput: pass
        class AnyType:
            @staticmethod
            def Input(*args, **kwargs): return args[0] if args else "ANY"
            @staticmethod
            def Output(*args, **kwargs): return args[0] if args else "ANY"
        class Combo:
            @staticmethod
            def Input(*args, **kwargs): return args[0] if args else "COMBO"
        class Float:
            @staticmethod
            def Input(*args, **kwargs): return "FLOAT"
        class Int:
            @staticmethod
            def Input(*args, **kwargs): return "INT"
        class Boolean:
            @staticmethod
            def Input(*args, **kwargs): return "BOOLEAN"
        class DynamicCombo:
            @staticmethod
            def Input(*args, **kwargs): return args[0] if args else "DYNAMIC"
            class Option: pass
    class ComfyExtension: pass
    class NestedTensor:
        def __init__(self, tensors):
            self.tensors = list(tensors)
            self.is_nested = True
    def override(func): return func

# ==========================================
# Register model folder
# ==========================================
_LATENT_UPSCALE_FOLDER = "latent_upscale_models"
if _LATENT_UPSCALE_FOLDER not in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path(
        _LATENT_UPSCALE_FOLDER,
        os.path.join(folder_paths.models_dir, _LATENT_UPSCALE_FOLDER)
    )

VAE_DOWNSAMPLE = 16

# v12 训练用 F.interpolate(bilinear) 做中段上采样，且 forward 里没有全局残差支路。
# 推理必须与训练一致，否则输入分布不一致（表现为晃动/发糊）。
# 仅当加载旧 Ultimate + LocalEnsemble 架构权重时才置 True。
USE_LOCAL_ENSEMBLE = False

# 上采样路径自证：把「这次实际走的是学习型还是回退到插值」打到控制台。
# 背景：_upsample_mid 在倍率不是整数 k 时会静默回退到 bilinear，从外部无法分辨，
#       容易误判成「学习型效果和插值一样」。同一倍率只打一次，避免时间分块刷屏。
_UP_PATH_LOGGED = set()


def _log_up_path(desc, s):
    key = round(float(s), 4)
    if key in _UP_PATH_LOGGED:
        return
    _UP_PATH_LOGGED.add(key)
    print(f"[MinimaxH3-2D] 上采样路径: {desc} | 目标倍率 {float(s):.4f}")

# 时间分块的块长（帧）。上下文额外需要 ±感受野（v12 = 24 帧），单块前向实际帧数为
# chunk + 48。显存不够可下调（如 16），代价是多跑几块。
CHUNK_FRAMES = 32

# ==========================================
# Minimax H3 latent normalization stats (24 channels)
# ==========================================
LATENTS_MEAN = [
    0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075,
    -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975,
    -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
    -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543,
    -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279,
    -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264
]
LATENTS_STD = [
    1.2223774194717407, 1.2767263650894165, 1.6831774711608887, 1.7549455165863037,
    1.5636216402053833, 2.194143533706665, 0.9653137922286987, 1.0569885969161987,
    0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647,
    0.7996809482574463, 0.44988900423049927, 0.7197399735450745, 0.6936293244361877,
    2.961095094680786, 2.7694199085235596, 3.0496184825897217, 2.1088054180145264,
    3.276226282119751, 3.1627357006073, 2.2816812992095947, 2.6127843856811523
]

def _make_norm_tensors(device, dtype):
    mean = torch.tensor(LATENTS_MEAN, dtype=dtype, device=device).view(1, -1, 1, 1, 1)
    std = torch.tensor(LATENTS_STD, dtype=dtype, device=device).view(1, -1, 1, 1, 1)
    return mean, std

# ==========================================
# ROCm / Device Helper Functions
# ==========================================
def _is_rocm_build():
    return getattr(torch.version, "hip", None) is not None

def _resolve_device(backend):
    if backend == "cpu":
        return torch.device("cpu")
    if backend == "rocm":
        if not _is_rocm_build():
            raise RuntimeError("ROCm was selected, but this PyTorch build has no HIP/ROCm support.")
        if not torch.cuda.is_available():
            raise RuntimeError("ROCm was selected, but PyTorch cannot access an AMD GPU.")
        return torch.device("cuda")
    if backend == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise ValueError(f"Unsupported device backend: {backend}")

def _backend_label(device):
    if device.type == "cuda" and _is_rocm_build():
        return f"ROCm/HIP {torch.version.hip}"
    if device.type == "cuda":
        return f"CUDA {getattr(torch.version, 'cuda', None) or 'unknown'}"
    return "CPU"

# ==========================================
# Local Ensemble Upsample (LIIF Core)
# ==========================================
def local_ensemble_upsample(feat, target_size):
    """
    Local Ensemble 上采样：消除任意倍率下的锯齿 (LIIF 核心)
    feat: [N, C, H, W]
    target_size: (H_new, W_new)
    """
    N, C, H, W = feat.shape
    H_new, W_new = target_size

    grid_y = torch.linspace(-1, 1, H_new, device=feat.device, dtype=feat.dtype)
    grid_x = torch.linspace(-1, 1, W_new, device=feat.device, dtype=feat.dtype)
    grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing='ij')
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(N, -1, -1, -1)

    cell_W = 1.0 / W_new if W_new > 1 else 0.0
    cell_H = 1.0 / H_new if H_new > 1 else 0.0

    preds = [F.grid_sample(feat, grid, mode='bilinear', padding_mode='border', align_corners=False)]

    for dy in [-cell_H, cell_H]:
        for dx in [-cell_W, cell_W]:
            grid_shift = grid.clone()
            grid_shift[..., 0] += dx
            grid_shift[..., 1] += dy
            preds.append(F.grid_sample(feat, grid_shift, mode='bilinear', padding_mode='border', align_corners=False))

    return torch.stack(preds, dim=0).mean(dim=0)

# ==========================================
# Ultimate Architecture Components
# ==========================================
def normalization(channels):
    return nn.GroupNorm(32 if channels >= 32 else channels, channels)

def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

class FourierScaleEmbedding2D(nn.Module):
    """scale_h / scale_w 的 Fourier 编码，可选拼接【锐度档位】的 Fourier 编码。

    【2026-11-05 新增 tier_dim】与训练脚本 Q-latent-2d-v15 的 EnhancedScaleEmbedding 逐位对齐：
      输入 = [ sin/cos(scale_h) *16 频 → 32 维 ]
             [ sin/cos(scale_w) *16 频 → 32 维 ]
             [ sin/cos((tier-1)) *16 频 → 32 维 ]   ← tier_dim=32 时才存在
    tier_dim=0 时输入仍是 4*freq（64 维），**旧权重行为完全不变**；
    tier_dim=32 时是 96 维（e25 及之后带档位的权重）。
    """

    def __init__(self, embed_dim, num_frequencies=16, tier_dim=0):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.tier_dim = int(tier_dim)
        in_dim = 4 * num_frequencies + self.tier_dim
        self.linear = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        # 仅对"新建模型"有意义：档位通道零初始化 → 等价于无档位。加载权重后这些列会被覆盖。
        if self.tier_dim > 0:
            with torch.no_grad():
                self.linear[0].weight[:, 4 * num_frequencies:].zero_()

    def forward(self, scale_h, scale_w, tier=None):
        # 【对齐 v12】与训练脚本 EnhancedScaleEmbedding 完全一致：
        #   1) 支持 float/int 输入（训练里 scale_h 是 Python float）
        #   2) 相位不乘 π —— 乘 π 后 scale*freq*π 在「倍率相差偶数整数」时恰为 π 的整数倍，
        #      sin 恒 0、cos 恒 ±1，高频分量整体塌缩（1.2x/3.2x、2.0x/4.0x 无法区分）
        #   3) 三角函数在 fp32 下计算，算完再回到权重 dtype
        dev = self.linear[0].weight.device
        if isinstance(scale_h, (float, int)): scale_h = torch.tensor([float(scale_h)], device=dev)
        if isinstance(scale_w, (float, int)): scale_w = torch.tensor([float(scale_w)], device=dev)
        scale_h = scale_h.to(device=dev, dtype=torch.float32)
        scale_w = scale_w.to(device=dev, dtype=torch.float32)
        if scale_h.dim() == 0: scale_h = scale_h.unsqueeze(0)
        if scale_w.dim() == 0: scale_w = scale_w.unsqueeze(0)

        freqs = 2.0 ** torch.arange(self.num_frequencies, device=dev).float()
        emb_h = torch.cat([torch.sin(scale_h.unsqueeze(1) * freqs.unsqueeze(0)),
                           torch.cos(scale_h.unsqueeze(1) * freqs.unsqueeze(0))], dim=1)
        emb_w = torch.cat([torch.sin(scale_w.unsqueeze(1) * freqs.unsqueeze(0)),
                           torch.cos(scale_w.unsqueeze(1) * freqs.unsqueeze(0))], dim=1)

        emb = torch.cat([emb_h, emb_w], dim=1)

        if self.tier_dim > 0:
            # 与训练脚本一致：相位用 (tier-1)，档位 1 是规范零点。
            # tier=None 表示"调用方没指定" → 按 t1 处理（相位 0），
            # 不能按 0 处理（那会变成 tier=0，是训练里根本不存在的档位，属于分布外输入）。
            if tier is None:
                t = torch.ones(1, device=dev)
            elif isinstance(tier, (float, int)):
                t = torch.tensor([float(tier)], device=dev)
            else:
                t = tier.to(device=dev, dtype=torch.float32)
                if t.dim() == 0: t = t.unsqueeze(0)
            t = t - 1.0
            emb = torch.cat([emb,
                             torch.sin(t.unsqueeze(1) * freqs.unsqueeze(0)),
                             torch.cos(t.unsqueeze(1) * freqs.unsqueeze(0))], dim=1)

        # 拼接后转回输入 dtype（如 fp16），以匹配 self.linear 的权重 dtype
        return self.linear(emb.to(self.linear[0].weight.dtype))

class CoordConv2D(nn.Module):
    def forward(self, x):
        B, C, H, W = x.shape
        yc = torch.linspace(-1, 1, H, device=x.device, dtype=x.dtype).view(1, 1, H, 1).expand(B, 1, H, W)
        xc = torch.linspace(-1, 1, W, device=x.device, dtype=x.dtype).view(1, 1, 1, W).expand(B, 1, H, W)
        return torch.cat([x, yc, xc], dim=1)

class SEBlock(nn.Module):
    def __init__(self, ch, reduction=8):
        super().__init__()
        mid = max(ch // reduction, 16)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, mid),
            nn.SiLU(True),
            nn.Linear(mid, ch),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(x).unsqueeze(-1).unsqueeze(-1)

class LargeKernelBlock(nn.Module):
    def __init__(self, ch, ks=7):
        super().__init__()
        self.norm = normalization(ch)
        self.dw = nn.Conv2d(ch, ch, ks, padding=ks // 2, groups=ch)
        self.pw = nn.Conv2d(ch, ch, 1)
        nn.init.zeros_(self.pw.weight)
        nn.init.zeros_(self.pw.bias)

    def forward(self, x):
        return x + self.pw(self.dw(F.silu(self.norm(x))))

class PostUpsampleRefine(nn.Module):
    def __init__(self, ch, n=2):
        super().__init__()
        layers = []
        for i in range(n):
            layers.append(nn.Conv2d(ch, ch, 3, padding=1))
            if i < n - 1:
                layers.append(nn.SiLU(True))
        self.refine = nn.Sequential(*layers)
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    def forward(self, x):
        return x + self.refine(x)

class MultiScaleTemporalConv(nn.Module):
    def __init__(self, ch, kernel_sizes=(3, 5, 9)):
        super().__init__()
        self.norm = normalization(ch)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(ch, ch, (ks, 1, 1), padding=(ks // 2, 0, 0), groups=ch),
                nn.Conv3d(ch, ch, 1)
            ) for ks in kernel_sizes
        ])
        self.fusion = nn.Conv3d(ch * len(kernel_sizes), ch, 1)
        nn.init.zeros_(self.fusion.weight)
        nn.init.zeros_(self.fusion.bias)

    def forward(self, x):
        B, C, T, H, W = x.shape
        h = rearrange(x, "b c t h w -> (b t) c h w")
        h = self.norm(h)
        h = rearrange(h, "(b t) c h w -> b c t h w", b=B, t=T)
        return x + self.fusion(torch.cat([br(F.silu(h)) for br in self.branches], dim=1))

class TemporalConv(nn.Module):
    # 【对齐 v12】参数名必须是 dwconv / pwconv，否则 strict=False 下加载不到权重
    # （dwconv 是随机初始化，静默 mismatch 会直接毁掉输出）
    def __init__(self, ch, ks=5):
        super().__init__()
        self.norm = normalization(ch)
        self.dwconv = nn.Conv3d(ch, ch, (ks, 1, 1), padding=(ks // 2, 0, 0), groups=ch)
        self.pwconv = nn.Conv3d(ch, ch, 1)
        nn.init.zeros_(self.pwconv.weight)
        nn.init.zeros_(self.pwconv.bias)

    def forward(self, x):
        B, C, T, H, W = x.shape
        h = rearrange(x, "b c t h w -> (b t) c h w")
        h = self.norm(h)
        h = rearrange(h, "(b t) c h w -> b c t h w", b=B, t=T)
        return x + self.pwconv(self.dwconv(F.silu(h)))

class ResBlockEmb(nn.Module):
    def __init__(self, channels, emb_channels, dropout=0, out_channels=None, use_se=True, se_reduction=8):
        super().__init__()
        self.out_channels = out_channels or channels
        self.in_layers = nn.Sequential(
            normalization(channels), nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, 2 * self.out_channels),
        )
        self.out_norm = normalization(self.out_channels)
        self.out_layers = nn.Sequential(
            nn.SiLU(), nn.Dropout(p=dropout),
            zero_module(nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)),
        )
        self.skip = (
            nn.Conv2d(channels, self.out_channels, 1)
            if self.out_channels != channels else nn.Identity()
        )
        self.se = SEBlock(self.out_channels, se_reduction) if use_se else None

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        h = self.out_norm(h) * (1 + scale) + shift
        h = self.out_layers(h)
        if self.se is not None:
            h = self.se(h)
        return self.skip(x) + h

# ==========================================
# Ultimate LatentResizer + Video Wrapper
# ==========================================
# ====================== 学习型上采样支路（2026-09-21，与训练脚本 Q-latent-2d-v16.py 对齐） ======================
# 与插值上采样【严格互斥】：learned_up 存在时输出完全由 PixelShuffle 给出，不叠加 bilinear。
# 训练侧用双线性核初始化；推理侧只构建结构并加载权重（若权重无 learned_up.* 键则不构建，保证旧权重兼容）。
def _bilinear_taps(k):
    """k 个长度为 3 的 1D 插值 tap（对应输入索引 i-1, i, i+1），align_corners=False 口径。"""
    taps = []
    for a in range(k):
        t = (a + 0.5) / k - 0.5
        lo = math.floor(t)
        frac = t - lo
        v = torch.zeros(3)
        v[lo + 1] = 1.0 - frac
        v[lo + 2] = frac
        taps.append(v)
    return taps


class PSUpsample(nn.Module):
    """单倍率 PixelShuffle(k) 学习型上采样（独占式，不与插值叠加）。"""

    def __init__(self, channels, k=2):
        super().__init__()
        self.k = int(k)
        assert self.k >= 2, f"PS 倍数必须 >= 2，收到 {k}"
        # padding_mode 必须 'replicate'：与训练侧一致，且使初始化输出与
        # F.interpolate(bilinear) 的边界 clamp 行为对齐（'zeros' 会让边缘系统性偏低）
        self.conv = nn.Conv2d(channels, self.k * self.k * channels, 3, padding=1,
                              padding_mode='replicate')
        self.ps = nn.PixelShuffle(self.k)
        self._init_bilinear_kernel()

    def _init_bilinear_kernel(self):
        """双线性核初始化（与训练侧完全一致）。正常会被权重覆盖，仅作未加载时的合理兜底。"""
        k, C = self.k, self.conv.in_channels
        taps = _bilinear_taps(k)
        with torch.no_grad():
            w = torch.zeros_like(self.conv.weight)
            for a in range(k):
                for b in range(k):
                    ker = torch.outer(taps[a], taps[b])
                    idx = a * k + b
                    for c in range(C):
                        w[c * k * k + idx, c] = ker
            self.conv.weight.copy_(w)
            if self.conv.bias is not None:
                nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.ps(self.conv(x))


class PSUpsampleBank(nn.Module):
    """一组 PS 支路，按目标倍率精确选 k。"""

    def __init__(self, channels, ks):
        super().__init__()
        self.ks = sorted({int(k) for k in ks})
        assert self.ks and all(k >= 2 for k in self.ks), f"learned_upsample_ks 非法: {ks}"
        self.branches = nn.ModuleDict({str(k): PSUpsample(channels, k) for k in self.ks})

    def select_k(self, scale):
        for k in self.ks:                       # 精确匹配优先
            if abs(k - scale) <= 1e-3:
                return k
        for k in self.ks:                       # 兜底：最小的 k > scale
            if k > scale:
                return k
        return self.ks[-1]

    def forward(self, x, scale):
        return self.branches[str(self.select_k(scale))](x)


class LatentResizer(nn.Module):
    """2D Ultimate Backbone with Local Ensemble"""
    def __init__(self, in_channels=24, in_blocks=8, out_blocks=8, channels=768,
                 dropout=0.1, attn=False, use_dual_scale=True, use_coord=True,
                 use_se=True, se_reduction=8, use_large_kernel=True, large_kernel_size=7,
                 large_kernel_every=3, use_post_refine=True, post_refine_layers=2,
                 dual_embed_freq=16, embed_dim=128, learned_upsample_ks=None, tier_dim=0):
        super().__init__()
        # 【对齐 v12】EMBED_DIM=128，必须由权重推断后传入，不能写死
        self.use_dual_scale = use_dual_scale
        self.use_coord = use_coord
        # 【2026-11-05】锐度档位通道宽度：0 = 旧权重无档位，32 = 带档位权重（e25 及之后）
        self.tier_dim = int(tier_dim)

        if use_dual_scale:
            self.embed = FourierScaleEmbedding2D(embed_dim, dual_embed_freq, tier_dim=self.tier_dim)
        else:
            self.embed = nn.Sequential(nn.Linear(1, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))

        conv_in_ch = in_channels + 2 if use_coord else in_channels
        self.coord_conv = CoordConv2D() if use_coord else None
        self.conv_in = nn.Conv2d(conv_in_ch, channels, 3, padding=1)

        # 【2026-09-21】学习型上采样支路（与训练脚本 v16 同款）。
        # 仅当权重里存在 learned_up.* 时才构建（由 _detect_arch 推断），否则为 None → 纯插值，
        # 保证旧权重仍能被 load_state_dict 严格校验通过。
        self.learned_up = PSUpsampleBank(channels, learned_upsample_ks) if learned_upsample_ks else None

        self.in_blocks = nn.ModuleList()
        for b in range(in_blocks):
            self.in_blocks.append(ResBlockEmb(channels, embed_dim, dropout, use_se=use_se, se_reduction=se_reduction))
            if use_large_kernel and b % large_kernel_every == (large_kernel_every - 1):
                self.in_blocks.append(LargeKernelBlock(channels, large_kernel_size))

        self.out_blocks = nn.ModuleList()
        for b in range(out_blocks):
            self.out_blocks.append(ResBlockEmb(channels, embed_dim, dropout, use_se=use_se, se_reduction=se_reduction))
            if use_large_kernel and b % large_kernel_every == (large_kernel_every - 1):
                self.out_blocks.append(LargeKernelBlock(channels, large_kernel_size))

        self.post_up = PostUpsampleRefine(channels, post_refine_layers) if use_post_refine else None
        self.norm_out = normalization(channels)
        self.conv_out = nn.Conv2d(channels, in_channels, 3, padding=1)

    def _upsample_mid(self, x, size):
        """中段上采样：学习型 PixelShuffle 与插值 bilinear【互斥二选一】。

        - learned_up 存在且目标倍率精确等于某个 k → 完全用学习型输出（不叠加任何插值）
        - 否则（无支路 / 倍率非整数，如 1.5x）→ 回退 F.interpolate(bilinear)
        注意：训练脚本只在「倍率精确等于 k」的桶上训练学习型支路，推理端非整数倍率
        必须回退插值，否则会因尺寸不符而裁掉内容。
        """
        if self.learned_up is not None:
            s = min(size[0] / x.shape[-2], size[1] / x.shape[-1])
            k = self.learned_up.select_k(s)
            if abs(k - s) <= 1e-3 and x.shape[-2] * k >= size[0] and x.shape[-1] * k >= size[1]:
                y = self.learned_up(x, s)
                if y.shape[-2] != size[0] or y.shape[-1] != size[1]:
                    y = y[..., :size[0], :size[1]]      # 仅裁剪对齐，不引入插值
                _log_up_path(f"学习型 PixelShuffle(k={k})", s)
                return y
            _log_up_path(f"回退插值：目标倍率不匹配整数 k={self.learned_up.ks}（学习型支路未生效）", s)
            return local_ensemble_upsample(x, size) if USE_LOCAL_ENSEMBLE else \
                F.interpolate(x, size=size, mode="bilinear")
        s = min(size[0] / x.shape[-2], size[1] / x.shape[-1])
        _log_up_path("回退插值：该权重不含学习型支路（旧权重）", s)
        return local_ensemble_upsample(x, size) if USE_LOCAL_ENSEMBLE else \
            F.interpolate(x, size=size, mode="bilinear")

    def forward(self, x, scale_h=None, scale_w=None, target_hw=None, tier=None):
        if target_hw is not None:
            size = target_hw
        elif scale_h is not None and scale_w is not None:
            size = (int(round(x.shape[-2] * scale_h)), int(round(x.shape[-1] * scale_w)))
        else:
            return x

        if size == x.shape[-2:]:
            return x

        if scale_h is None: scale_h = size[0] / x.shape[-2]
        if scale_w is None: scale_w = size[1] / x.shape[-1]

        # 【对齐 v12】嵌入输入是原始倍率（不减 1）
        if self.use_dual_scale:
            emb = self.embed(scale_h, scale_w, tier=tier)
        else:
            emb = self.embed((scale_h + scale_w) / 2.0, (scale_h + scale_w) / 2.0)

        if self.use_coord:
            x = self.coord_conv(x)
        x = self.conv_in(x)

        for b in self.in_blocks:
            if isinstance(b, ResBlockEmb):
                x = b(x, emb.expand(x.shape[0], -1))
            else:
                x = b(x)

        # 【2026-09-21】中段上采样：学习型 / 插值【互斥二选一】（见 _upsample_mid）
        x = self._upsample_mid(x, size)
        if self.post_up:
            x = self.post_up(x)

        for b in self.out_blocks:
            if isinstance(b, ResBlockEmb):
                x = b(x, emb.expand(x.shape[0], -1))
            else:
                x = b(x)

        return self.conv_out(F.silu(self.norm_out(x)))


class VideoLatentResizer(nn.Module):
    """5D Ultimate Wrapper with Temporal Blocks + Global Skip Local Ensemble"""
    def __init__(self, in_channels=24, in_blocks=8, out_blocks=8, channels=768,
                 dropout=0.1, attn=False, temporal_every=2, use_multiscale_temporal=True,
                 temporal_kernel_sizes=(3, 5, 9), temporal_kernel=5, **resizer_kwargs):
        super().__init__()
        self.resizer = LatentResizer(
            in_channels=in_channels, in_blocks=in_blocks, out_blocks=out_blocks,
            channels=channels, dropout=dropout, attn=attn, **resizer_kwargs
        )
        self.temporal_every = temporal_every
        self.temporal_blocks = nn.ModuleList()

        if temporal_every > 0:
            if use_multiscale_temporal:
                self.temporal_blocks.append(MultiScaleTemporalConv(channels, temporal_kernel_sizes))
                self.temporal_blocks.append(MultiScaleTemporalConv(channels, temporal_kernel_sizes))
            else:
                self.temporal_blocks.append(TemporalConv(channels, temporal_kernel))
                self.temporal_blocks.append(TemporalConv(channels, temporal_kernel))

    def forward(self, x, scale=None, target_hw=None, enable_chunking=True,
                chunk_frames=None, tier=None):
        B, C, T, H, W = x.shape

        if target_hw is not None:
            size = target_hw
        elif scale is not None:
            size = (int(round(H * scale)), int(round(W * scale)))
        else:
            size = (H, W)

        # Temporal Chunking for long videos
        # 【上下文宽度 = 时间感受野，不是单个卷积核】
        # temporal conv 在 in_blocks / out_blocks 里各插入 (blocks // temporal_every) 次，
        # 每次影响 ±(k//2) 帧 → v12（k=5、每 2 层插一次、12+12 层）总感受野 = 2*(6+6) = 24 帧。
        # 按 kernel 长度（5）给上下文会少 19 帧，块边界末段帧的邻居被当成 0 填充 → 接缝闪烁。
        tk = 0
        for b in self.temporal_blocks:
            if isinstance(b, TemporalConv):
                tk = b.dwconv.weight.shape[2]
                break
            elif isinstance(b, MultiScaleTemporalConv):
                tk = max(br[0].weight.shape[2] for br in b.branches)
                break

        radius = max(tk // 2, 1)
        n_temporal = sum(1 for i in range(len(self.resizer.in_blocks)) if i % self.temporal_every == 0)
        n_temporal += sum(1 for i in range(len(self.resizer.out_blocks)) if i % self.temporal_every == 0)
        overlap = radius * max(n_temporal, 1)
        chunk = max(1, int(chunk_frames)) if chunk_frames else CHUNK_FRAMES

        if not enable_chunking or T <= chunk or len(self.temporal_blocks) == 0:
            return self._forward_full(x, size, tier=tier)

        # Chunked inference：每块只负责输出 [seg_start, seg_end)，输入额外带 ±overlap 的**真实**帧。
        # 不补显式 0 帧：显式 0 帧经 GroupNorm/bias 后 ≠「不存在的帧」（conv 的隐式填充才是真 0 贡献），
        # 会在视频首尾引入偏差。分段边缘交给 Conv3d 自带 padding，行为与训练时完全一致。
        # 上下文给足 overlap 后，分块结果与整段一次前向只差浮点噪声，纯粹是省显存。
        out_full = torch.zeros(B, C, T, size[0], size[1], device=x.device, dtype=x.dtype)

        start = 0
        while start < T:
            seg_start = start
            seg_end = min(T, start + chunk)
            lo = max(0, seg_start - overlap)          # 左侧多给 overlap 帧真实上下文
            hi = min(T, seg_end + overlap)            # 右侧同理

            seg = x[:, :, lo:hi].contiguous()
            seg_out = self._forward_full(seg, size, tier=tier)

            s0 = seg_start - lo                       # seg 内 s0 帧 = 视频第 seg_start 帧
            out_full[:, :, seg_start:seg_end] = seg_out[:, :, s0:s0 + (seg_end - seg_start)]

            start += chunk
            del seg, seg_out
            if start % (chunk * 4) == 0:
                gc.collect()

        return out_full

    def _forward_full(self, x, size, tier=None):
        """与 v12 训练脚本 VideoLatentResizer.forward 逐行对齐（无全局残差支路）。"""
        B, C, T, H, W = x.shape
        x_flat = rearrange(x, "b c t h w -> (b t) c h w")

        scale_h = size[0] / H
        scale_w = size[1] / W

        if len(self.temporal_blocks) == 0:
            out = self.resizer(x_flat, scale_h=scale_h, scale_w=scale_w, target_hw=size, tier=tier)
            return rearrange(out, "(b t) c h w -> b c t h w", b=B, t=T)

        if self.resizer.use_dual_scale:
            emb = self.resizer.embed(scale_h, scale_w, tier=tier)
        else:
            emb = self.resizer.embed((scale_h + scale_w) / 2.0, (scale_h + scale_w) / 2.0)

        feat = self.resizer.coord_conv(x_flat) if self.resizer.use_coord else x_flat
        out = self.resizer.conv_in(feat)

        for i, block in enumerate(self.resizer.in_blocks):
            if isinstance(block, ResBlockEmb):
                out = block(out, emb.expand(B * T, -1))
            else:
                out = block(out)
            if i % self.temporal_every == 0 and len(self.temporal_blocks) > 0:
                out_3d = rearrange(out, "(b t) c h w -> b c t h w", b=B, t=T)
                out_3d = self.temporal_blocks[0](out_3d)
                out = rearrange(out_3d, "b c t h w -> (b t) c h w")

        # 【2026-09-21】中段上采样：学习型 / 插值【互斥二选一】，训练与推理必须同款
        out = self.resizer._upsample_mid(out, size)
        if self.resizer.post_up:
            out = self.resizer.post_up(out)

        for i, block in enumerate(self.resizer.out_blocks):
            if isinstance(block, ResBlockEmb):
                out = block(out, emb.expand(B * T, -1))
            else:
                out = block(out)
            if i % self.temporal_every == 0 and len(self.temporal_blocks) > 1:
                out_3d = rearrange(out, "(b t) c h w -> b c t h w", b=B, t=T)
                out_3d = self.temporal_blocks[1](out_3d)
                out = rearrange(out_3d, "b c t h w -> (b t) c h w")

        out = self.resizer.norm_out(out)
        out = F.silu(out)
        out = self.resizer.conv_out(out)
        return rearrange(out, "(b t) c h w -> b c t h w", b=B, t=T)


# ==========================================
# Model Loading & Auto-Detection
# ==========================================
MODEL_CACHE = {}

def get_models_dir():
    return folder_paths.get_folder_paths(_LATENT_UPSCALE_FOLDER)[0]

def scan_models():
    names = [
        name for name in folder_paths.get_filename_list(_LATENT_UPSCALE_FOLDER)
        if os.path.splitext(name)[1].lower() in (".pth", ".safetensors")
    ]
    return names if names else [f"(place models in: {get_models_dir()})"]

def _load_raw_sd(path):
    if path.endswith('.safetensors'):
        try:
            from safetensors import safe_open
            with safe_open(path, framework="pt", device="cpu") as f:
                sd = {k: f.get_tensor(k) for k in f.keys()}
        except ImportError:
            from safetensors.torch import load_file
            sd = load_file(path, device='cpu')
    else:
        sd = torch.load(path, map_location='cpu', weights_only=False)

    if isinstance(sd, dict) and 'model' in sd:
        sd = sd['model']
    sd = {k: v.to(torch.float16) if v.dtype == torch.float8_e4m3fn else v
          for k, v in sd.items()}
    return sd

def _extract_upscaler_sd(sd):
    if any(k.startswith("upscaler.") for k in sd):
        return {k[len("upscaler."):]: v for k, v in sd.items() if k.startswith("upscaler.")}
    return sd

def _detect_arch(sd):
    """Auto-detect Ultimate architecture parameters from state_dict"""
    cfg = {
        "in_channels": 24, "in_blocks": 8, "out_blocks": 8, "channels": 768,
        "dropout": 0.1, "attn": False, "temporal_every": 2,
        "use_dual_scale": True, "use_coord": True, "use_se": True, "se_reduction": 8,
        "use_large_kernel": True, "large_kernel_size": 7, "large_kernel_every": 3,
        "use_post_refine": True, "post_refine_layers": 2,
        "embed_dim": 128, "dual_embed_freq": 16,
        "tier_dim": 0,   # 【2026-11-05】锐度档位通道宽度：0 = 旧权重无档位；32 = 带档位（e25+）
        "use_multiscale_temporal": True, "temporal_kernel_sizes": (3, 5, 9), "temporal_kernel": 5,
        "learned_upsample_ks": [],   # 【2026-09-21】学习型上采样 PixelShuffle 的 k 列表；空 = 纯插值
    }

    # Detect channels from conv_in
    conv_key = 'resizer.conv_in.weight'
    if conv_key in sd:
        w = sd[conv_key]
        cfg["channels"] = w.shape[0]
        conv_in_ch = w.shape[1]
        
        # 【关键修复】CoordConv2D 无参数，不在 state_dict 中
        # 通过 conv_in 输入通道数判断：26 = 24(latent) + 2(coord)
        # 同时检查 conv_out 输出通道数确认真实 latent 通道数
        conv_out_key = 'resizer.conv_out.weight'
        if conv_out_key in sd:
            real_in_channels = sd[conv_out_key].shape[0]  # conv_out 输出 = 真实 latent 通道数
        else:
            real_in_channels = 24  # fallback
        
        if conv_in_ch == real_in_channels + 2:
            cfg["use_coord"] = True
            cfg["in_channels"] = real_in_channels
        else:
            cfg["use_coord"] = False
            cfg["in_channels"] = conv_in_ch

    # Count blocks
    in_ids, out_ids = set(), set()
    for k in sd.keys():
        m = re.match(r'resizer\.in_blocks\.(\d+)\.', k)
        if m: in_ids.add(int(m.group(1)))
        m = re.match(r'resizer\.out_blocks\.(\d+)\.', k)
        if m: out_ids.add(int(m.group(1)))
    if in_ids: cfg["in_blocks"] = max(in_ids) + 1
    if out_ids: cfg["out_blocks"] = max(out_ids) + 1

    # Detect SE Block
    cfg["use_se"] = any('.se.fc.' in k for k in sd)

    # Detect Large Kernel
    cfg["use_large_kernel"] = any('.dw.weight' in k and 'resizer.in_blocks' in k
                                   and 'temporal' not in k for k in sd)
    if cfg["use_large_kernel"]:
        for k in sd.keys():
            if '.dw.weight' in k and 'resizer.in_blocks' in k and 'temporal' not in k:
                cfg["large_kernel_size"] = sd[k].shape[2]
                break

    # Detect Post Refine
    cfg["use_post_refine"] = any('resizer.post_up.refine.' in k for k in sd)

    # 【2026-09-21】Detect 学习型上采样（PixelShuffle）支路：从键名 resizer.learned_up.branches.<k>. 推断。
    # 权重中没有该前缀 → 空列表 → 模型不构建该支路 → 走纯插值（旧权重完全兼容）。
    _lu_ks = set()
    for k in sd.keys():
        m = re.match(r'resizer\.learned_up\.branches\.(\d+)\.', k)
        if m:
            _lu_ks.add(int(m.group(1)))
    cfg["learned_upsample_ks"] = sorted(_lu_ks)

    # Detect Dual Scale Embedding（【v12】embed_dim / 频率数都由权重形状决定，不能写死）
    cfg["use_dual_scale"] = any('resizer.embed.linear.' in k for k in sd)
    emb_k = sd.get('resizer.embed.linear.0.weight')
    if emb_k is not None:
        cfg["embed_dim"] = int(emb_k.shape[0])          # Linear(in_dim, embed_dim)
        _in_dim = int(emb_k.shape[1])
        # 【2026-11-05】输入宽度判定：
        #   64 = 4*16            → 只有 scale_h/scale_w，无档位（v12 ~ e24）
        #   96 = 4*16 + 32       → 额外 32 维锐度档位 Fourier 编码（e25 及之后的 tier 权重）
        # 训练脚本 num_frequencies 固定为 16，所以只认这两种；其它宽度按"无档位"兜底。
        if _in_dim == 96:
            cfg["dual_embed_freq"], cfg["tier_dim"] = 16, 32
        else:
            cfg["dual_embed_freq"], cfg["tier_dim"] = _in_dim // 4, 0

    # Detect MultiScale Temporal
    cfg["use_multiscale_temporal"] = any('temporal_blocks.0.branches.' in k for k in sd)
    if cfg["use_multiscale_temporal"]:
        ks = set()
        for k in sd.keys():
            m = re.match(r'temporal_blocks\.0\.branches\.(\d+)\.0\.weight', k)
            if m:
                ks.add(sd[k].shape[2])
        if ks:
            cfg["temporal_kernel_sizes"] = tuple(sorted(ks))
    else:
        # 【v12】TemporalConv 的参数名是 dwconv/pwconv（旧版为 dw/pw），两种都要识别
        for k in sd.keys():
            if 'temporal_blocks.0.dwconv.weight' in k or 'temporal_blocks.0.dw.weight' in k:
                cfg["temporal_kernel"] = sd[k].shape[2]
                break

    cfg["attn"] = False  # Force off for inference
    return cfg

def load_model(name, device, precision):
    backend_lbl = _backend_label(device)
    cache_key = f"{name}::{backend_lbl}::{precision}"
    if cache_key in MODEL_CACHE:
        model = MODEL_CACHE[cache_key]
        return model.to(device, non_blocking=True)

    try:
        path = folder_paths.get_full_path_or_raise(_LATENT_UPSCALE_FOLDER, name)
    except Exception as e:
        raise FileNotFoundError(f"Model file not found: {name}") from e

    raw_sd = _load_raw_sd(path)
    up_sd = _extract_upscaler_sd(raw_sd)
    cfg = _detect_arch(up_sd)

    resizer_kwargs = {
        "use_dual_scale": cfg["use_dual_scale"], "use_coord": cfg["use_coord"],
        "use_se": cfg["use_se"], "se_reduction": cfg["se_reduction"],
        "use_large_kernel": cfg["use_large_kernel"], "large_kernel_size": cfg["large_kernel_size"],
        "large_kernel_every": cfg["large_kernel_every"],
        "use_post_refine": cfg["use_post_refine"], "post_refine_layers": cfg["post_refine_layers"],
        "dual_embed_freq": cfg["dual_embed_freq"],
        "embed_dim": cfg["embed_dim"],
        "tier_dim": cfg.get("tier_dim", 0),                  # 【2026-11-05】0 = 无档位，32 = 支持锐度档位
        "learned_upsample_ks": cfg["learned_upsample_ks"],   # 【2026-09-21】空列表 → 纯插值
    }

    model = VideoLatentResizer(
        in_channels=cfg["in_channels"], in_blocks=cfg["in_blocks"], out_blocks=cfg["out_blocks"],
        channels=cfg["channels"], dropout=cfg["dropout"], attn=cfg["attn"],
        temporal_every=cfg["temporal_every"],
        use_multiscale_temporal=cfg["use_multiscale_temporal"],
        temporal_kernel_sizes=cfg["temporal_kernel_sizes"],
        temporal_kernel=cfg["temporal_kernel"],
        **resizer_kwargs,
    )

    missing, unexpected = model.load_state_dict(up_sd, strict=False)
    # 【v12 新增】strict=False 会静默跳过缺失/多余的键：如果缺的是 embed 或 temporal 权重，
    # 模型照常能跑、也不报错，但输出完全是错的。这里直接硬失败。
    if missing:
        preview = "\n".join(f"      - {k}" for k in missing[:20])
        raise RuntimeError(
            f"[MinimaxH3-2D] 权重与模型结构不匹配：{len(missing)}/{len(up_sd)} 个键未加载。\n"
            f"      strict=False 会让这些层保持随机初始化，输出结果无意义。\n"
            f"      缺失键（最多 20 个）:\n{preview}\n"
            f"      请确认权重版本与训练脚本 Q-latent-2d-v12.py 一致。"
        )
    if unexpected:
        print(f"[MinimaxH3-2D] ⚠️ 权重含 {len(unexpected)} 个未使用的多余键: {unexpected[:5]}")
    else:
        print(f"[MinimaxH3-2D] 🔒 权重校验通过：{len(up_sd)} 个键全部匹配")

    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}.get(precision, torch.float32)
    model = model.to(device).eval().requires_grad_(False)
    if dtype != torch.float32:
        model = model.to(dtype)

    MODEL_CACHE[cache_key] = model

    arch_info = []
    arch_info.append(f"Emb{cfg['embed_dim']}")
    if cfg.get("tier_dim"):
        arch_info.append(f"Tier{cfg['tier_dim']}")
    if cfg["use_dual_scale"]: arch_info.append("DualScale")
    if cfg["use_coord"]: arch_info.append("CoordConv")
    if cfg["use_se"]: arch_info.append("SE")
    if cfg["use_large_kernel"]: arch_info.append(f"LK{cfg['large_kernel_size']}")
    if cfg["use_post_refine"]: arch_info.append("PostRefine")
    if cfg["use_multiscale_temporal"]: arch_info.append(f"MSTemp{cfg['temporal_kernel_sizes']}")
    else: arch_info.append(f"Temp{cfg['temporal_kernel']}")
    if cfg["learned_upsample_ks"]:
        arch_info.append(f"LearnedPS{cfg['learned_upsample_ks']}")
    arch_info.append("LocalEnsemble" if USE_LOCAL_ENSEMBLE else "Bilinear")

    print(f"[MinimaxH3-2D] ✅ Loaded: {name}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,} | "
          f"Arch: {' + '.join(arch_info)} | "
          f"Backend: {backend_lbl} | Precision: {precision}")
    return model


# ==========================================
# ComfyUI Node
# ==========================================
class UpscaleMode(str, Enum):
    SCALE_BY = "scale by multiplier"
    TARGET_DIMENSIONS = "target dimensions"
    MEGAPIXELS = "megapixels"

class UpscaleConfig(TypedDict):
    mode: UpscaleMode
    scale: float
    width: int
    height: int
    megapixels: float

class MinimaxH3LatentUpscalerNode2D(io.ComfyNode):
    """Minimax H3 Latent Upscaler (2D) with Local Ensemble + Full Ultimate Architecture"""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MinimaxH3LatentUpscalerNode2D",
            display_name="Minimax H3 Latent Upscaler (2D)",
            category="video/MinimaxH3",
            search_aliases=["minimax", "h3", "latent", "upscale", "2d", "local ensemble"],
            inputs=[
                io.AnyType.Input("latent", tooltip="Input latent (image or video)."),
                io.Combo.Input("model_name", options=scan_models(), tooltip="Minimax H3 upscale model."),
                io.DynamicCombo.Input(
                    "mode",
                    tooltip="How the target size is computed.",
                    options=[
                        io.DynamicCombo.Option(UpscaleMode.SCALE_BY, [
                            io.Float.Input("scale", default=2.0, min=1.0, max=4.0, step=0.05, tooltip="Upscale factor."),
                        ]),
                        io.DynamicCombo.Option(UpscaleMode.TARGET_DIMENSIONS, [
                            io.Int.Input("width", default=1280, min=64, max=8192, step=8, tooltip="Target pixel width."),
                            io.Int.Input("height", default=704, min=64, max=8192, step=8, tooltip="Target pixel height.")
                        ]),
                        io.DynamicCombo.Option(UpscaleMode.MEGAPIXELS, [
                            io.Float.Input("megapixels", default=1.0, min=0.1, max=16.0, step=0.1, tooltip="Target megapixels.")
                        ])
                    ],
                ),
                io.Int.Input("sharpness", display_name="Sharpness",
                             default=1, min=1, max=6, step=1,
                             tooltip="1 = most faithful to the reference (softest, lowest MSE); "
                                     "6 = sharpest (most detail, furthest from the exact target). "
                                     "Only models trained with tier conditioning respond. / "
                                     "1 = 最贴近原图（最柔和，MSE 最低）；6 = 最锐（细节最多，偏离精确目标最远）。"
                                     "只对带档位训练的权重生效，旧权重自动忽略。"),
                io.Int.Input("align", default=32, min=1, max=512, step=1,
                             tooltip="Pixel-space alignment. 32 is strictly recommended."),
                # 【2026-11-05】时间分块：选 on 时才出现「分块帧数」，选 off 时该参数消失
                io.DynamicCombo.Input(
                    "temporal_chunking",
                    display_name="Temporal Chunking",
                    tooltip="Chunked inference for long videos to save VRAM. "
                            "chunked = customise frames per chunk; single pass = one forward "
                            "for the whole clip (most VRAM). / "
                            "长视频分块推理以省显存。选 chunked 可自定义每块帧数；选 single pass 则整段一次前向（最费显存）。",
                    # 注意：选项值不要用 "on"/"off" —— DD-Translation 插件会把字面量 "off"
                    # 全局翻成「关闭」，导致下拉里冒中文。这里用不会被翻译的英文词，
                    # 后端同时兼容旧工作流里存的 "on"/"off"。
                    options=[
                        io.DynamicCombo.Option("chunked", [
                            io.Int.Input("chunk_frames", display_name="Frames per chunk",
                                         default=32, min=4, max=256, step=4,
                                         tooltip="Fewer frames per chunk = less VRAM, more chunks. "
                                                 "Each chunk actually forwards this value + 48 frames "
                                                 "(temporal receptive-field context). / "
                                                 "每块的帧数。越小越省显存、块数越多；"
                                                 "单块实际前向帧数 = 该值 + 48（时间感受野上下文），"
                                                 "所以放得下就别调太小。"),
                        ]),
                        io.DynamicCombo.Option("single pass", []),
                    ],
                ),
                io.Boolean.Input("force_unload", default=True,
                                 tooltip="Unload model to CPU after inference to free VRAM for subsequent nodes."),
                io.Combo.Input("device", options=["cuda", "rocm", "cpu"], default="cuda"),
                io.Combo.Input("precision", options=["fp32", "fp16", "bf16"], default="fp16"),
            ],
            outputs=[
                io.AnyType.Output("latent", tooltip="Upscaled latent."),
            ],
        )

    # ==========================================
    # New ComfyUI API entry (comfy_api.latest)
    # Convention matches the 3D node: io.ComfyNode + define_schema + execute,
    # with the full pipeline inline (no legacy INPUT_TYPES / FUNCTION wrapper).
    # ==========================================
    @classmethod
    def execute(cls, latent: dict, model_name: str, mode: UpscaleConfig,
                align: int, force_unload: bool,
                device: str, precision: str,
                sharpness: int = 1, temporal_chunking: dict = None) -> io.NodeOutput:
        if model_name.startswith('('):
            raise ValueError("Please place model files into the latent_upscale_models directory")

        # ---- AV：只取视频流处理，音频留在原 dict 里不动 ----
        src = latent["samples"]
        is_av = getattr(src, "is_nested", False)
        if is_av: src = src.unbind()[0]

        selected_mode = mode["mode"]
        orig_dtype = src.dtype
        was_4d = (src.dim() == 4)

        dev = _resolve_device(device)
        compute_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]

        s = src.to(device=dev, dtype=compute_dtype, copy=True)
        if was_4d:
            s = s.unsqueeze(2)

        b, c, t, h_in, w_in = s.shape
        downsample = VAE_DOWNSAMPLE

        # 1. Calculate Target Size
        if selected_mode == UpscaleMode.SCALE_BY:
            scale_val = mode["scale"]
            w_pixel_target = w_in * downsample * scale_val
            h_pixel_target = h_in * downsample * scale_val
            effective_scale = scale_val
        elif selected_mode == UpscaleMode.TARGET_DIMENSIONS:
            w_pixel_target = float(mode["width"])
            h_pixel_target = float(mode["height"])
            effective_scale = (w_pixel_target / (w_in * downsample) + h_pixel_target / (h_in * downsample)) / 2.0
        elif selected_mode == UpscaleMode.MEGAPIXELS:
            mp = mode["megapixels"]
            target_pixels = mp * 1024 * 1024
            aspect_ratio = w_in / h_in
            h_pixel_target = (target_pixels / aspect_ratio) ** 0.5
            w_pixel_target = h_pixel_target * aspect_ratio
            effective_scale = (w_pixel_target / (w_in * downsample) + h_pixel_target / (h_in * downsample)) / 2.0
        else:
            raise ValueError(f"Unsupported mode: {selected_mode}")

        # 2. Pixel-space alignment
        alignment = max(1, align)
        w_pixel_aligned = round(w_pixel_target / alignment) * alignment
        h_pixel_aligned = round(h_pixel_target / alignment) * alignment
        w_pixel_final = round(w_pixel_aligned / downsample) * downsample
        h_pixel_final = round(h_pixel_aligned / downsample) * downsample
        w_out = max(1, int(w_pixel_final // downsample))
        h_out = max(1, int(h_pixel_final // downsample))

        if effective_scale < 1.0 and (w_out < w_in or h_out < h_in):
            raise ValueError("This model only supports upscaling (effective scale >= 1.0).")

        if w_out == w_in and h_out == h_in:
            return io.NodeOutput(latent)

        print(f"[MinimaxH3-2D] Latent {w_in}x{h_in} -> {w_out}x{h_out} | "
              f"Pixels {w_out * downsample}x{h_out * downsample} | scale={effective_scale:.3f}")

        # 3. Inference
        model = load_model(model_name, dev, precision)
        norm_mean, norm_std = _make_norm_tensors(dev, compute_dtype)

        # 【2026-11-05】时间分块：chunked → 用可选的分块帧数；single pass → 整段一次前向
        # 【2026-11-09】兼容旧工作流：旧值可能是 "on"/"off"（"off" 会被 DD-Translation 显示成「关闭」）
        _tc = temporal_chunking or {}
        _tc_mode = str(_tc.get("temporal_chunking", "chunked")).strip().lower()
        _chunk_on = _tc_mode not in ("single pass", "off", "false", "0", "none")
        _chunk_frames = int(_tc.get("chunk_frames", CHUNK_FRAMES)) if _chunk_on else None
        print(f"[MinimaxH3-2D] 时间分块: "
              + (f"每 {_chunk_frames} 帧一块" if _chunk_on else "关闭（整段一次前向）"))

        # 【2026-11-05】锐度档位（内部仍叫 tier，与训练脚本的 t1~t6 一致）
        # 只有带档位通道的权重才响应，旧权重静默忽略
        _tier_dim = getattr(model.resizer, "tier_dim", 0)
        tier_val = int(sharpness) if _tier_dim > 0 else None
        if _tier_dim > 0:
            print(f"[MinimaxH3-2D] 锐度 {int(sharpness)}/6（1 = 最贴近原图，6 = 最锐）")
        elif int(sharpness) != 1:
            print(f"[MinimaxH3-2D] ⚠️ 该权重不含锐度档位通道（sharpness 参数无效），已按 1 处理")

        with torch.inference_mode():
            s_norm = (s - norm_mean) / norm_std
            del s
            out = model(s_norm, scale=effective_scale, target_hw=(h_out, w_out),
                        enable_chunking=_chunk_on, chunk_frames=_chunk_frames, tier=tier_val)
            del s_norm
            out = out * norm_std + norm_mean

        if was_4d:
            out = out.squeeze(2)

        out = out.to(device="cpu", dtype=orig_dtype, non_blocking=True)

        # 4. VRAM Management
        if dev.type == "cuda":
            if force_unload:
                model.to("cpu", non_blocking=True)
                print("[MinimaxH3-2D] ✅ Model offloaded to CPU. VRAM released.")
            if HAS_COMFY_MM:
                mm.soft_empty_cache()
            else:
                torch.cuda.empty_cache()
            gc.collect()
                    
        latent["samples"] = NestedTensor((out, latent["samples"].unbind()[1])) if is_av else out
        return io.NodeOutput(latent)


# ==========================================
# Registration
# ==========================================
if USE_NEW_API:
    class MinimaxH3Extension(ComfyExtension):
        @override
        async def get_node_list(self) -> list[type[io.ComfyNode]]:
            return [MinimaxH3LatentUpscalerNode2D]

    async def comfy_entrypoint() -> MinimaxH3Extension:
        return MinimaxH3Extension()

NODE_CLASS_MAPPINGS = {
    "MinimaxH3LatentUpscalerNode2D": MinimaxH3LatentUpscalerNode2D,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MinimaxH3LatentUpscalerNode2D": "Minimax H3 Latent Upscaler (2D)",
}