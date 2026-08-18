"""
H3 Latent Upscaler - 公共推理逻辑 (纯3D卷积版本)
- 自动检测模型结构 (通道数、块数、Temporal 位置)
- 支持 FP32 / FP16 / BF16 推理
- 强制关闭注意力 (attn=False)
- 模型从 ComfyUI/models/latent_upscale_models/ 加载
- https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler

本模块被「按分辨率放大」「按像素放大」两个节点复用，避免重复实现模型加载与推理。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
import folder_paths
import re
import math
from comfy_extras.nodes_lt import LTXVConcatAVLatent, LTXVSeparateAVLatent
from einops import rearrange

# ==========================================
# 注册模型文件夹
# ==========================================
_LATENT_UPSCALE_FOLDER = "latent_upscale_models"
if _LATENT_UPSCALE_FOLDER not in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path(
        _LATENT_UPSCALE_FOLDER,
        os.path.join(folder_paths.models_dir, _LATENT_UPSCALE_FOLDER)
    )

# ==========================================
# Minimax H3 归一化参数 (来自训练代码，24通道)
# ==========================================
LATENTS_MEAN = [
    0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075,
    -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975,
    -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
    -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543,
    -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279,
    -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264
]
LATENTS_STD  = [
    1.2223774194717407, 1.2767263650894165, 1.6831774711608887, 1.7549455165863037,
    1.5636216402053833, 2.194143533706665, 0.9653137922286987, 1.0569885969161987,
    0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647,
    0.7996809482574463, 0.44988900423049927, 0.7197399735450745, 0.6936293244361877,
    2.961095094680786, 2.7694199085235596, 3.0496184825897217, 2.1088054180145264,
    3.276226282119751, 3.1627357006073, 2.2816812992095947, 2.6127843856811523
]

_PRECISION_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

def _make_norm_tensors(device, dtype):
    """创建与 3D latent 张量形状兼容的归一化参数。"""
    mean = torch.tensor(LATENTS_MEAN, dtype=dtype, device=device).view(1, -1, 1, 1, 1)
    std = torch.tensor(LATENTS_STD, dtype=dtype, device=device).view(1, -1, 1, 1, 1)
    return mean, std


def _align_latent_to_grid(pixel_width, pixel_height, grid):
    """
    把目标像素尺寸转换成对齐到指定 grid 倍数的 latent 尺寸。

    Minimax H3 的 VAE 下采样率为 16（像素 = latent × 16）。
    `grid` 作用于 latent 空间：输出 latent 的 H、W 都会向上取整到 grid 的整数倍，
    例如 grid=16 表示 latent 尺寸是 16 的倍数（对应像素 256 的倍数），
    grid=8 / 32 同理。这样可保证放大后的尺寸满足下游某些对齐要求。
    """
    grid = max(1, int(grid))
    latent_w = math.ceil(float(pixel_width) / 16.0)
    latent_h = math.ceil(float(pixel_height) / 16.0)
    latent_w = ((latent_w + grid - 1) // grid) * grid
    latent_h = ((latent_h + grid - 1) // grid) * grid
    return latent_w, latent_h


# ==========================================
# 3D 网络组件 (与训练代码一致)
# ==========================================
def normalization(channels):
    return nn.GroupNorm(32, channels)

def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

class AttnBlock3D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = normalization(in_channels)
        self.q = nn.Conv3d(in_channels, in_channels, 1)
        self.k = nn.Conv3d(in_channels, in_channels, 1)
        self.v = nn.Conv3d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv3d(in_channels, in_channels, 1)

    def forward(self, x):
        h = self.norm(x)
        q = rearrange(self.q(h), "b c t h w -> b 1 (t h w) c")
        k = rearrange(self.k(h), "b c t h w -> b 1 (t h w) c")
        v = rearrange(self.v(h), "b c t h w -> b 1 (t h w) c")
        h = F.scaled_dot_product_attention(q, k, v)
        h = rearrange(h, "b 1 (t h w) c -> b c t h w", t=x.shape[2], h=x.shape[3], w=x.shape[4])
        return x + self.proj_out(h)

class ResBlockEmb3D(nn.Module):
    def __init__(self, channels, emb_channels, dropout=0, out_channels=None):
        super().__init__()
        self.out_channels = out_channels or channels
        self.in_layers = nn.Sequential(
            normalization(channels), nn.SiLU(),
            nn.Conv3d(channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(), nn.Linear(emb_channels, 2 * self.out_channels),
        )
        self.out_norm = normalization(self.out_channels)
        self.out_layers = nn.Sequential(
            nn.SiLU(), nn.Dropout(p=dropout),
            zero_module(nn.Conv3d(self.out_channels, self.out_channels, 3, padding=1)),
        )
        self.skip = (
            nn.Conv3d(channels, self.out_channels, 1)
            if self.out_channels != channels else nn.Identity()
        )

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        h = self.out_norm(h) * (1 + scale) + shift
        h = self.out_layers(h)
        return self.skip(x) + h

class TemporalConv(nn.Module):
    def __init__(self, channels, kernel_size=5):
        super().__init__()
        padding = kernel_size // 2
        self.norm = normalization(channels)
        self.dwconv = nn.Conv3d(channels, channels,
                                kernel_size=(kernel_size, 1, 1),
                                padding=(padding, 0, 0),
                                groups=channels)
        self.pwconv = nn.Conv3d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.pwconv.weight)
        nn.init.zeros_(self.pwconv.bias)

    def forward(self, x):
        identity = x
        h = self.norm(x)
        h = F.silu(h)
        h = self.dwconv(h)
        h = self.pwconv(h)
        return identity + h

# ==========================================
# 纯3D主干网络 (与训练代码一致)
# ==========================================
class LatentResizer3D(nn.Module):
    def __init__(self, in_channels=24, in_blocks=12, out_blocks=12,
                 channels=512, dropout=0.1, attn=False,
                 temporal_every=2, temporal_kernel=5):
        super().__init__()
        self.conv_in = nn.Conv3d(in_channels, channels, 3, padding=1)
        embed_dim = 64
        self.embed = nn.Sequential(
            nn.Linear(1, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))

        self.in_blocks = nn.ModuleList()
        for b in range(in_blocks):
            if (b == 1 or b == in_blocks - 1) and attn:
                self.in_blocks.append(AttnBlock3D(channels))
            self.in_blocks.append(ResBlockEmb3D(channels, embed_dim, dropout))
            if temporal_every > 0 and b % temporal_every == 0:
                self.in_blocks.append(TemporalConv(channels, temporal_kernel))

        self.out_blocks = nn.ModuleList()
        for b in range(out_blocks):
            if (b == 1 or b == out_blocks - 1) and attn:
                self.out_blocks.append(AttnBlock3D(channels))
            self.out_blocks.append(ResBlockEmb3D(channels, embed_dim, dropout))
            if temporal_every > 0 and b % temporal_every == 0:
                self.out_blocks.append(TemporalConv(channels, temporal_kernel))

        self.norm_out = normalization(channels)
        self.conv_out = nn.Conv3d(channels, in_channels, 3, padding=1)

    def forward(self, x, scale=None, target_size=None):
        if target_size is not None:
            size = target_size
        elif scale is not None:
            size = tuple(int(round(s * scale)) for s in x.shape[-3:])
        else:
            return x

        if size == x.shape[-3:]:
            return x

        scale_emb = torch.tensor(
            [scale - 1 if scale is not None else 0.0],
            dtype=x.dtype, device=x.device).unsqueeze(0)
        emb = self.embed(scale_emb)

        x = self.conv_in(x)
        for b in self.in_blocks:
            if isinstance(b, ResBlockEmb3D):
                emb_t = emb.expand(x.shape[0], -1)
                x = b(x, emb_t)
            else:
                x = b(x)

        # 三线性插值
        x = F.interpolate(x, size=size, mode="trilinear", align_corners=False)

        for b in self.out_blocks:
            if isinstance(b, ResBlockEmb3D):
                emb_t = emb.expand(x.shape[0], -1)
                x = b(x, emb_t)
            else:
                x = b(x)

        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)
        return x

# ==========================================
# 模型加载 (纯3D版本)
# ==========================================
MODEL_CACHE = {}

def get_models_dir():
    return folder_paths.get_folder_paths(_LATENT_UPSCALE_FOLDER)[0]

def scan_models():
    files = []
    model_dir = get_models_dir()
    for ext in ("*.pth", "*.safetensors"):
        files.extend(glob.glob(os.path.join(model_dir, ext)))
    names = sorted(os.path.basename(f) for f in files)
    return names if names else [f"(请将模型放入: {model_dir})"]

def _load_raw_sd(path):
    if path.endswith('.safetensors'):
        from safetensors.torch import load_file
        sd = load_file(path, device='cpu')
    else:
        sd = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and 'model' in sd:
        sd = sd['model']
    # 移除可能的前缀 (如果有) 并处理 FP8 格式
    sd = {k: v.to(torch.float16) if v.dtype == torch.float8_e4m3fn else v
          for k, v in sd.items()}
    return sd

def _extract_upscaler_sd(sd):
    # 兼容之前可能包含 'upscaler.' 前缀的合并权重
    if any(k.startswith("upscaler.") for k in sd):
        return {k[len("upscaler."):]: v for k, v in sd.items() if k.startswith("upscaler.")}
    return sd

def _detect_arch(sd):
    """
    从 state_dict 推断模型结构参数。
    返回: dict 包含 in_blocks, out_blocks, channels, in_channels, dropout, attn, temporal_every, temporal_kernel
    """
    cfg = {
        "in_channels": 24,
        "in_blocks": 12,
        "out_blocks": 12,
        "channels": 512,
        "dropout": 0.1,
        "attn": False,
        "temporal_every": 2,
        "temporal_kernel": 5,
    }

    conv_key = 'conv_in.weight'
    if conv_key in sd:
        cfg["in_channels"] = sd[conv_key].shape[1]
        cfg["channels"] = sd[conv_key].shape[0]

    in_ids = set()
    out_ids = set()
    temporal_in_indices = set()
    temporal_out_indices = set()
    for k in sd.keys():
        m = re.match(r'in_blocks\.(\d+)\.in_layers\.', k)
        if m:
            in_ids.add(int(m.group(1)))
        m = re.match(r'out_blocks\.(\d+)\.in_layers\.', k)
        if m:
            out_ids.add(int(m.group(1)))
        m = re.match(r'in_blocks\.(\d+)\.dwconv\.weight', k)
        if m:
            temporal_in_indices.add(int(m.group(1)))
        m = re.match(r'out_blocks\.(\d+)\.dwconv\.weight', k)
        if m:
            temporal_out_indices.add(int(m.group(1)))

    if in_ids:
        cfg["in_blocks"] = len(in_ids)
    if out_ids:
        cfg["out_blocks"] = len(out_ids)

    if temporal_in_indices or temporal_out_indices:
        cfg["temporal_every"] = 2
        for k in sd.keys():
            if 'dwconv.weight' in k and k.endswith('dwconv.weight'):
                kernel_t = sd[k].shape[2]
                cfg["temporal_kernel"] = kernel_t
                break
    else:
        cfg["temporal_every"] = 0

    # 推理时为了性能和稳定性，强制 attn=False
    cfg["attn"] = False

    return cfg

def load_model(name, device, precision):
    cache_key = f"{name}::{device}::{precision}"
    if cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key]

    path = os.path.join(get_models_dir(), name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"模型文件不存在: {path}")

    raw_sd = _load_raw_sd(path)
    up_sd = _extract_upscaler_sd(raw_sd)
    cfg = _detect_arch(up_sd)

    model = LatentResizer3D(
        in_channels=cfg["in_channels"],
        in_blocks=cfg["in_blocks"],
        out_blocks=cfg["out_blocks"],
        channels=cfg["channels"],
        dropout=cfg["dropout"],
        attn=cfg["attn"],
        temporal_every=cfg["temporal_every"],
        temporal_kernel=cfg["temporal_kernel"],
    )
    model.load_state_dict(up_sd, strict=True)
    dtype = _PRECISION_MAP.get(precision, torch.float32)
    model = model.to(device).eval()
    if dtype != torch.float32:
        model = model.to(dtype)

    MODEL_CACHE[cache_key] = model

    print(f"[MinimaxH3-3D] 加载放大模型: {name}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,} | "
          f"Attn: 强制关闭 | Temporal: {'✓' if cfg['temporal_every']>0 else '✗'} "
          f"(every={cfg['temporal_every']}, kernel={cfg['temporal_kernel']})")
    return model

def run_upscale(video_latent, audio_latent, model_name, device, precision,
                target_width, target_height, target_scale):
    """
    把分离出的 video latent 放大到 (target_height, target_width) 并重新合并音视频。

    Args:
        video_latent: 已用 LTXVSeparateAVLatent 分离出的视频 latent dict。
        audio_latent: 对应的音频 latent dict。
        model_name: 下拉框选中的模型文件名。
        device: "cuda" / "cpu"。
        precision: "fp32" / "fp16" / "bf16"。
        target_width / target_height: 目标 latent 空间尺寸 (H×W 的 W/H)。
        target_scale: 等效放大倍率，用于驱动模型内部的 scale embedding。

    Returns:
        合并后的 LATENT dict，可直接送 VAE 解码。
    """
    source_samples = video_latent["samples"]
    source_height = int(source_samples.shape[-2])
    source_width = int(source_samples.shape[-1])

    if target_width == source_width and target_height == source_height:
        return LTXVConcatAVLatent.execute(video_latent, audio_latent)

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = load_model(model_name, dev, precision)

    s = source_samples.clone()
    orig_dtype = s.dtype
    if len(s.shape) == 4:
        s = s.unsqueeze(2)  # (B, C, 1, H, W)

    compute_dtype = _PRECISION_MAP.get(precision, torch.float32)
    s = s.to(dev, compute_dtype)

    norm_mean, norm_std = _make_norm_tensors(dev, compute_dtype)
    s = (s - norm_mean) / norm_std

    with torch.no_grad():
        T = int(s.shape[2])
        target_size = (T, target_height, target_width)
        out = model(s, scale=target_scale, target_size=target_size)

    out = out * norm_std + norm_mean

    if len(source_samples.shape) == 4:
        out = out.squeeze(2)

    out = out.cpu().to(orig_dtype)

    if dev.type == "cuda":
        torch.cuda.empty_cache()

    video_latent["samples"] = out
    return LTXVConcatAVLatent.execute(video_latent, audio_latent)
