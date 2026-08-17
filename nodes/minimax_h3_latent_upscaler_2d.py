"""
Minimax H3 Latent Upscaler - ComfyUI 推理节点 (2D主干 + Temporal 3D卷积)
- 使用 VideoLatentResizer (与训练代码完全一致)
- 自动检测模型结构并加载权重 (支持合并文件中的 upscaler. 前缀)
- 强制关闭注意力 (attn=False) 以优化推理速度
- 模型从 ComfyUI/models/latent_upscale_models/ 加载
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
import folder_paths
import re
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
# Minimax H3 归一化参数 (24通道)
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

def _make_norm_tensors(device, dtype):
    mean = torch.tensor(LATENTS_MEAN, dtype=dtype, device=device).view(1, -1, 1, 1, 1)
    std = torch.tensor(LATENTS_STD, dtype=dtype, device=device).view(1, -1, 1, 1, 1)
    return mean, std

# ==========================================
# 与训练代码完全一致的网络组件 (2D + Temporal)
# ==========================================
def normalization(channels):
    return nn.GroupNorm(32, channels)

def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = normalization(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, 1)
        self.k = nn.Conv2d(in_channels, in_channels, 1)
        self.v = nn.Conv2d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        h = self.norm(x)
        q = rearrange(self.q(h), "b c h w -> b 1 (h w) c")
        k = rearrange(self.k(h), "b c h w -> b 1 (h w) c")
        v = rearrange(self.v(h), "b c h w -> b 1 (h w) c")
        h = F.scaled_dot_product_attention(q, k, v)
        h = rearrange(h, "b 1 (h w) c -> b c h w", h=x.shape[-2], w=x.shape[-1])
        return x + self.proj_out(h)

class ResBlockEmb(nn.Module):
    def __init__(self, channels, emb_channels, dropout=0, out_channels=None):
        super().__init__()
        self.out_channels = out_channels or channels
        self.in_layers = nn.Sequential(
            normalization(channels), nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(), nn.Linear(emb_channels, 2 * self.out_channels),
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
        B, C, T, H, W = x.shape
        # 先 reshape 为 (B*T, C, H, W) 应用 2D norm
        h = rearrange(x, "b c t h w -> (b t) c h w")
        h = self.norm(h)
        h = rearrange(h, "(b t) c h w -> b c t h w", b=B, t=T)
        h = F.silu(h)
        h = self.dwconv(h)
        h = self.pwconv(h)
        return identity + h

class LatentResizer(nn.Module):
    """2D 主干 (与训练代码完全一致)"""
    def __init__(self, in_channels=24, in_blocks=12, out_blocks=12,
                 channels=640, dropout=0.1, attn=False):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, channels, 3, padding=1)
        embed_dim = 64
        self.embed = nn.Sequential(
            nn.Linear(1, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))
        self.in_blocks = nn.ModuleList()
        for b in range(in_blocks):
            if (b == 1 or b == in_blocks - 1) and attn:
                self.in_blocks.append(AttnBlock(channels))
            self.in_blocks.append(ResBlockEmb(channels, embed_dim, dropout))
        self.out_blocks = nn.ModuleList()
        for b in range(out_blocks):
            if (b == 1 or b == out_blocks - 1) and attn:
                self.out_blocks.append(AttnBlock(channels))
            self.out_blocks.append(ResBlockEmb(channels, embed_dim, dropout))
        self.norm_out = normalization(channels)
        self.conv_out = nn.Conv2d(channels, in_channels, 3, padding=1)

    def forward(self, x, scale=None, target_hw=None):
        if target_hw is not None:
            size = target_hw
        elif scale is not None:
            size = tuple(int(round(s * scale)) for s in x.shape[-2:])
        else:
            return x
        if size == x.shape[-2:]:
            return x

        scale_emb = torch.tensor(
            [scale - 1 if scale is not None else 0.0],
            dtype=x.dtype, device=x.device).unsqueeze(0)
        emb = self.embed(scale_emb)

        x = self.conv_in(x)
        for b in self.in_blocks:
            if isinstance(b, ResBlockEmb):
                x = b(x, emb)
            else:
                x = b(x)
        x = F.interpolate(x, size=size, mode="bilinear")
        for b in self.out_blocks:
            if isinstance(b, ResBlockEmb):
                x = b(x, emb)
            else:
                x = b(x)
        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)
        return x

class VideoLatentResizer(nn.Module):
    """5D 包装器 (含 Temporal 块) 与训练代码完全一致"""
    def __init__(self, in_channels=24, in_blocks=12, out_blocks=12,
                 channels=640, dropout=0.1, attn=False,
                 temporal_every=2, temporal_kernel=5):
        super().__init__()
        self.resizer = LatentResizer(
            in_channels=in_channels,
            in_blocks=in_blocks,
            out_blocks=out_blocks,
            channels=channels,
            dropout=dropout,
            attn=attn,
        )
        self.temporal_blocks = nn.ModuleList()
        if temporal_every > 0:
            self.temporal_blocks.append(TemporalConv(channels, temporal_kernel))
            self.temporal_blocks.append(TemporalConv(channels, temporal_kernel))
        self.temporal_every = temporal_every
        self.temporal_kernel = temporal_kernel

    def forward(self, x, scale=None, target_hw=None):
        B, C, T, H, W = x.shape
        if target_hw is not None:
            size = target_hw
        elif scale is not None:
            size = (int(round(H * scale)), int(round(W * scale)))
        else:
            size = (H, W)

        if len(self.temporal_blocks) == 0:
            x_flat = rearrange(x, "b c t h w -> (b t) c h w")
            out = self.resizer(x_flat, scale=scale, target_hw=size)
            return rearrange(out, "(b t) c h w -> b c t h w", b=B, t=T)

        x_flat = rearrange(x, "b c t h w -> (b t) c h w")
        emb = torch.tensor(
            [scale - 1 if scale is not None else 0.0],
            dtype=x.dtype, device=x.device).unsqueeze(0)
        emb = self.resizer.embed(emb)

        out = self.resizer.conv_in(x_flat)
        for i, block in enumerate(self.resizer.in_blocks):
            if isinstance(block, ResBlockEmb):
                emb_t = emb.expand(B * T, -1)
                out = block(out, emb_t)
            else:
                out = block(out)
            if i % self.temporal_every == 0:
                out_3d = rearrange(out, "(b t) c h w -> b c t h w", b=B, t=T)
                out_3d = self.temporal_blocks[0](out_3d)
                out = rearrange(out_3d, "b c t h w -> (b t) c h w")

        out = F.interpolate(out, size=size, mode="bilinear")

        for i, block in enumerate(self.resizer.out_blocks):
            if isinstance(block, ResBlockEmb):
                emb_t = emb.expand(B * T, -1)
                out = block(out, emb_t)
            else:
                out = block(out)
            if i % self.temporal_every == 0:
                out_3d = rearrange(out, "(b t) c h w -> b c t h w", b=B, t=T)
                out_3d = self.temporal_blocks[1](out_3d)
                out = rearrange(out_3d, "b c t h w -> (b t) c h w")

        out = self.resizer.norm_out(out)
        out = F.silu(out)
        out = self.resizer.conv_out(out)
        out = rearrange(out, "(b t) c h w -> b c t h w", b=B, t=T)
        return out

# ==========================================
# 模型加载 (适配训练权重)
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
    # 转换 FP8 为 FP16 方便处理
    sd = {k: v.to(torch.float16) if v.dtype == torch.float8_e4m3fn else v
          for k, v in sd.items()}
    return sd

def _extract_upscaler_sd(sd):
    # 兼容合并权重中的 upscaler. 前缀
    if any(k.startswith("upscaler.") for k in sd):
        return {k[len("upscaler."):]: v for k, v in sd.items() if k.startswith("upscaler.")}
    return sd

def _detect_arch(sd):
    """
    从 state_dict 推断模型结构参数 (与训练代码一致)
    返回: dict 包含 in_blocks, out_blocks, channels, in_channels,
          dropout, attn, temporal_every, temporal_kernel
    """
    cfg = {
        "in_channels": 24,
        "in_blocks": 12,
        "out_blocks": 12,
        "channels": 640,
        "dropout": 0.1,
        "attn": False,
        "temporal_every": 2,
        "temporal_kernel": 5,
    }

    # 从 conv_in 推断通道数
    if 'resizer.conv_in.weight' in sd:
        w = sd['resizer.conv_in.weight']
        cfg["in_channels"] = w.shape[1]
        cfg["channels"] = w.shape[0]

    # 统计 in_blocks 中的 ResBlock 数量 (通过 in_layers 键)
    in_ids = set()
    out_ids = set()
    for k in sd.keys():
        m = re.match(r'resizer\.in_blocks\.(\d+)\.in_layers\.', k)
        if m:
            in_ids.add(int(m.group(1)))
        m = re.match(r'resizer\.out_blocks\.(\d+)\.in_layers\.', k)
        if m:
            out_ids.add(int(m.group(1)))

    if in_ids:
        cfg["in_blocks"] = len(in_ids)
    if out_ids:
        cfg["out_blocks"] = len(out_ids)

    # 检测是否包含 temporal 块
    has_temporal = any('temporal_blocks' in k for k in sd)
    if has_temporal:
        # 尝试从 temporal_blocks.0.dwconv.weight 提取 kernel size
        for k in sd.keys():
            if 'temporal_blocks.0.dwconv.weight' in k:
                kernel_t = sd[k].shape[2]
                cfg["temporal_kernel"] = kernel_t
                break
        cfg["temporal_every"] = 2  # 训练固定值
    else:
        cfg["temporal_every"] = 0

    # 检测是否包含 attn (但推理时强制关闭)
    if any('attn' in k for k in sd):
        cfg["attn"] = True  # 仅用于记录，实际加载时强制 False

    # 推理时强制 attn=False 以提高性能
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

    # 构建模型 (与训练完全一致)
    model = VideoLatentResizer(
        in_channels=cfg["in_channels"],
        in_blocks=cfg["in_blocks"],
        out_blocks=cfg["out_blocks"],
        channels=cfg["channels"],
        dropout=cfg["dropout"],
        attn=cfg["attn"],                # 强制 False
        temporal_every=cfg["temporal_every"],
        temporal_kernel=cfg["temporal_kernel"],
    )

    # 加载权重，忽略不匹配的键 (如 attn 层若训练有但推理强制关闭)
    missing, unexpected = model.load_state_dict(up_sd, strict=False)
    if missing:
        print(f"[MinimaxH3] 缺少键: {missing[:5]}... (可能由于 attn 强制关闭)")
    if unexpected:
        print(f"[MinimaxH3] 多余键: {unexpected[:5]}... (可能来自合并文件的其他部分)")

    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}.get(precision, torch.float32)
    model = model.to(device).eval()
    if dtype != torch.float32:
        model = model.to(dtype)

    MODEL_CACHE[cache_key] = model

    print(f"[MinimaxH3] 加载放大模型: {name}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,} | "
          f"Attn: 强制关闭 | Temporal: {'✓' if cfg['temporal_every']>0 else '✗'} "
          f"(every={cfg['temporal_every']}, kernel={cfg['temporal_kernel']})")
    return model

# ==========================================
# ComfyUI 节点 (2D 版本)
# ==========================================
class MinimaxH3LatentUpscalerNode2D:
    """Minimax H3 Latent 放大节点 (2D主干 + Temporal，与训练模型兼容，scale 1.0~4.0)"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "model_name": (scan_models(),),
                "scale": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.1}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "precision": (["fp32", "fp16", "bf16"], {"default": "fp32"}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "run"
    CATEGORY = "video/MinimaxH3"

    def run(self, latent, model_name, scale, device, precision):
        if model_name.startswith('('):
            raise ValueError("请将模型文件放入 latent_upscale_models 目录")

        if abs(scale - 1.0) < 1e-6:
            return (latent,)

        if scale < 1.0:
            raise ValueError("仅支持放大 (scale >= 1.0)")

        dev = torch.device(device if torch.cuda.is_available() else "cpu")
        model = load_model(model_name, dev, precision)

        s = latent["samples"].clone()
        orig_dtype = s.dtype
        # 确保是 5D (B, C, T, H, W)
        if len(s.shape) == 4:
            s = s.unsqueeze(2)  # (B, C, 1, H, W)

        compute_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]
        s = s.to(dev, compute_dtype)

        # 归一化
        norm_mean, norm_std = _make_norm_tensors(dev, compute_dtype)
        s = (s - norm_mean) / norm_std

        with torch.no_grad():
            # 目标空间尺寸 (H, W)，时间维度不变
            T, H, W = s.shape[2], s.shape[3], s.shape[4]
            target_hw = (int(round(H * scale)), int(round(W * scale)))
            out = model(s, scale=scale, target_hw=target_hw)

        # 反归一化
        out = out * norm_std + norm_mean

        # 还原维度
        if len(latent["samples"].shape) == 4:
            out = out.squeeze(2)

        out = out.cpu().to(orig_dtype)

        if dev.type == "cuda":
            torch.cuda.empty_cache()

        return ({"samples": out},)

# ==========================================
# 节点注册
# ==========================================
NODE_CLASS_MAPPINGS = {
    "MinimaxH3LatentUpscalerNode2D": MinimaxH3LatentUpscalerNode2D,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MinimaxH3LatentUpscalerNode2D": "Minimax H3 Latent Upscaler (2D)",
}