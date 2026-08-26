"""
Minimax H3 Latent Upscaler - ComfyUI inference node (pure 3D conv version)
- [终极整合] 引入时间分块(Temporal Chunking)开关，解决长视频显存峰值
- [终极整合] 修复分块边缘效应(Replicate Padding)，彻底解决末尾帧闪烁
- [终极整合] 推理结束后将模型踢回CPU，释放显存给后续节点(解决二次采样爆显存/速度慢)
- [终极整合] 强制长宽双向对齐到32的倍数，彻底解决底部光带(Light Banding)问题
- [终极整合] 放弃 In-place 原地操作，保障 FP16/FP32 下的数值精度与画质
- [新增] 完美支持 ROCm (AMD GPU) 加速
- 3 resize modes: scale by multiplier / target dimensions / megapixels
- Auto-detects model architecture (channels, blocks, temporal layout)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import folder_paths
import re
from einops import rearrange
from enum import Enum
from typing import TypedDict

# Try to import the new API
try:
    from comfy_api.latest import ComfyExtension, io
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
# ROCm / Device Helper Functions
# ==========================================
def _is_rocm_build():
    """ROCm PyTorch exposes AMD GPUs through the torch.cuda API."""
    return getattr(torch.version, "hip", None) is not None

def _resolve_device(backend):
    """Map the user-facing backend name to the device expected by PyTorch."""
    if backend == "cpu":
        return torch.device("cpu")
    if backend == "rocm":
        if not _is_rocm_build():
            raise RuntimeError("ROCm was selected, but this PyTorch build has no HIP/ROCm support. Install a ROCm build of PyTorch.")
        if not torch.cuda.is_available():
            raise RuntimeError("ROCm was selected, but PyTorch cannot access an AMD GPU. Check drivers and permissions.")
        return torch.device("cuda") # PyTorch uses 'cuda' device type for ROCm
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
# 3D network components 
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
# Pure-3D backbone with Temporal Chunking Switch
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

    def forward(self, x, scale=None, target_size=None, enable_chunking=True):
        if target_size is not None:
            size = target_size
        elif scale is not None:
            size = tuple(int(round(s * scale)) for s in x.shape[-3:])
        else:
            return x

        if size == x.shape[-3:]:
            return x

        B, C, T, H, W = x.shape
        
        tk = 0
        for b in self.in_blocks:
            if isinstance(b, TemporalConv):
                tk = b.dwconv.weight.shape[2]
                break
        
        overlap = tk  # 重叠帧数
        chunk = 24    # 每个分块的有效帧数
        
        if not enable_chunking or T <= chunk:
            return self._forward_seg(x, scale, size)

        print(f"[MinimaxH3-3D] temporal chunking: T={T} chunks={(T + chunk - 1) // chunk} overlap={overlap}")
        
        x_padded = F.pad(x, (0, 0, 0, 0, overlap, overlap), mode='replicate')
        
        out_full = torch.zeros(B, C, T, size[-2], size[-1], device=x.device, dtype=x.dtype)
        weight_full = torch.zeros(1, 1, T, 1, 1, device=x.device, dtype=x.dtype)
        
        start = 0
        while start < T:
            seg_start = start
            seg_end = min(T, start + chunk)
            
            # 【核心修复】输出范围包括重叠区域，确保分块之间有重叠
            out_start = max(0, seg_start - overlap)
            out_end = min(T, seg_end + overlap)
            
            # 在 padded 张量上的切片索引（需要额外的上下文）
            lo = max(0, out_start - overlap)
            hi = min(T + 2 * overlap, out_end + overlap)
            
            seg = x_padded[:, :, lo:hi]
            seg_size = (hi - lo, size[-2], size[-1])
            seg_out = self._forward_seg(seg, scale, seg_size)
            
            # 计算 seg_out 中对应 [out_start, out_end) 的索引
            s0 = (out_start + overlap) - lo
            s1 = s0 + (out_end - out_start)
            
            valid_out = seg_out[:, :, s0:s1]
            n_valid = out_end - out_start
            
            # 【核心修复】创建正确的权重：有效区域权重为 1，重叠区域权重渐变
            weight = torch.ones(n_valid, device=x.device, dtype=x.dtype)
            
            # 在 [out_start, seg_start) 区域（前重叠区域），权重从 0 递增到 1
            if seg_start > out_start:
                blend_len = seg_start - out_start
                weight[:blend_len] = torch.arange(1, blend_len + 1, device=x.device, dtype=x.dtype) / (blend_len + 1)
            
            # 在 [seg_end, out_end) 区域（后重叠区域），权重从 1 递减到 0
            if out_end > seg_end:
                blend_len = out_end - seg_end
                weight[-blend_len:] = torch.arange(blend_len, 0, -1, device=x.device, dtype=x.dtype) / (blend_len + 1)
            
            # 累加加权后的分块输出
            out_full[:, :, out_start:out_end] += valid_out * weight.view(1, 1, n_valid, 1, 1)
            weight_full[:, :, out_start:out_end] += weight.view(1, 1, n_valid, 1, 1)
            
            start += chunk
        
        # 归一化
        out_full = out_full / weight_full.clamp(min=1e-8)
        return out_full
            
        return torch.cat(outs, dim=2)

    def _forward_seg(self, x, scale, size):
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
# Model loading
# ==========================================
MODEL_CACHE = {}

def get_models_dirs():
    """Return every ComfyUI path registered for latent upscaler models."""
    return folder_paths.get_folder_paths(_LATENT_UPSCALE_FOLDER)

def scan_models():
    """Use ComfyUI's model discovery so extra_model_paths.yaml is honored."""
    names = [
        name
        for name in folder_paths.get_filename_list(_LATENT_UPSCALE_FOLDER)
        if os.path.splitext(name)[1].lower() in {".pth", ".safetensors"}
    ]
    if names:
        return names

    model_dirs = "; ".join(get_models_dirs())
    return [f"(no models found in registered latent_upscale_models paths: {model_dirs})"]

def resolve_model_path(name):
    """Resolve a model name against all registered ComfyUI model paths."""
    path = folder_paths.get_full_path(_LATENT_UPSCALE_FOLDER, name)
    if path is None:
        searched = "; ".join(get_models_dirs())
        raise FileNotFoundError(
            f"Model file not found: {name}. Searched: {searched}"
        )
    return path

def _load_raw_sd(path):
    if path.endswith('.safetensors'):
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
    cfg = {
        "in_channels": 24, "in_blocks": 12, "out_blocks": 12, "channels": 512,
        "dropout": 0.1, "attn": False, "temporal_every": 2, "temporal_kernel": 5,
    }
    conv_key = 'conv_in.weight'
    if conv_key in sd:
        cfg["in_channels"] = sd[conv_key].shape[1]
        cfg["channels"] = sd[conv_key].shape[0]

    in_ids, out_ids = set(), set()
    temporal_in_indices, temporal_out_indices = set(), set()
    for k in sd.keys():
        m = re.match(r'in_blocks\.(\d+)\.in_layers\.', k)
        if m: in_ids.add(int(m.group(1)))
        m = re.match(r'out_blocks\.(\d+)\.in_layers\.', k)
        if m: out_ids.add(int(m.group(1)))
        m = re.match(r'in_blocks\.(\d+)\.dwconv\.weight', k)
        if m: temporal_in_indices.add(int(m.group(1)))
        m = re.match(r'out_blocks\.(\d+)\.dwconv\.weight', k)
        if m: temporal_out_indices.add(int(m.group(1)))

    if in_ids: cfg["in_blocks"] = len(in_ids)
    if out_ids: cfg["out_blocks"] = len(out_ids)

    if temporal_in_indices or temporal_out_indices:
        cfg["temporal_every"] = 2
        for k in sd.keys():
            if 'dwconv.weight' in k and k.endswith('dwconv.weight'):
                cfg["temporal_kernel"] = sd[k].shape[2]
                break
    else:
        cfg["temporal_every"] = 0

    if any('attn' in k for k in sd): cfg["attn"] = True 
    cfg["attn"] = False
    return cfg

def load_model(name, device, precision):
    backend_lbl = _backend_label(device)
    cache_key = f"{name}::{backend_lbl}::{precision}"
    if cache_key in MODEL_CACHE:
        model = MODEL_CACHE[cache_key]
        print(f"[MinimaxH3-3D] 🔄 Loading model from cache to {device}")
        return model.to(device)

    path = resolve_model_path(name)

    raw_sd = _load_raw_sd(path)
    up_sd = _extract_upscaler_sd(raw_sd)
    cfg = _detect_arch(up_sd)

    model = LatentResizer3D(
        in_channels=cfg["in_channels"], in_blocks=cfg["in_blocks"], out_blocks=cfg["out_blocks"],
        channels=cfg["channels"], dropout=cfg["dropout"], attn=cfg["attn"],
        temporal_every=cfg["temporal_every"], temporal_kernel=cfg["temporal_kernel"],
    )
    model.load_state_dict(up_sd, strict=True)
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}.get(precision, torch.float32)
    model = model.to(device).eval().requires_grad_(False)
    if dtype != torch.float32:
        model = model.to(dtype)

    MODEL_CACHE[cache_key] = model
    print(f"[MinimaxH3-3D] Loaded upscale model: {name}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,} | "
          f"Attn: forced off | Temporal: {'on' if cfg['temporal_every'] > 0 else 'off'} "
          f"(every={cfg['temporal_every']}, kernel={cfg['temporal_kernel']}) | "
          f"Backend: {backend_lbl} | Precision: {precision}")
    return model

# ==========================================
# ComfyUI node (new API)
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

class MinimaxH3LatentUpscaler3D(io.ComfyNode):
    """Minimax H3 latent upscaler with pixel-space alignment and aspect-ratio lock."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MinimaxH3LatentUpscaler3D",
            display_name="Minimax H3 Latent Upscaler (3D)",
            category="video/MinimaxH3",
            search_aliases=["minimax", "h3", "latent", "upscale", "3d"],
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
                            io.Int.Input("width", default=1280, min=64, max=4096, step=8, tooltip="Target pixel width."),
                            io.Int.Input("height", default=704, min=64, max=4096, step=8, tooltip="Target pixel height.")
                        ]),
                        io.DynamicCombo.Option(UpscaleMode.MEGAPIXELS, [
                            io.Float.Input("megapixels", default=1.0, min=0.1, max=8.0, step=0.1, tooltip="Target megapixels (1024x1024 = 1MP).")
                        ])
                    ],
                ),

                io.Int.Input("align", default=32, min=1, max=512, step=1,
                             tooltip="Pixel-space alignment: output W/H are rounded to multiples of this value (e.g. 16/32/64). 32 is strictly recommended to avoid light banding."),
                
                io.Boolean.Input("enable_chunking", default=True,
                                 tooltip="Enable temporal chunking to save VRAM for long videos. Turn off for short videos (<16 frames) for pure full-context inference."),
                
                io.Combo.Input("device", options=["cuda", "rocm", "cpu"], default="cuda", 
                               tooltip="Execution backend. ROCm requires a HIP-enabled PyTorch build."),
                io.Combo.Input("precision", options=["fp32", "fp16", "bf16"], default="fp16"), 
            ],
            outputs=[
                io.AnyType.Output("latent", tooltip="Upscaled latent."),
            ],
        )

    @classmethod
    def execute(cls, latent: dict, model_name: str, mode: UpscaleConfig,
                align: int, enable_chunking: bool, device: str, precision: str) -> io.NodeOutput:

        if model_name.startswith('('):
            raise ValueError("Please place model files into the latent_upscale_models directory")

        selected_mode = mode["mode"]
        src = latent["samples"]
        orig_dtype = src.dtype
        was_4d = (src.dim() == 4)

        dev = _resolve_device(device)
        compute_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]

        s = src.to(device=dev, dtype=compute_dtype).clone()
        if was_4d:
            s = s.unsqueeze(2)  # (B, C, 1, H, W)

        b, c, t, h_in, w_in = s.shape
        downsample = VAE_DOWNSAMPLE

        # 1. Theoretical target size in PIXEL space
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

        # 2. Pixel-space alignment (强制长宽双向对齐，彻底解决底部光带)
        alignment = max(1, align)
        w_pixel_aligned = round(w_pixel_target / alignment) * alignment
        h_pixel_aligned = round(h_pixel_target / alignment) * alignment

        # 3. Snap to VAE grid so latent sizes are exact integers
        w_pixel_final = round(w_pixel_aligned / downsample) * downsample
        h_pixel_final = round(h_pixel_aligned / downsample) * downsample

        # 4. Back to LATENT space
        w_out = max(1, int(w_pixel_final // downsample))
        h_out = max(1, int(h_pixel_final // downsample))

        if effective_scale < 1.0 and (w_out < w_in or h_out < h_in):
            raise ValueError("This model only supports upscaling (effective scale >= 1.0).")

        if w_out == w_in and h_out == h_in:
            return io.NodeOutput(latent)

        print(f"[MinimaxH3-3D] Latent {w_in}x{h_in} -> {w_out}x{h_out} | "
              f"Pixels {w_out * downsample}x{h_out * downsample} | scale={effective_scale:.3f}")

        # 5. Inference
        model = load_model(model_name, dev, precision)
        norm_mean, norm_std = _make_norm_tensors(dev, compute_dtype)

        with torch.inference_mode():
            # 放弃 in-place 操作，保障精度与画质
            s_norm = (s - norm_mean) / norm_std
            # 传递 enable_chunking 参数
            out = model(s_norm, scale=effective_scale, target_size=(t, h_out, w_out), enable_chunking=enable_chunking)
            del s_norm
            out = out * norm_std + norm_mean

        if was_4d:
            out = out.squeeze(2)

        out = out.to(device="cpu", dtype=orig_dtype)

        # 推理结束后将模型踢回 CPU，释放显存给后续节点 (如 KSampler)
        if dev.type == "cuda":
            model.to("cpu")
            torch.cuda.empty_cache()
            print(f"[MinimaxH3-3D] ✅ Model offloaded to CPU. VRAM released for next node.")

        return io.NodeOutput({"samples": out})

# ==========================================
# New-API extension registration
# ==========================================
if USE_NEW_API:
    class MinimaxH3Extension(ComfyExtension):
        @override
        async def get_node_list(self) -> list[type[io.ComfyNode]]:
            return [MinimaxH3LatentUpscaler3D]

    async def comfy_entrypoint() -> MinimaxH3Extension:
        return MinimaxH3Extension()

# ==========================================
# Registration (both APIs)
# ==========================================
NODE_CLASS_MAPPINGS = {
    "MinimaxH3LatentUpscaler3D": MinimaxH3LatentUpscaler3D,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MinimaxH3LatentUpscaler3D": "Minimax H3 Latent Upscaler (3D)",
}