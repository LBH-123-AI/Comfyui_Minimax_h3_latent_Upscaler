"""
Minimax H3 Latent Upscaler - 按目标总像素(百万像素)放大 (纯3D卷积版本)
- 输入目标总像素 (百万像素)，节点按原图比例计算宽高，对齐到指定网格并反推放大倍率。
- 复用 h3_upscaler_common 中的模型加载与推理逻辑。
"""
import torch
import math
from comfy_extras.nodes_lt import LTXVSeparateAVLatent
from . import h3_upscaler_common as C


class H3LatentUpscalerNodeMegapixels:
    """按目标总像素(百万像素)放大视频 latent，保持原宽高比，保持音频 latent 不变。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "model_name": (C.scan_models(),),
                "target_megapixels": ("FLOAT", {
                    "default": 1.2, "min": 0.1, "max": 8.0, "step": 0.1,
                }),
                "align": ("INT", {
                    "default": 16, "min": 2, "step": 2,
                    "tooltip": "目标 latent 尺寸的整除网格 (常见 8/16/32)，向上取整对齐",
                }),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "precision": (["fp32", "fp16", "bf16"], {"default": "fp32"}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "run"
    CATEGORY = "video/MinimaxH3"

    def run(self, latent, model_name, target_megapixels, align, device, precision):
        """按目标总像素计算放大倍率并放大视频 latent，保持音频 latent 不变。"""
        if model_name.startswith('('):
            raise ValueError("请将模型文件放入 latent_upscale_models 目录")

        video_latent, audio_latent = LTXVSeparateAVLatent.execute(latent)
        source_samples = video_latent["samples"]
        source_height = int(source_samples.shape[-2])
        source_width = int(source_samples.shape[-1])

        # 按目标总像素 + 原比例计算目标像素宽高
        target_pixels = float(target_megapixels) * 1_000_000
        aspect = source_width / source_height
        target_height_pixels = math.sqrt(target_pixels / aspect)
        target_width_pixels = target_height_pixels * aspect

        # 对齐到 grid 倍数的 latent 尺寸
        target_width, target_height = C._align_latent_to_grid(
            target_width_pixels, target_height_pixels, align)
        if target_width < source_width or target_height < source_height:
            raise ValueError("target_megapixels 不能小于当前 latent 对应的总像素数")

        target_scale = math.sqrt((target_width * target_height) / (source_width * source_height))

        return C.run_upscale(
            video_latent, audio_latent, model_name, device, precision,
            target_width, target_height, target_scale)


NODE_CLASS_MAPPINGS = {
    "H3LatentUpscalerNodeMegapixels": H3LatentUpscalerNodeMegapixels,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3LatentUpscalerNodeMegapixels": "H3 Latent Upscaler (Megapixels)",
}
