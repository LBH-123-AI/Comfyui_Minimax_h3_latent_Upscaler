"""
Minimax H3 Latent Upscaler - 按目标分辨率放大 (纯3D卷积版本)
- 输入目标像素 width / height，节点自动对齐到指定网格并计算放大倍率。
- 复用 h3_upscaler_common 中的模型加载与推理逻辑。
"""
import torch
import math
from comfy_extras.nodes_lt import LTXVSeparateAVLatent
from . import h3_upscaler_common as C


class H3LatentUpscalerNodeResolution:
    """按目标像素分辨率放大视频 latent，保持音频 latent 不变。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "model_name": (C.scan_models(),),
                "width": ("INT", {
                    "default": 768, "min": 1, "step": 1,
                }),
                "height": ("INT", {
                    "default": 432, "min": 1, "step": 1,
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

    def run(self, latent, model_name, width, height, align, device, precision):
        """按目标像素分辨率计算放大倍率并放大视频 latent，保持音频 latent 不变。"""
        if model_name.startswith('('):
            raise ValueError("请将模型文件放入 latent_upscale_models 目录")

        video_latent, audio_latent = LTXVSeparateAVLatent.execute(latent)
        source_samples = video_latent["samples"]
        source_height = int(source_samples.shape[-2])
        source_width = int(source_samples.shape[-1])

        # 把目标像素分辨率对齐到 grid 倍数的 latent 尺寸
        target_width, target_height = C._align_latent_to_grid(width, height, align)
        if target_width < source_width or target_height < source_height:
            raise ValueError("目标分辨率不能小于当前 latent 分辨率")

        target_scale = math.sqrt((target_width * target_height) / (source_width * source_height))

        return C.run_upscale(
            video_latent, audio_latent, model_name, device, precision,
            target_width, target_height, target_scale)


NODE_CLASS_MAPPINGS = {
    "H3LatentUpscalerNodeResolution": H3LatentUpscalerNodeResolution,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3LatentUpscalerNodeResolution": "H3 Latent Upscaler (Target Resolution)",
}
