"""
Minimax H3 Latent - 条件同步节点 (3D)
- 仅做「透传 + 同步缩放 CONDITIONING」：不做任何放大。
- 放大请使用 H3 Latent Upscaler (Target Resolution) / (Megapixels) 节点。
- 本节点读取输入 latent（通常是已被放大后的）的视频空间尺寸，把 positive/negative
  CONDITIONING 中嵌入的图像/视频引用 resize 到对应像素尺寸，避免第二阶段重绘时
  因分辨率变化导致 PackedLayout 行数不一致、尺寸不匹配报错。
"""
import torch
from comfy_extras.nodes_lt import LTXVConcatAVLatent, LTXVSeparateAVLatent

try:
    from .H3LatentResize import _resize_conditioning
except ImportError:
    from H3LatentResize import _resize_conditioning

_H3_VAE_DOWNSAMPLE = 16  # 像素 = latent × 16


class H3LatentUpscalerNode3DV3:
    """透传 latent，并把 CONDITIONING 中的图像/视频引用 resize 到当前 latent 尺寸。"""

    @classmethod
    def INPUT_TYPES(cls):
        """声明节点输入：仅 latent，可选正负条件。"""
        return {
            "required": {
                "latent": ("LATENT",),
            },
            "optional": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
            },
        }

    RETURN_TYPES = ("LATENT", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("latent", "positive", "negative")
    FUNCTION = "run"
    CATEGORY = "video/MinimaxH3"

    def run(self, latent, positive=None, negative=None):
        """透传 latent，并将 CONDITIONING 中的图像引用同步到 latent 当前尺寸。"""
        # 分离 AV，仅取视频 latent 的空间尺寸作为目标像素尺寸（音频不参与同步）。
        video_latent, audio_latent = LTXVSeparateAVLatent.execute(latent)
        samples = video_latent["samples"]
        target_width = int(samples.shape[-1]) * _H3_VAE_DOWNSAMPLE
        target_height = int(samples.shape[-2]) * _H3_VAE_DOWNSAMPLE

        # 同步缩放 positive / negative 中的图像引用（尺寸相同则跳过，避免无意义计算）。
        out_pos = _resize_conditioning(positive, target_width, target_height)
        out_neg = _resize_conditioning(negative, target_width, target_height)

        # 透传：重建 AV latent 原样返回（视频/音频内容均不改变）。
        out_latent, = LTXVConcatAVLatent.execute(video_latent, audio_latent)
        return (out_latent, out_pos, out_neg)


NODE_CLASS_MAPPINGS = {
    "H3LatentUpscalerNode3DV3": H3LatentUpscalerNode3DV3,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3LatentUpscalerNode3DV3": "H3 Latent Cond Sync (3D)",
}
