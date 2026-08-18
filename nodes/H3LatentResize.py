"""
H3 Latent Resize utilities
提供 conditioning 中图像/视频引用的尺寸同步缩放，避免第二阶段重绘时因分辨率
变化导致 PackedLayout 行数不一致。
"""
import torch
import torch.nn.functional as F


def _resize_conditioning(conditioning, width, height):
    """
    同步缩放 CONDITIONING 中可能存在的 4D 图像张量引用到目标像素尺寸。

    ComfyUI 的 CONDITIONING 是一个 list/tuple，每个元素为 (cond_tensor, dict)。
  某些注入的图像/视频条件（如 IP-Adapter、CLIP Vision、关键帧引用等）会以 4D
    张量形式存放在 dict 中；当 latent 被放大后，这些引用需要同步 resize，否则
    后续采样节点可能因空间尺寸不匹配而报错。

    Args:
        conditioning: ComfyUI CONDITIONING 结构，或为 None。
        width: 目标像素宽度。
        height: 目标像素高度。

    Returns:
        缩放后的 CONDITIONING；如果输入为 None，返回 None。
    """
    if conditioning is None:
        return None

    out = []
    for item in conditioning:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            out.append(item)
            continue

        cond_tensor, params = item[0], item[1]
        new_params = dict(params) if isinstance(params, dict) else params

        if isinstance(new_params, dict):
            for key, value in list(new_params.items()):
                if isinstance(value, torch.Tensor) and value.ndim == 4:
                    # 4D 图像张量，常见于 (B, C, H, W) 或 (B, H, W, C)
                    # 只对空间尺寸与目标不同的张量进行缩放，避免无意义计算。
                    if value.shape[-2] != height or value.shape[-1] != width:
                        orig_dtype = value.dtype
                        value_f = value.to(torch.float32)
                        # 默认按 (B, C, H, W) 处理；若最后两维不是 H×W 会自然跳过
                        resized = F.interpolate(
                            value_f,
                            size=(height, width),
                            mode="bilinear",
                            align_corners=False,
                        )
                        new_params[key] = resized.to(orig_dtype)

        out.append([cond_tensor, new_params])

    return out
