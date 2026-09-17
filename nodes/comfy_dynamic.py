"""Shared ComfyUI DynamicVRAM helpers for the MiniMax H3 latent upscalers."""

import torch
import torch.nn as nn

import comfy.model_management as model_management
import comfy.model_patcher
import comfy.ops


def resolve_devices(requested_device):
    """Honor an explicit CPU request; otherwise use ComfyUI's configured devices."""
    requested_device = torch.device(requested_device)
    if requested_device.type == "cpu":
        cpu = torch.device("cpu")
        return cpu, cpu

    return (
        model_management.get_torch_device(),
        model_management.unet_offload_device(),
    )


def _replace_one(module, device, dtype):
    """Create an equivalent Comfy manual_cast layer without copying initialized weights."""
    ops = comfy.ops.manual_cast

    if type(module) is nn.Conv2d:
        return ops.Conv2d(
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=module.bias is not None,
            padding_mode=module.padding_mode,
            device=device,
            dtype=dtype,
        )

    if type(module) is nn.Conv3d:
        return ops.Conv3d(
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=module.bias is not None,
            padding_mode=module.padding_mode,
            device=device,
            dtype=dtype,
        )

    if type(module) is nn.Linear:
        return ops.Linear(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            device=device,
            dtype=dtype,
        )

    if type(module) is nn.GroupNorm:
        return ops.GroupNorm(
            module.num_groups,
            module.num_channels,
            eps=module.eps,
            affine=module.affine,
            device=device,
            dtype=dtype,
        )

    return None


def replace_with_comfy_ops(module, device, dtype):
    """Recursively replace weight-bearing PyTorch layers with Comfy manual_cast layers."""
    for name, child in list(module.named_children()):
        replacement = _replace_one(child, device, dtype)
        if replacement is not None:
            setattr(module, name, replacement)
        else:
            replace_with_comfy_ops(child, device, dtype)
    return module


def build_comfy_model(factory, state_dict, requested_device, dtype, strict=True):
    """
    Build on meta (avoids a duplicate initialized CPU model), replace supported layers
    with Comfy manual_cast equivalents, then load through CoreModelPatcher semantics.
    """
    load_device, offload_device = resolve_devices(requested_device)

    # The original constructors perform harmless initialization (including zero_module).
    # On meta this creates no backing storage, so we avoid a second full model allocation.
    with torch.device("meta"):
        model = factory()

    replace_with_comfy_ops(model, offload_device, dtype)
    model.eval()

    patcher = comfy.model_patcher.CoreModelPatcher(
        model,
        load_device=load_device,
        offload_device=offload_device,
    )

    result = model.load_state_dict(
        state_dict,
        strict=strict,
        assign=patcher.is_dynamic(),
    )
    model.requires_grad_(False)
    return patcher, result


def estimate_upscale_activation_memory(
    input_shape,
    target_hw,
    hidden_channels,
    dtype,
    temporal_chunk=None,
    temporal_overlap=0,
):
    """
    Conservative activation reservation for Comfy's model loader.

    This is intentionally an estimate: DynamicVRAM manages weights, while the
    upscaler's feature maps remain ordinary CUDA allocations. Reserving their
    approximate peak prevents Comfy from filling otherwise-needed VRAM with weights.
    """
    b, _c, t, h, w = input_shape
    out_h, out_w = target_hw
    bytes_per = torch.empty((), dtype=dtype).element_size()

    if temporal_chunk is not None and t > temporal_chunk:
        work_t = min(t, temporal_chunk + 4 * max(0, temporal_overlap))
    else:
        work_t = t

    peak_h = max(h, out_h)
    peak_w = max(w, out_w)
    feature = b * hidden_channels * work_t * peak_h * peak_w * bytes_per

    # A residual block can transiently hold input, normalized/intermediate, and output.
    # Add latent/output bookkeeping headroom as well.
    latent = b * _c * t * peak_h * peak_w * bytes_per
    return int(feature * 3.25 + latent * 2.0)


def force_unload_patcher(patcher):
    """
    Legacy force-unload behavior, but keep Comfy's loaded-model bookkeeping coherent.

    ComfyUI currently has no public single-patcher unload helper. Use its LoadedModel
    registry when available, then fall back to patcher.detach for older versions.
    """
    loaded_models = getattr(model_management, "current_loaded_models", None)
    if loaded_models is not None:
        for i in range(len(loaded_models) - 1, -1, -1):
            loaded = loaded_models[i]
            if getattr(loaded, "model", None) is patcher:
                loaded.model_unload()
                loaded_models.pop(i)
                return True

    try:
        patcher.detach(unpatch_all=False)
        return True
    except TypeError:
        patcher.detach()
        return True
