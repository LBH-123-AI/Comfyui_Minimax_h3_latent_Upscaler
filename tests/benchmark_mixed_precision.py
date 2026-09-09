"""Benchmark the actual checkpoint on a local CUDA GPU; no ComfyUI server needed.

python tests/benchmark_mixed_precision.py /path/to/model.safetensors --output /tmp/h3-benchmark.json
The FP16 fallback variant is experimental and does not change node settings.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time

import torch

from test_mixed_precision import node


def gpu_snapshot():
    return subprocess.check_output([
        "nvidia-smi", "--query-gpu=name,memory.used,utilization.gpu,temperature.gpu",
        "--format=csv,noheader",
    ], text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 1:
        parser.error("repeats and warmup must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    torch.set_num_threads(4)
    torch.manual_seed(1234)
    report = {
        "checkpoint": str(args.checkpoint), "torch": torch.__version__,
        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(), "gpu_before": gpu_snapshot(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "warmup": args.warmup, "repeats": args.repeats,
        "input": "seeded synthetic normalized latents; no VAE or video-quality comparison",
        "cases": [],
    }
    with args.checkpoint.open("rb") as handle:
        report["checkpoint_sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
    sd = node._extract_upscaler_sd(node._load_raw_sd(str(args.checkpoint)))
    cfg = node._detect_arch(sd)
    node.folder_paths.get_full_path_or_raise = lambda *_: str(args.checkpoint)

    def build(variant):
        if variant == "mixed_fp32_fallback":
            return node.load_model(args.checkpoint.name, torch.device("cpu"), "bf16")
        model = node.LatentResizer3D(**cfg).eval().requires_grad_(False)
        model.load_state_dict(sd, strict=True)
        if variant == "uniform_bf16":
            model.bfloat16()
        elif variant == "mixed_fp16_fallback":
            node._configure_precision(model, sd, "bf16")
            for layer in model.modules():
                if isinstance(layer, (torch.nn.Conv3d, torch.nn.Linear, torch.nn.GroupNorm)):
                    if layer.weight.dtype == torch.float32:
                        layer.half()
        return model

    def save_report():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    cases = [
        ("short_384x256", (1, 24, 2, 8, 12), False),
        ("video_1280x768", (1, 24, 8, 24, 40), False),
        ("chunked_640x384", (1, 24, 33, 12, 20), True),
    ]
    with torch.inference_mode():
        for name, shape, chunking in cases:
            cpu_input = torch.randn(shape).bfloat16()
            target = (shape[2], shape[3] * 2, shape[4] * 2)
            case = {"name": name, "input_shape": shape, "target_size": target,
                    "chunking": chunking, "variants": {}}
            outputs = {}
            report["cases"].append(case)
            for variant in ("uniform_bf16", "mixed_fp32_fallback", "mixed_fp16_fallback"):
                print(f"Running {name}: {variant}", flush=True)
                model = build(variant).cuda()
                x = cpu_input.cuda()

                def forward():
                    return model(x, scale=2.0, target_size=target, enable_chunking=chunking)

                for _ in range(args.warmup):
                    out = forward()
                    torch.cuda.synchronize()
                    del out
                torch.cuda.reset_peak_memory_stats()
                times = []
                for _ in range(args.repeats):
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    out = forward()
                    torch.cuda.synchronize()
                    times.append((time.perf_counter() - started) * 1000)
                    del out
                peak_mib = torch.cuda.max_memory_allocated() / 1024**2
                out = forward()
                torch.cuda.synchronize()
                assert tuple(out.shape) == (shape[0], shape[1], *target)
                finite = bool(torch.isfinite(out).all())
                outputs[variant] = out.float().cpu()
                result = {
                    "median_ms": statistics.median(times), "samples_ms": times,
                    "peak_allocated_mib": peak_mib, "finite": finite,
                    "output_dtype": str(out.dtype),
                    "gpu_after": gpu_snapshot(),
                }
                case["variants"][variant] = result
                print(json.dumps({"case": name, "variant": variant, **result}), flush=True)
                model.cpu()
                node.MODEL_CACHE.clear()
                del out, model, x
                gc.collect()
                torch.cuda.empty_cache()
                save_report()

            # Compare against FP32 math with exactly the same quantized input/weights.
            # This measures numerical deviation, not perceptual video quality.
            if name == "short_384x256":
                print("Running FP32 numerical reference", flush=True)
                model = build("uniform_fp32").cuda()
                reference = model(cpu_input.float().cuda(), scale=2.0, target_size=target,
                                  enable_chunking=chunking).float().cpu()
                assert torch.isfinite(reference).all()
                for variant, output in outputs.items():
                    delta = output - reference
                    case["variants"][variant]["rmse_vs_fp32"] = delta.square().mean().sqrt().item()
                    case["variants"][variant]["max_abs_error_vs_fp32"] = delta.abs().max().item()
                del model, reference
                gc.collect()
                torch.cuda.empty_cache()
            save_report()
    report["gpu_after"] = gpu_snapshot()
    save_report()
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
