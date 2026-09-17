"""CPU regression tests; run with python -m unittest discover -s tests -v."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from safetensors.torch import save_file


def import_node():
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.folder_names_and_paths = {"latent_upscale_models": ([], set())}
    path = Path(__file__).resolve().parents[1] / "nodes/minimax_h3_latent_upscaler_3d.py"
    spec = importlib.util.spec_from_file_location("upscaler_3d_under_test", path)
    module = importlib.util.module_from_spec(spec)
    # Exercise model code without booting ComfyUI or initializing CUDA.
    with patch.dict(sys.modules, {
        "folder_paths": folder_paths, "comfy.model_management": None,
        "comfy_api.latest": None,
    }):
        spec.loader.exec_module(module)
    return module


node = import_node()


def make_checkpoint():
    torch.manual_seed(42)
    model = node.LatentResizer3D(channels=32, in_blocks=2, out_blocks=2)
    sd = {key: value.to(torch.bfloat16) for key, value in model.state_dict().items()}
    # Match the actual Asirus layout, including BF16 biases next to FP16 weights.
    for key, value in model.state_dict().items():
        fallback = (
            key in ("conv_in.weight", "conv_out.weight")
            or key.startswith("embed.")
            or ".emb_layers." in key
            or key.endswith(".in_layers.0.weight")
            or ".out_norm." in key
            or ".norm." in key
        )
        if fallback:
            sd[key] = value.to(torch.float16)
    # Exactly representable in FP16, but lost by the former model.to(bfloat16).
    sd["conv_in.weight"].flatten()[0] = 1.0009765625
    return sd


class MixedPrecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.sd = make_checkpoint()
        self.path = Path(self.tmp.name) / "renamed_checkpoint.safetensors"
        save_file(self.sd, self.path)
        node.MODEL_CACHE.clear()
        self.addCleanup(node.MODEL_CACHE.clear)
        resolver = patch.object(
            node.folder_paths, "get_full_path_or_raise", return_value=str(self.path), create=True,
        )
        resolver.start()
        self.addCleanup(resolver.stop)

    def load(self, precision="bf16"):
        return node.load_model(self.path.name, torch.device("cpu"), precision)

    def test_mixed_checkpoint_preserves_every_parameter_value(self):
        model = self.load()
        for key, parameter in model.state_dict().items():
            with self.subTest(key=key):
                torch.testing.assert_close(parameter.float(), self.sd[key].float(), rtol=0, atol=0)
        self.assertEqual(model.conv_in.weight.dtype, torch.float32)
        self.assertEqual(model.conv_in.bias.dtype, torch.float32)
        self.assertEqual(model.in_blocks[0].in_layers[2].weight.dtype, torch.bfloat16)
        self.assertEqual(model.conv_in.weight.flatten()[0].item(), 1.0009765625)

    def test_fallback_convolution_accepts_bf16_without_fp16_overflow(self):
        model = self.load()
        x = torch.full((1, 24, 2, 3, 3), 80000.0, dtype=torch.bfloat16)
        actual = model.conv_in(x)
        expected = F.conv3d(x.float(), self.sd["conv_in.weight"].float(),
                            self.sd["conv_in.bias"].float(), padding=1)
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_forward_and_temporal_chunking_keep_output_head_precision(self):
        model = self.load()
        with torch.inference_mode():
            for frames, chunking in ((1, False), (3, False), (33, True)):
                with self.subTest(frames=frames, chunking=chunking):
                    x = torch.randn(1, 24, frames, 2, 3).bfloat16()
                    out = model(x, scale=2, target_size=(frames, 4, 6), enable_chunking=chunking)
                    self.assertEqual(out.shape, (1, 24, frames, 4, 6))
                    self.assertEqual(out.dtype, torch.float32)
                    self.assertTrue(torch.isfinite(out).all())

    def test_explicit_uniform_precisions_still_work(self):
        for precision, dtype in (("fp16", torch.float16), ("fp32", torch.float32)):
            with self.subTest(precision=precision):
                model = self.load(precision)
                self.assertEqual({p.dtype for p in model.parameters()}, {dtype})
                self.assertFalse(any(m._forward_pre_hooks for m in model.modules()))
                with torch.inference_mode():
                    out = model(torch.randn(1, 24, 2, 2, 3).to(dtype), scale=2,
                                target_size=(2, 4, 6))
                self.assertEqual(out.dtype, dtype)
                self.assertTrue(torch.isfinite(out).all())

    def test_node_returns_original_latent_dtype_for_images_and_videos(self):
        with patch.object(node.io, "NodeOutput", side_effect=lambda value: (value,)):
            for shape in ((1, 24, 2, 3), (1, 24, 3, 2, 3)):
                with self.subTest(shape=shape):
                    latent = {"samples": torch.randn(shape, dtype=torch.float32)}
                    result, = node.MinimaxH3LatentUpscaler3D.execute(
                        latent, self.path.name,
                        {"mode": node.UpscaleMode.SCALE_BY, "scale": 2.0},
                        align=32, enable_temporal_chunking=True, force_unload=True,
                        device="cpu", precision="bf16",
                    )
                    self.assertEqual(result["samples"].shape, (*shape[:-2], 4, 6))
                    self.assertEqual(result["samples"].dtype, torch.float32)
                    self.assertTrue(torch.isfinite(result["samples"]).all())

    def test_uniform_bf16_checkpoint_uses_original_inference_path(self):
        uniform = {key: value.bfloat16() for key, value in self.sd.items()}
        save_file(uniform, self.path)
        model = self.load()
        self.assertEqual({p.dtype for p in model.parameters()}, {torch.bfloat16})
        self.assertFalse(any(m._forward_pre_hooks for m in model.modules()))
        reference = node.LatentResizer3D(channels=32, in_blocks=2, out_blocks=2).eval()
        reference.load_state_dict(uniform)
        reference.bfloat16()
        x = torch.randn(1, 24, 2, 2, 3).bfloat16()
        with torch.inference_mode():
            torch.testing.assert_close(
                model(x, scale=2, target_size=(2, 4, 6)),
                reference(x, scale=2, target_size=(2, 4, 6)), rtol=0, atol=0,
            )

    def test_cache_and_cpu_offload_preserve_mixed_dtypes(self):
        model = self.load()
        expected = {key: p.dtype for key, p in model.named_parameters()}
        hook_count = sum(len(m._forward_pre_hooks) for m in model.modules())
        model.to("cpu")
        cached = self.load()
        self.assertIs(model, cached)
        self.assertEqual(expected, {key: p.dtype for key, p in cached.named_parameters()})
        self.assertEqual(hook_count, sum(len(m._forward_pre_hooks) for m in cached.modules()))

    def test_prefixed_checkpoint_and_strict_loading(self):
        save_file({f"upscaler.{key}": value for key, value in self.sd.items()}, self.path)
        self.assertEqual(self.load().conv_in.weight.flatten()[0].item(), 1.0009765625)
        node.MODEL_CACHE.clear()
        del self.sd["conv_out.weight"]
        save_file(self.sd, self.path)
        with self.assertRaisesRegex(RuntimeError, "Missing key"):
            self.load()


if __name__ == "__main__":
    unittest.main()
