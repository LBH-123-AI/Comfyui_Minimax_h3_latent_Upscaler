<p align="center">
  <a href="./README.md"><strong>English</strong></a> ·
  <a href="./README_zh.md">中文</a>
</p>

<div align="center">

# ComfyUI Minimax H3 Latent Upscaler

**Neural Latent Upscaler for Minimax H3 Video Generation**
Learned · High-fidelity · 2D & 3D Variants

</div>

A custom ComfyUI node that upscales **Minimax H3** VAE latents (24 channels) with a trained
neural network instead of naive interpolation. Its main purpose is to **accelerate
high-resolution video generation** and improve quality:

- **Skip the slow decode → pixel-upscale → encode round-trip.** Minimax H3 ships a heavy
  ~5B-parameter VAE, so decoding and re-encoding latents is expensive. Upscaling directly in
  latent space avoids that costly round-trip entirely.
- **Enable a faster generation pipeline:** generate at low resolution (far fewer latent tokens),
  upscale the latent with this node, then refine at the target resolution.

It also **avoids the ghosting / double-image artifacts** that naive latent interpolation
(bilinear/bicubic) introduces, and plays a role similar to the latent upscaler in **LTX2.3**.

⚠️ This saves **time, not VRAM** — the refinement pass still runs at the target resolution, so
peak memory is comparable to generating high-res directly. The win is purely speed.

Several node variants are provided, all registered under the `video/MinimaxH3` category:

Three ways to choose the output size, all registered under `video/MinimaxH3`:

**By scale factor** (the original nodes):
- **Minimax H3 Latent Upscaler (2D)** — a 2D ResBlock backbone with Temporal 3D-Conv layers
  inserted for temporal consistency. Spatial (H×W) upscaling; the time dimension is preserved.
  Lightweight and fast.
- **Minimax H3 Latent Upscaler (3D)** — a fully 3D-convolution backbone (3D ResBlocks +
  TemporalConv + trilinear interpolation). Processes the spatiotemporal volume jointly for
  stronger temporal coherence; heavier on compute/memory.

**By explicit size** (community-contributed, 3D backbone):
- **H3 Latent Upscaler (Target Resolution)** — set the target pixel `width`/`height` directly;
  the node aligns the latent to a configurable grid (8 / 16 / 32 …) and derives the scale factor
  automatically. Great when you know the exact output resolution.
- **H3 Latent Upscaler (Megapixels)** — set a target total pixel count in megapixels (e.g. `1.2`),
  keep the original aspect ratio, align to the grid, and derive the scale factor automatically.
  Great for "I just want ~1.2 MP" workflows.
- **H3 Latent Cond Sync (3D)** — a pass-through helper (no upscaling). It reads the current latent
  size and synchronizes `positive`/`negative` `CONDITIONING` image refs to that size, so the
  second-stage refine/re-sample pass won't hit a size mismatch. Place it right after an upscaler.

> The two "explicit size" nodes compute the equivalent `scale` internally and feed it to the same
> trained model, so any target between 1.0×–4.0× works. They also keep the audio latent untouched.

> The three upscaling nodes support **upscaling only** (`scale >= 1.0`). `scale = 1.0` returns the
> input unchanged; `scale < 1.0` raises an error. **H3 Latent Cond Sync** is not an upscaler — it
> just passes the latent through and resizes the conditioning to match.

---

## 📸 Examples

**Video upscale comparison**

<video src="examples/Minimax_h3_latent_Upscaler_001.mp4" controls width="640"></video>

**Image upscale comparison**

![](examples/Minimax_h3_latent_Upscaler_002.jpg)

---

## 📁 Project Structure

```text
Comfyui_Minimax_h3_latent_Upscaler/
├── examples/
│   ├── Minimax_h3_latent_Upscaler_001.mp4
│   └── Minimax_h3_latent_Upscaler_002.jpg
├── workflow_templates/
│   └── minimax_h3_r2v_Latent Upscaler example workflow.json  # example workflow for ComfyUI templates
├── nodes/
│   ├── __init__.py                       # merges all node mappings
│   ├── minimax_h3_latent_upscaler_2d.py  # 2D backbone + Temporal 3D Conv (scale mode)
│   ├── minimax_h3_latent_upscaler_3d.py  # pure 3D convolution (scale mode)
│   ├── h3_upscaler_common.py             # shared model load + 3D inference logic
│   ├── H3_latent_upscaler_resolution.py  # explicit target resolution node (community)
│   ├── H3_latent_upscaler_megapixels.py  # megapixels target node (community)
│   ├── H3_latent_upscaler_3d_v3.py       # Cond Sync passthrough: sync CONDITIONING to latent size (community)
│   └── H3LatentResize.py                 # shared conditioning resize helper
├── README.md
├── README_zh.md
└── __init__.py
```

> The model weights are **not** included in this repo. Place them in your ComfyUI models folder
> (see Model Placement below).

---

## 🚀 Key Features

- ✅ **Learned latent upscaling** — neural network trained for Minimax H3 latents, far sharper
  than bilinear/bicubic interpolation.
- ✅ **Two backbones** — pick the fast **2D** variant or the temporally-coherent **3D** variant.
- ✅ **Arbitrary scale 1.0×–4.0×** — continuous `scale` with 0.1 step (default 2.0).
- ✅ **24-channel Minimax H3 latent** — uses the exact per-channel mean/std normalization from
  training.
- ✅ **Auto architecture detection** — reads `in_channels`, block counts, temporal config and
  kernel size straight from the checkpoint; no manual config needed.
- ✅ **Robust weight loader** — supports `.safetensors` and `.pth`; auto-converts FP8→FP16;
  tolerates the `upscaler.` prefix in merged checkpoints.
- ✅ **Flexible precision/device** — `cuda`/`cpu` and fp32/fp16/bf16 options.
- ✅ **Plug-and-play** — standard ComfyUI node, no changes to your existing workflow.

Inference forces attention **off** (`attn=False`) for speed and stability. Loaded models are
cached by `(name, device, precision)` so repeated runs stay cheap.

---

## 📦 Installation

1. Clone this repository into ComfyUI's `custom_nodes` folder:

   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler.git
   ```

2. Required dependencies (`torch`, `einops`, `safetensors`) are already present in a standard
   ComfyUI environment — no extra install needed.

3. Restart ComfyUI.

---

## 🤖 Model Placement

The nodes scan and load weights from:

```text
ComfyUI/models/latent_upscale_models/
```

Put your Minimax H3 latent upscaler checkpoint (`.safetensors` or `.pth`) there. It will appear
automatically in the node's `model_name` dropdown.

Pre-trained checkpoints are available at:
[huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler)

The loader auto-detects the architecture, so a single checkpoint works for both the 2D and 3D
nodes as long as the stored structure matches.

---

## 🧩 Usage

Add the node you need from the `video/MinimaxH3` menu, connect a `LATENT`, pick the model, set the
resize mode / scale, and decode.

**Typical workflow:**
- **Quick preview:** `[Minimax H3 Latent] → [H3 Latent Upscaler] → [VAE Decode]`
- **High quality / time-saving (recommended):** `[Low-res Latent] → [H3 Latent Upscaler] → [Refine / Re-sample] → [VAE Decode]`

Compared to the naive approach `[Latent] → [VAE Decode] → [Pixel Upscaler] → [VAE Encode] → …`,
upscaling directly in latent space skips the expensive VAE decode/encode round-trip. Minimax H3's
~5B-parameter VAE makes decode and re-encode notably slow, so this is where most of the time is
saved. It also avoids the **ghosting / double-image artifacts** that direct latent interpolation
(bilinear/bicubic) causes.

⚠️ **Saves time, not VRAM:** the refinement still runs at the target resolution, so peak memory is
roughly the same as generating high-res directly. The benefit is purely faster turnaround.

### Node Reference — 2D / 3D (classic)

| Parameter | Type | Default | Range / Options | Description |
| :--- | :--- | :--- | :--- | :--- |
| `latent` | LATENT | — | — | Input Minimax H3 latent (B,C,T,H,W) or (B,C,H,W) |
| `model_name` | dropdown | auto | scanned files | Checkpoint in `latent_upscale_models/` |
| `scale` | FLOAT | 2.0 | 1.0 – 4.0 (step 0.1) | Spatial upscale factor |
| `device` | dropdown | cuda | cuda / cpu | Inference device |
| `precision` | dropdown | fp32 | fp32 / fp16 / bf16 | Inference precision. fp16/bf16 use less memory and run faster; fp32 is most accurate |

**Output:** `LATENT` — the upscaled latent, ready for VAE decode.

### Node Reference — Target Resolution (3D, community)

| Parameter | Type | Default | Range / Options | Description |
| :--- | :--- | :--- | :--- | :--- |
| `latent` | LATENT | — | — | Input Minimax H3 latent (B,C,T,H,W) or (B,C,H,W) |
| `model_name` | dropdown | auto | scanned files | Checkpoint in `latent_upscale_models/` |
| `width` | INT | 768 | ≥ 1 | Target pixel **width** |
| `height` | INT | 432 | ≥ 1 | Target pixel **height** |
| `align` | INT | 16 | 2,4,6,… (step 2) | Latent-grid divisor (8/16/32…); target latent H×W is rounded up to a multiple of `align` |
| `device` | dropdown | cuda | cuda / cpu | Inference device |
| `precision` | dropdown | fp32 | fp32 / fp16 / bf16 | Inference precision |

**Output:** `LATENT` — the upscaled latent. The effective scale is derived from `width`/`height` and `align`.

### Node Reference — Megapixels (3D, community)

| Parameter | Type | Default | Range / Options | Description |
| :--- | :--- | :--- | :--- | :--- |
| `latent` | LATENT | — | — | Input Minimax H3 latent (B,C,T,H,W) or (B,C,H,W) |
| `model_name` | dropdown | auto | scanned files | Checkpoint in `latent_upscale_models/` |
| `target_megapixels` | FLOAT | 1.2 | 0.1 – 8.0 (step 0.1) | Target total megapixels; aspect ratio is kept |
| `align` | INT | 16 | 2,4,6,… (step 2) | Latent-grid divisor (8/16/32…); target latent H×W is rounded up to a multiple of `align` |
| `device` | dropdown | cuda | cuda / cpu | Inference device |
| `precision` | dropdown | fp32 | fp32 / fp16 / bf16 | Inference precision |

**Output:** `LATENT` — the upscaled latent. The effective scale is derived from the computed pixel size and `align`.

### Node Reference — Cond Sync (3D, community)

| Parameter | Type | Default | Range / Options | Description |
| :--- | :--- | :--- | :--- | :--- |
| `latent` | LATENT | — | — | Input latent (already upscaled by a previous node). Passed through unchanged |
| `positive` | CONDITIONING | optional | — | Positive conditioning; image refs are resized to the latent's current size |
| `negative` | CONDITIONING | optional | — | Negative conditioning; image refs are resized to the latent's current size |

**Outputs:** `(LATENT, CONDITIONING, CONDITIONING)` — the unchanged latent + resized positive + resized negative.

> This node does **no upscaling** — it only syncs the conditioning to the latent's current
> resolution. Chain it right after an upscaler so the second-stage refine/re-sample pass gets
> matching sizes.

> **Which node to pick?** Use **2D / 3D (scale)** for a simple `1.0×–4.0×` factor. Use
> **Target Resolution** when you know the exact output WIDTH×HEIGHT. Use **Megapixels** for a
> "I just want ~1.2 MP" workflow. Use **Cond Sync** after an upscaler when the second-stage refine
> needs the `CONDITIONING` resized to match. The two explicit-size nodes share one model via
> `h3_upscaler_common.py`.

---

## 🧪 Model / Architecture

- **Latent format:** 24-channel Minimax H3 VAE latent, normalized per-channel with the training
  mean/std before inference and de-normalized after.
- **Default detected architecture** (overridden automatically if the checkpoint differs):
  `in_channels=24`, `in_blocks=12`, `out_blocks=12`, `base_channels=512`, `dropout=0.1`,
  `temporal_every=2`, `temporal_kernel=5`, `attn=False`.
- **Interpolation:** the 2D node uses bilinear feature interpolation; the 3D node uses trilinear.
- **Temporal handling:** both variants preserve the time dimension (only H×W are scaled).

---

## 📊 Training Data

The upscaler was trained on **~80,000 paired samples** (a low-resolution latent paired with its
high-resolution target), balanced across modalities and scale factors to maximize generalization.

**By data modality:**

| Modality | Pairs | Share |
| :--- | :--- | :--- |
| Video clips | ~70,000 | ~87.5% |
| 2K images | ~8,000 | ~10% |

**By upscale factor (scale distribution, approximate):**

| Scale | Share | Note |
| :--- | :--- | :--- |
| 2× | 40% | Dominant factor — the most common real-world case |
| 1.5× | 10% | — |
| 2.5× | 10% | — |
| 3× | 10% | — |
| 4× | 10% | — |
| 1.0×–4.0× (arbitrary decimals) | 10% | Improves generalization to in-between / non-fixed scales |

The heavy emphasis on **2× (40%)** matches the most common practical use case, while the **10% spread
of arbitrary decimal scales between 1 and 4** prevents overfitting to the fixed 1.5×/2×/2.5×/3×/4×
buckets — letting the model handle any continuous `scale` in the 1.0×–4.0× range at inference time.

---

## 🙏 Acknowledgments

This node follows the neural-latent-upscaling approach pioneered by
[ComfyUi_NNLatentUpscale](https://github.com/Ttl/ComfyUi_NNLatentUpscale) by **Ttl**
(https://github.com/Ttl). The model architecture also draws on and references the
**LTX 2.3 Spatial Upscaler** (`ltx-2.3-spatial-upscaler-x2-1.1.safetensors`).
Thanks to both projects for the open-source foundation this work builds upon.
