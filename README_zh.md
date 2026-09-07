<p align="center">
  <a href="./README.md">English</a> ·
  <a href="./README_zh.md"><strong>中文</strong></a>
</p>

<div align="center">

# ComfyUI Minimax H3 Latent Upscaler

**Minimax H3 视频生成专用 Latent 神经网络放大节点**
学习型 · 高保真 · 2D 与 3D 双版本

</div>

## 📰 更新动态

- [2026-08-28] 🚀 **新增组合节点 — MMH3 Split Upscale**：新增三个节点（`MMH3 Temporal Split Params`、`MMH3 Spatial Split Params`、`MMH3 Split Upscale`），对 H3 视频做**分块式「高清重采样」放大**。该组合是在 **Comfyui-MMH3-UltimateUpscale** 项目基础上拆解优化而来——把 AV latent 沿时间切块、沿空间分块，逐块重采样后再无缝拼接；相比原版主要增强：更稳的拼接（`seam_denoise` + probe 门控二道缝修补）、颜色零漂移（空间+时间两级颜色匹配 + 首块/每 chunk 钉源）、三重时间锚（frame-0 / motion / identity）、更易用（空间参数简化为重叠率/渐变率百分比并自动吸附网格）、更轻量（移除内置放大模型，改接外部预放大 latent）。
- [2026-08-26] 🚀 **3D 节点显存与使用优化**：采用零拷贝模型加载，并接入 ComfyUI 原生 `soft_empty_cache` 与显式 `.contiguous()`，降低加载与推理期间的显存/内存峰值；推理后卸载到 CPU 改为可选开关 `force_unload`（默认开启）；`enable_chunking` 更名为 `enable_temporal_chunking`，分块有效帧数由 24 提升至 32；模型文件现可放入子目录（PR #30）；`target dimensions` 与 `megapixels` 上限分别放宽到 8192 像素与 16 MP。
- [2026-08-23] 🚀 **3D 节点改进**：新增 `enable_chunking` 开关（短片段可关闭，使用全上下文推理）；修复时间分块边缘伪影——采用 Replicate Padding 与重叠区域加权融合，消除末尾帧闪烁；新增 ROCm（AMD 显卡）后端支持，设备选项新增 `rocm`。
- [2026-08-21] 🚀 **3D 节点优化**：节点执行后自动将模型卸载到 CPU，为后续二次采样释放显存；宽高分别独立对齐到 `align` 网格（默认 32），修复底部光带问题；归一化/反归一化由原地操作改为标准运算，保留时间分块以支持长视频。
- [2026-08-19] 🚀 **3D 节点重构**：三种缩放模式（`scale by multiplier`、`target dimensions`、`megapixels`）合并进单一节点；修复特定模式下造成的比例不统一，修复各类特定尺寸造成的边缘伪影；新增示例工作流供参考。
- [2026-08-18] 🔥 **精度选择**：2D / 3D 节点均支持 `fp32` / `fp16` / `bf16` 推理。
- [2026-08-17] 🎉 **初始发布**：Minimax H3 Latent Upscaler 2D + 3D 双节点、双语 README、内联示例。

一个 ComfyUI 自定义节点，用训练好的神经网络（而非简单插值）对 **Minimax H3** 的 VAE latent
（24 通道）进行放大。它的**主要目的是加速高分辨率视频的生成**，并提升画质：

- **跳过「解码 → 像素放大 → 再编码」的慢速往返。** Minimax H3 自带约 5B 参数的笨重 VAE，
  对 latent 做解码和再编码都很耗时；直接在 latent 空间放大，就彻底省掉了这一往返开销。
- **支持更快的生成流程：** 先低分辨率生成（latent token 少得多），用本节点放大 latent，再在
  目标分辨率下二次采样 / 重绘精修。

同时，它**避免了直接对 latent 插值（双线性/双三次）带来的鬼影、重影问题**，作用类似于
**LTX2.3** 中的 latent 放大模型。

⚠️ 它节省的是**时间，不是显存** —— 精修阶段仍在目标分辨率下运行，峰值显存与直接高清生成接近，
收益纯粹是更快出片。

提供两个节点变体，均注册在 `video/MinimaxH3` 分类下：

- **Minimax H3 Latent Upscaler (2D)**：2D 残差主干 + 穿插的时序 3D 卷积，仅做空间（H×W）放大，
  时间维度保持不变，轻量且快速。使用简单的 `scale` 倍数（1.0×–4.0×）。
- **Minimax H3 Latent Upscaler (3D)**：纯 3D 卷积主干（3D 残差块 + 时序卷积 + 三线性插值），
  联合处理时空体，时间一致性更强，但算力/显存开销更高。一个节点内置**三种缩放模式**：
  - `scale by multiplier`：经典 `scale` 倍数（1.0×–4.0×）。
  - `target dimensions`：直接填目标像素 `width` / `height`。
  - `megapixels`：填目标总像素（百万像素，例如 `1.2`），保持原宽高比。
  `target dimensions` 与 `megapixels` 两种模式会按可配置的像素网格对齐输出，并自动反推等效倍率。

> 3D 节点在尺寸模式下会在内部计算出等效 `scale` 并喂给同一个训练好的模型，因此 1.0×–4.0× 之间
> 任意目标都能用。

> 两个节点均**只支持放大**（等效 `scale >= 1.0`）。`scale = 1.0` 返回输入原样；等效倍率小于 1.0 会报错。

### MMH3 Split Upscale（组合节点）

> **来源：** 该组合是在 **Comfyui-MMH3-UltimateUpscale** 项目基础上**拆解优化**而来。原本的单节点被拆成三个可组合节点（时间切分 / 空间切分 / 主放大），
> 并在原有分块逻辑之上，针对拼接稳定性、色彩一致性、时间连续性、易用性与轻量化做了系统性增强。

一套独立的**三节点组合**，在扩散采样器内部做**分块式「高清重采样」放大**（而非用预训练放大网络）。
它接收 H3 的 **AV latent**（嵌套：视频 24 通道 + 音频 32 通道），沿**时间切块**、沿**空间分块**，
对每一块单独跑采样器，再用缝降噪、两级颜色匹配与时间锚点把结果**无缝拼接**回去，避免出现接缝和鬼影。
两个 `* Split Params` 节点负责配置切分方式，并喂给主节点 `MMH3 Split Upscale`（两者均可选——
不连接即为整段/整帧单次处理）。它需要 `model` + `conditioning` + `sampler`/`sigmas`，即会在更高
分辨率上重新跑一遍采样。

**相比原版主要增强（拆解优化点）：**

- **🔧 更稳的拼接（Seam / Ghosting）**：缝邻域降噪上限（`seam_denoise`）+ probe 门控的**二道缝修补**（`seam_polish`），消除接缝与鬼影；在高降噪 + 快速运动下，运动物体也不再被切断（"断肢"）。
- **🎨 颜色零漂移（Color Zero-Drift）**：**空间 + 时间两级颜色匹配**，并对**首块**与**每个 chunk** 做全局色基准钉源（`grade_pin`），彻底解决块间闪烁、左上角发青等问题。
- **⏱️ 更强的时间连续性（Temporal Continuity）**：`frame-0 锚` + `motion 锚` + `identity 锚` **三重时间锚**，高降噪时自动兜底身份一致性，人物不漂移。
- **🧩 更易用（Usability）**：空间参数由原本的 9 项精简为**百分比**（重叠率 / 渐变率），并自动吸附 latent 网格；`negative` 紧跟 `conditioning` 布局，连线更直观。
- **🪶 更轻量（Lightweight）**：移除内置放大模型，改接**外部预放大 latent**，显存更可控、节点更专注。

**新增 / 优化的模块（相对于原项目）：**

| 模块 | 新增 / 优化的内容 |
| :--- | :--- |
| **预防（Prevention）** | 冻结预填重叠带 + **三重时间锚**（frame-0 锚 + motion 锚 + identity 锚）+ cross-fade 拼接——让接缝与分叉在生成之初就不出现。 |
| **校正（Correction）** | **空间/时间两级颜色匹配** + 首块源参考 + 每块（chunk）全局钉源（`grade_pin`）——保证各瓦片、各时间块之间的色彩与亮度一致，杜绝块间闪烁与局部发青。 |
| **抗分叉（Anti-forking）** | `seam_denoise` 上限——在高降噪 + 快速运动下，缝邻域以*中等*降噪续写邻居内容，防止运动物体被切断（"断肢"）；配合 probe 门控的二道缝修补，接缝与鬼影一并消除（建议 0.5–0.8；1.0 = 关闭）。 |
| **修复（Fixes）** | **每瓦片独立 Guider** + 裁剪 keyframes，以及**修复版 probe 门控去缝**（`seam_polish`）——每个瓦片获得正确作用域的条件/关键帧，去缝门控不再过度或不足。 |
| **简化（Simplification）** | `overlap` / `fade` 改为**百分比参数**（重叠率 / 渐变率，与分辨率无关），替代原来的绝对像素值，并自动吸附 latent 网格；`negative` 紧跟 `conditioning`，布局更直观。 |

---

## 📸 示例

**视频放大对比**

<video src="examples/Minimax_h3_latent_Upscaler_001.mp4" controls width="640"></video>

**图像放大对比**

![](examples/Minimax_h3_latent_Upscaler_002.jpg)

---

## 📁 项目结构

```text
Comfyui_Minimax_h3_latent_Upscaler/
├── examples/
│   ├── Minimax_h3_latent_Upscaler_001.mp4
│   └── Minimax_h3_latent_Upscaler_002.jpg
├── workflow_templates/
│   └── minimax_h3_r2v_Latent Upscaler example workflow.json  # ComfyUI 模板示例工作流
├── nodes/
│   ├── __init__.py                       # 合并 2D/3D/组合节点映射
│   ├── minimax_h3_latent_upscaler_2d.py  # 2D 主干 + Temporal 3D Conv（倍数模式）
│   ├── minimax_h3_latent_upscaler_3d.py  # 纯 3D 卷积，支持 3 种缩放模式
│   └── MMH3_Split_Upscale.py             # MMH3 Split Upscale 组合（基于 Comfyui-MMH3-UltimateUpscale 拆解优化）
├── README.md
├── README_zh.md
└── __init__.py
```

> 模型权重**不**随仓库提供，请按下方「模型放置」说明放入 ComfyUI 模型目录。

---

## 🚀 核心特性

- ✅ **学习型 latent 放大** — 针对 Minimax H3 latent 训练的神经网络，比双线性/双三次插值清晰得多。
- ✅ **两种主干** — 追求速度选轻量的 **2D** 版，追求时间一致性选 **3D** 版。
- ✅ **3D 节点三种输出尺寸方式** — 支持 `scale by multiplier` / `target dimensions` /
  `megapixels`，均带像素网格对齐与宽高比锁定。
- ✅ **分块式高清重采样（MMH3 Split Upscale 组合）** — 把 H3 的 AV latent 沿时间切块 + 空间分块，
  在更高分辨率上重采样，配合缝降噪、两级颜色匹配与时间锚点，避免接缝与鬼影。
- ✅ **24 通道 Minimax H3 latent** — 使用训练时一致的逐通道均值/标准差做归一化。
- ✅ **自动识别模型结构** — 直接从权重读取通道数、块数、时序配置与卷积核大小，无需手动配置。
- ✅ **鲁棒的权重加载器** — 支持 `.safetensors` 与 `.pth`；自动 FP8→FP16；兼容合并权重中的
  `upscaler.` 前缀。
- ✅ **可选精度与设备** — `cuda`/`cpu` 与 fp32/fp16/bf16 选项。
- ✅ **即插即用** — 标准 ComfyUI 节点，不改变现有工作流。

推理时强制关闭注意力（`attn=False`）以提升速度与稳定性；已加载模型按 `(名称, 设备, 精度)`
缓存，重复调用开销很低。

---

## 📦 安装

1. 将仓库克隆到 ComfyUI 的 `custom_nodes` 文件夹：

   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler.git
   ```

2. 所需依赖（`torch`、`einops`、`safetensors`）在标准 ComfyUI 环境中已自带，无需额外安装。

3. 重启 ComfyUI。

---

## 🤖 模型放置

节点从这里扫描并加载权重：

```text
ComfyUI/models/latent_upscale_models/
```

把你的 Minimax H3 latent 放大模型权重（`.safetensors` 或 `.pth`）放进该目录，节点下拉框会自动列出。

预训练权重可在此下载：
[huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler)

加载器会自动识别结构，只要权重存储的结构匹配，2D 与 3D 节点可共用同一份权重。

---

## 🧩 使用方法

从 `video/MinimaxH3` 菜单中添加所需节点，连入一个 `LATENT`，选择模型、设置缩放模式/参数，再解码即可。

**典型流程：**
- **快速预览：** `[Minimax H3 Latent] → [H3 Latent Upscaler] → [VAE 解码]`
- **高品质 / 省时（推荐）：** `[低清 Latent] → [H3 Latent Upscaler] → [二次采样重绘/精修] → [VAE 解码]`

相比「[Latent] → [VAE 解码] → [像素放大] → [VAE 编码] → …」这种朴素做法，直接在 latent 空间
放大跳过了昂贵的 VAE 解码/编码往返。Minimax H3 的约 5B 参数 VAE 让解码与再编码都明显偏慢，时间
主要就省在这里。同时也避免了直接对 latent 插值（双线性/双三次）造成的**鬼影 / 重影**问题。

⚠️ **省时间，不省显存：** 精修仍在目标分辨率下运行，峰值显存与直接高清生成接近，收益纯粹是更快出片。

### 节点参数 — 2D

| 参数 | 类型 | 默认值 | 范围 / 选项 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| `latent` | LATENT | — | — | 输入的 Minimax H3 latent（(B,C,T,H,W) 或 (B,C,H,W)） |
| `model_name` | 下拉框 | 自动 | 扫描到的文件 | `latent_upscale_models/` 中的模型 |
| `scale` | FLOAT | 2.0 | 1.0 – 4.0（步进 0.1） | 空间放大倍数 |
| `device` | 下拉框 | cuda | cuda / cpu | 推理设备 |
| `precision` | 下拉框 | fp32 | fp32 / fp16 / bf16 | 推理精度 |

**输出：** `LATENT` — 放大后的 latent，可直接送 VAE 解码。

### 节点参数 — 3D

| 参数 | 类型 | 默认值 | 范围 / 选项 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| `latent` | LATENT | — | — | 输入的 Minimax H3 latent（(B,C,T,H,W) 或 (B,C,H,W)） |
| `model_name` | 下拉框 | 自动 | 扫描到的文件 | `latent_upscale_models/` 中的模型 |
| `mode` | 下拉框 | `scale by multiplier` | `scale by multiplier` / `target dimensions` / `megapixels` | 输出尺寸选择方式 |
| `scale` | FLOAT | 2.0 | 1.0 – 4.0（步进 0.05） | `mode` 为 `scale by multiplier` 时使用 |
| `width` | INT | 1280 | 64 – 8192（步进 8） | 目标像素宽（`target dimensions` 使用） |
| `height` | INT | 704 | 64 – 8192（步进 8） | 目标像素高（`target dimensions` 使用） |
| `megapixels` | FLOAT | 1.0 | 0.1 – 16.0（步进 0.1） | 目标总百万像素（`megapixels` 使用），保持宽高比 |
| `align` | INT | 32 | 1 – 512 | 像素网格对齐：输出 W/H 会分别独立取整到该值的倍数（如 16/32/64） |
| `enable_temporal_chunking` | BOOLEAN | True | True / False | 长视频时间分块以压低显存峰值；短片段可关闭，使用全上下文推理 |
| `force_unload` | BOOLEAN | True | True / False | 推理后将模型卸载到 CPU，为后续节点释放显存；若重复使用本节点可关闭，避免重复加载开销 |
| `device` | 下拉框 | cuda | cuda / rocm / cpu | 推理后端（ROCm 需要支持 HIP 的 PyTorch 构建） |
| `precision` | 下拉框 | fp32 | fp32 / fp16 / bf16 | 推理精度 |

**输出：** `LATENT` — 放大后的 latent。

> **选哪个节点？** 帧间已较稳定、求快时选 **2D**；需要更强运动/时间一致性，或想直接按目标
> 像素/百万像素出图，选 **3D**。

### 节点参数 — MMH3 Split Upscale（组合）

一套**分块式「高清重采样」**的 H3 视频放大方案，**在 Comfyui-MMH3-UltimateUpscale 项目基础上拆解优化而来**。和 2D/3D 节点（用预训练放大网络）不同，这套组合会在
latent 的更高分辨率版本上重跑扩散采样器，并切块以适配显存。它操作的是 H3 的 **AV latent**
（嵌套：视频 24 通道 + 音频 32 通道），需要 `model` + `conditioning` + `sampler`/`sigmas`。

**`MMH3 Temporal Split Params`** — 沿时间怎么切（对齐 H3 原生的 5+17·m 帧/token 网格）：

| 参数 | 类型 | 默认值 | 范围 / 选项 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| `chunk_frames` | INT | 73 | 5 – 100000 | 每块帧数（自动对齐 H3 token 网格） |
| `temporal_overlap_frames` | INT | 22 | 0 – 100000 | 相邻块之间的重叠帧数 |
| `anchor_strength` | FLOAT | 0.999 | 0.0 – 1.0 | 时间锚强度 |
| `motion_anchor_frames` | 下拉框 | 22 | 0 / 5 / 22 / 39 | 运动锚长度（帧） |
| `identity_anchor_frames` | INT | 24 | 0 – 240 | 身份锚间距 |

输出：`temporal_split_param`（自定义类型，喂给主节点）。

**`MMH3 Spatial Split Params`** — 每帧怎么切瓦片：

| 参数 | 类型 | 默认值 | 范围 / 选项 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| `tile_width` | INT | 512 | 64 – 16384（步进 32） | 瓦片宽（像素） |
| `tile_height` | INT | 512 | 64 – 16384（步进 32） | 瓦片高（像素） |
| `overlap_ratio` | FLOAT | 0.25 | 0.0 – 0.90 | 重叠占比（占瓦片） |
| `fade_ratio` | FLOAT | 0.50 | 0.0 – 1.0 | 重叠区内的渐变带宽度 |
| `min_tile_size` | INT | 256 | 0 – 16384（步进 32） | 最小瓦片边长；避免边缘出现极小瓦片 |
| `seam_denoise` | FLOAT | 1.0 | 0.1 – 1.0 | 缝邻域降噪上限；<1 时高降噪下防止快速物体被切断（建议 0.5–0.8） |

输出：`spatial_split_param`（自定义类型）+ `grid_preview`（字符串，显示计算出的瓦片/重叠网格）。

**`MMH3 Split Upscale`** — 主调度节点：

| 参数 | 类型 | 默认值 | 范围 / 选项 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| `model` | MODEL | — | — | 用于重采样的扩散模型 |
| `conditioning` | CONDITIONING | — | — | 正向条件 |
| `negative` | CONDITIONING | — | 可选 | 负向条件 |
| `latent` | LATENT | — | — | 输入的 H3 AV latent（嵌套：视频 24 通道 + 音频 32 通道）；仅支持 batch 1 |
| `noise` | NOISE | — | — | 采样器用的噪声源 |
| `sampler` | SAMPLER | — | — | 采样器 |
| `sigmas` | SIGMAS | — | — | 调度序列（sigma schedule） |
| `cfg` | FLOAT | 1.0 | 0.0 – 100.0 | 无分类器引导强度 |
| `temporal_split_param` | 自定义 | — | 可选 | 来自 `MMH3 Temporal Split Params` |
| `spatial_split_param` | 自定义 | — | 可选 | 来自 `MMH3 Spatial Split Params` |
| `seam_polish` | 下拉框 | off | off / auto / all | 对检测到的接缝额外做去缝处理 |
| `color_match` | BOOLEAN | True | True / False | 跨瓦片/块的两级颜色匹配 |

**输出：** `LATENT` — 放大后的 H3 AV latent，可直接送 VAE 解码。

> **注意：** 两个 `* Split Params` 节点都可不接（即为整段/整帧单次处理，不做时间/空间切分）。
> 该组合会在更高分辨率上重跑采样器，因此以额外算力换取长视频 / 高分辨率下的显存余量。

---

## 🧪 模型与架构

- **Latent 格式：** 24 通道 Minimax H3 VAE latent，推理前按训练均值/标准差逐通道归一化，推理后反归一化。
- **默认检测结构**（若权重不同会被自动覆盖）：`in_channels=24`、`in_blocks=12`、
  `out_blocks=12`、`base_channels=512`、`dropout=0.1`、`temporal_every=2`、
  `temporal_kernel=5`、`attn=False`。
- **插值方式：** 2D 节点用双线性特征插值；3D 节点用三线性插值。
- **时间处理：** 两个节点均保持时间长度不变，仅放大空间分辨率（H×W）。

---

## 📊 训练数据

该放大模型在**近 8 万对样本**上训练（低分辨率 latent 与高分辨率目标配对），并在数据类型与缩放倍数上做了均衡，以提升泛化能力。

**按数据类型：**

| 数据类型 | 对数 | 占比 |
| :--- | :--- | :--- |
| 视频素材 | 约 70,000 对 | 约 87.5% |
| 2K 图像 | 约 8,000 对 | 约 10% |

**按缩放倍数（占比，约）：**

| 倍数 | 占比 | 说明 |
| :--- | :--- | :--- |
| 2× | 40% | 主力倍数 —— 最常见的实际使用场景 |
| 1.5× | 10% | — |
| 2.5× | 10% | — |
| 3× | 10% | — |
| 4× | 10% | — |
| 1.0×–4.0×（任意小数位） | 10% | 增强对中间任意倍数的泛化能力 |

重点放在 **2×（40%）**，对应最常见的实际使用场景；而 **1–4 之间的任意小数位占 10%** 这一设计，
是为了避免模型只对固定的 1.5×/2×/2.5×/3×/4× 几个档位过拟合，从而能在推理时处理 1.0×–4.0× 区间内
任意连续 `scale` 值。

---

## 🙏 致谢

本节点沿用了 **Ttl** 提出的「神经网络 latent 放大」思路
（[ComfyUi_NNLatentUpscale](https://github.com/Ttl/ComfyUi_NNLatentUpscale)，
https://github.com/Ttl）。本项目模型架构同时参考借鉴了 **LTX 2.3 Spatial Upscaler**
（`ltx-2.3-spatial-upscaler-x2-1.1.safetensors`）。感谢这些开源工作为本项目奠定基础。
