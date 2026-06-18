# differentiable_renderer

可微分的 Lightroom 风格调色 pipeline + 参数拟合器 + OpenCV 部署端。

给定一对 `(original.png, edited.png)`，求一组参数 θ 使得 `Render(original, θ) ≈ edited`；拟合用 PyTorch（需要梯度），最终渲染可以用配套的 [`opencv_renderer.py`](../opencv_renderer.py) 在生产环境跑（不依赖 PyTorch）。

> **重要**：早期版本用 Local Laplacian Filter 做边缘感知 highlights/shadows/clarity/texture，由于 LLF 内部多次 bilinear 上采样会累积低频化模糊，已被 **guided filter** 替换。两端（PyTorch 训练侧 + OpenCV 部署侧）使用相同的 guided filter 数学公式，参数 θ 可在两个渲染器间无损迁移。

## 模块文件

| 文件 | 作用 |
|------|------|
| [`renderer.py`](renderer.py) | `DifferentiableRenderer`——5-stage sRGB-native 全可微管线 |
| [`guided_filter.py`](guided_filter.py) | PyTorch 可微 guided filter（对应 `cv2.ximgproc.guidedFilter`） |
| [`losses.py`](losses.py) | `CompositeLoss`：linear L1 + LAB ab L1 + ΔE2000 + MS-SWD-LAB + 分桶 ΔE + 梯度域 L1 |
| [`color_spaces.py`](color_spaces.py) | sRGB↔linear↔XYZ↔LAB、Bradford CAT、CIEDE2000、CCT→白点 |
| [`crop_align.py`](crop_align.py) | 不可微的 SIFT 预对齐（target 通常被 LR crop 过） |
| [`amortized_init.py`](amortized_init.py) | siamese CNN，从 (orig, edited) 直接预测 θ 作为 fitter 的 warm start |
| [`fit.py`](fit.py) | 主拟合脚本（CLI） |
| [`local_laplacian.py`](local_laplacian.py) | **已弃用**，保留代码以备回退；renderer 不再 import |
| [`../opencv_renderer.py`](../opencv_renderer.py) | 纯 NumPy + `opencv-contrib-python` 的部署端渲染器，与 PyTorch 端**像素级一致** |

## 渲染管线

`DifferentiableRenderer.forward` 按以下顺序作用于 sRGB [0,1] 图：

```
Stage 1  曝光度 → tone curve (PCHIP) → 白色色阶 → 黑色色阶 → 对比度
Stage 1d guided-filter 边缘感知组：高光 → 阴影 → 清晰度 → 纹理
Stage 2  白平衡（Bradford CAT）
Stage 3  自然饱和度 → 饱和度 → 8 通道 HSL 混合 → 颜色分级
Stage 4  晕影
Stage 5  去朦胧 → 锐化 → 噪点消除
```

参数总数：49（Scalar 16 + 8×3 mixer + 9 color grading + 8-vec tone curve dy）。

### Guided-filter 边缘感知组

替代旧的 Local Laplacian。每个算子都用单通道亮度 `I = Rec.709 lum` 作为 guide：

| 算子 | radius | eps | 公式 |
|------|--------|-----|------|
| `_highlights_op` | 8 | 0.01² | `mask = clip((lum-0.5)/0.5)` → guided filter snap → 加性偏移 |
| `_shadows_op` | 8 | 0.01² | `mask = clip((0.5-lum)/0.5)` → guided filter snap → 加性偏移 |
| `_clarity_op` | 16 | 0.04² | `base = guided_filter(lum, x, large)` → unsharp `x + k·(x-base)` |
| `_texture_op` | 4 | 0.01² | `base = guided_filter(lum, x, small)` → unsharp `x + k·(x-base)` |

边缘感知 = halo 不会越过强边缘；同时**没有** LLF 的 bilinear 累积模糊，所以输出更清晰。

## 完整使用流程

```bash
# 安装 OpenCV contrib（包含 ximgproc.guidedFilter）
pip install opencv-contrib-python

# Step 1：训练 amortized init CNN（自监督，约 30–60 分钟）
#         CNN 从原图采样 θ → 用渲染器生成 (orig, edited, θ) 三元组 → 训练
python differentiable_renderer/amortized_init.py \
    --image_dir scraped_data_v4/train \
    --steps 20000 \
    --out amortized_init.pt

# Step 2：批量拟合（每张约 1–2 分钟）
python differentiable_renderer/fit.py \
    --data_dir scraped_data_v4/train \
    --output_dir fit_results_demo \
    --amortized_init amortized_init.pt \
    --device cuda

# Step 3：拿拟合好的 θ 用 OpenCV 渲染最终图
python opencv_renderer.py \
    scraped_data_v4/train/<sample_id>/original_captured_cropped.png \
    fit_results_demo/<sample_id>/params_v0.json \
    output.png
```

## fit.py 拟合 pipeline

每个样本依次跑：

1. **初始化**（优先级 低→高）：identity (全 0) → `--warm_start` 解析的 LR 参数 → `--amortized_init` 的 CNN 预测
2. **B&W 检测**：`detect_bw(target, thresh=0.03)` 看 RG/GB 通道平均绝对差是否 < 0.03。命中则 `saturation = vibrance = -100`，配合 `_saturation` 的 `factor.clamp(min=0)` 直接输出灰度
3. **多尺度 Adam**（默认 256→512，每尺度 500 步）：每个尺度用 `0.5^scale_idx` 衰减 LR；cosine 退火；gradient clipping (norm=1.0)
4. **块坐标下降 L-BFGS**（默认 2 sweep × 5 块 × 25 步）：strong-Wolfe 线搜索
5. **全局 L-BFGS 精修**（默认 60 步）：所有参数联合优化

5 个语义块（[`parameter_blocks`](fit.py)）：

```
wb     = [temp, tint]
color  = [vibrance, saturation, cg_shadow/midtone/highlight]
hsl    = [mixer_hue, mixer_sat, mixer_lum]    # 8-vec each
curves = [exposure, contrast, whites, blacks, tone_curve_dy]
local  = [highlights, shadows, clarity, texture, vignette,
          dehaze, sharpening, noise_red]
```

每个块的默认 Adam LR：`wb 0.5, color 0.3, hsl 0.5, curves 0.1, local 0.3`。

## fit.py CLI

```bash
python differentiable_renderer/fit.py [options]
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `--data_dir` | （必填） | 含若干 `<sample_id>/` 子目录，每个里要有 `original_captured_cropped.png` 和 `version_<v>/main_image.png` |
| `--sample_id` | None | 只跑这一张 |
| `--version` | 0 | edited 版本号 |
| `--output_dir` | `fit_results` | 每个样本输出 5 个文件（见下） |
| `--warm_start` | False | 用 `combined_details.json` 解析的 LR 参数初始化 |
| `--amortized_init` | None | CNN warm start 的 `.pt` 路径 |
| `--scales` | `256 512` | Adam 阶段多尺度 |
| `--adam_iters` | **500** | 每个尺度的 Adam 步数 |
| `--cd_sweeps` | 2 | coordinate descent 扫描遍数 |
| `--cd_block_iters` | 25 | 每块 L-BFGS 步数 |
| `--polish_iters` | 60 | 全局 L-BFGS 步数 |
| `--device` | `cuda` | |
| `--force` | False | 已有结果时覆盖（默认 skip） |

## 输出文件

每个样本目录：
- `params_v{v}.json`——人类可读的 θ（中文 LR 字段名），可直接喂给 `opencv_renderer.py`
- `rendered_v{v}.png`——`Render(orig, θ_fit)` 的全分辨率结果
- `compare_v{v}.png`——三联横向：`[orig | rendered | target]`
- `history_v{v}.json`——每步 loss 各分量
- `renderer_v{v}.pt`——`renderer.state_dict()`，可 `load_state_dict` 复用

## 损失函数

[`CompositeLoss`](losses.py) 默认权重：

```python
'lin_l1':    1.0    # 线性 RGB L1（曝光/白平衡的传感器域损失）
'lab_ab':    0.5    # LAB ab 通道 L1（只惩罚色度，不重复算亮度）
'de2000':    0.05   # 感知 ΔE2000（值域 0..30+，权重小一些平衡量级）
'ms_swd':    0.25   # 多尺度 Sliced Wasserstein in CIELAB（分布匹配）
'hue_de':    0.05   # 8 桶分色相 ΔE2000（与 mixer 几何对齐）
'grad_l1':   0.5    # 亮度梯度域 L1（惩罚模糊，驱动锐化/clarity）
```

## opencv_renderer.py（部署端）

把拟合得到的 `params_v0.json` 应用到原图，输出最终 PNG。

```bash
python opencv_renderer.py original.png params_v0.json output.png
```

也可以作为 Python 模块用：

```python
import json, cv2, numpy as np
from opencv_renderer import render

img = cv2.cvtColor(cv2.imread('original.png'), cv2.COLOR_BGR2RGB)
params = json.load(open('params_v0.json'))
out = render(img, params)        # (H, W, 3) uint8 RGB
cv2.imwrite('output.png', cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
```

**两端一致性已验证**：在 2008×1339 真实数据集图上，PyTorch 与 OpenCV 端的最大像素差异 = **1/255**（纯 float→uint8 舍入边界）。所以 fit 时 PyTorch 看到的图就是 OpenCV 最终输出的图，参数语义完全对齐。

## 已知 quirks 和兜底机制

**NaN 处理**：参数极端化（如 `temp` 推到 −100 → 1500K 边界、`exposure` → ±5 边界）可能让某层 forward 出 NaN，污染 Adam 的 m/v 矩。`clamp_params()` 先 `nan_to_num_(0)` 再 clamp，但只能修复参数本身——optimizer 状态被污染后必须重建：

```python
if not torch.isfinite(loss):
    renderer.clamp_params()
    optim = torch.optim.Adam(lr_groups)   # 清空 m/v
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(...)
    continue
```

每个 scale 最多重置 3 次，超过则放弃当前 scale。

**L-BFGS 数值崩溃**：strong-Wolfe 在 closure 里**不**调 `clamp_params`，会探到边界外。`_contrast` 公式在 `contrast≈129.5` 有奇点会 NaN；`_strong_wolfe` 在非光滑曲面上偶尔会触发 `IndexError: bracket[low_pos]`。`lbfgs_polish` 用 `try/except (RuntimeError, ValueError, IndexError)` 包住整个 `optim.step(closure)`，捕获后 `clamp_params` + skip 当前 block，不影响后续 block。

**Adam 在边界附近的死锁**：amortized init 把参数预测到 clamp 边界附近时，梯度持续指向边界外、optim 推过去、`clamp_params` 拉回来——loss 完全不动。`0.5^scale_idx` 的 LR 衰减能减轻，无法根除。

**B&W 模式半实现**：`detect_bw` 命中后 `saturation/vibrance = −100` 就让 `_saturation` 输出灰度（factor 被 clamp 到 0）。但 `_color_grading` 的 RGB 偏移和 `_color_mixer` 的 hue 通道仍可能引入色偏。如果发现"伪色"，需要在 `clamp_params` 里加入 `cg_shadow/midtone/highlight = 0` 和 `mixer_hue/sat = 0` 的强制重置。

## 调试单个失败样本

```python
import json, math
with open('fit_results_demo/<sid>/history_v0.json') as f:
    h = json.load(f)
nan_count = sum(1 for e in h
                if isinstance(e.get('total'), float) and math.isnan(e['total']))
print(f'{len(h)} entries, {nan_count} NaN')
print('first:', h[0])
print('last finite:', next(e for e in reversed(h)
                            if isinstance(e.get('total'), float)
                            and not math.isnan(e['total'])))
```

可视化检查 `compare_v0.png`：
- rendered ≈ orig 且参数全 0 → fitter 全程崩溃，连 amortized init 都没给出有效初值
- rendered 有调整但与 target 差距大 → 局部最优，可考虑 `--force` 重跑或加大 `--adam_iters`
- rendered 看起来比 target 模糊 → guided filter 的 radius 太小，调整 `_gf_cl_radius` 或 `_gf_tx_radius`

## 重新训练 amortized_init

如果改了渲染器的算子（比如调整 guided filter radius/eps、改了某个参数的 coefficient），旧的 `amortized_init.pt` 就不再对应当前管线。需要重新训练：

```bash
python differentiable_renderer/amortized_init.py \
    --image_dir scraped_data_v4/train \
    --steps 20000 \
    --out amortized_init.pt
```

训练是自监督的：脚本内部用当前的 `DifferentiableRenderer` 从原图随机采样 θ → 渲染合成 edited 图，所以**会自动跟随渲染器代码的变更**，不需要任何真实编辑数据。
