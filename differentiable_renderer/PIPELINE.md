# PIPELINE.md — 当前调色 pipeline 数据流详解

本文档描述一对 `(original.png, edited.png)` 进入系统之后，数据**逐阶段经过的每一步处理**，以及每一步是如何参数化、如何被优化的。是 `README.md` 的补充：README 偏重模块/CLI，本文偏重 **"一张图怎么从 input 走到 output"**。

> 与上一版相比的主要更新（2026-05）：新增 `chroma_pa` / `detail_pa` 两个 one-sided 损失，新增 multi-start 包装、LR-annealing-on-plateau、chroma+detail micro-fit 末段、scale 1024、BW shortcut、Adam divergence detection、跨 scale 的 robust restore。`_detail_transfer` 改为 additive luminance-only。

---

## 0. 全景图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              输入 (input)                                    │
│   original.png (RAW-converted sRGB)        edited.png (LR 导出 sRGB)         │
└──────────────────┬──────────────────────────────┬───────────────────────────┘
                   │                              │
                   ▼                              ▼
            ┌────────────────────────────────────────────┐
            │ (A) 不可微预处理：crop_align.py            │
            │   SIFT + 单应性 → 把 original crop 到与    │
            │   edited 一致的视野                        │
            └────────────────────┬───────────────────────┘
                                 │ orig_cropped, edited
                                 ▼
            ┌────────────────────────────────────────────┐
            │ (B) Multi-start 拟合器 (fit.py)            │
            │     fit_single → 多次 _fit_attempt         │
            │     取 full-res default-loss 最低的一次    │
            │                                            │
            │  _fit_attempt 的内部流水：                 │
            │   B1. 启发式 + init_strategy 初始化        │
            │       (warm / zero / perturb)              │
            │   B2. B&W 与 vignette / detail 启发式      │
            │   B3. multi-scale Adam (256→512→1024)      │
            │       带 LR annealing + divergence reset   │
            │   B4. block coordinate-descent L-BFGS      │
            │   B5. global L-BFGS polish                 │
            │   B6. chroma+detail micro-fit (boost loss) │
            │   B7. best-state restore (default loss)    │
            └────────────────────┬───────────────────────┘
                                 │ θ_fit (57 个标量)
                                 ▼
            ┌────────────────────────────────────────────┐
            │ (C) 输出 dump                              │
            │   params_v0.json     ← 人类可读 LR 字段名  │
            │   rendered_v0.png    ← R(orig, θ_fit)      │
            │   compare_v0.png     ← orig|rendered|tgt   │
            │   history_v0.json    ← 每步 loss 各分量    │
            │   renderer_v0.pt     ← state_dict          │
            └────────────────────┬───────────────────────┘
                                 │ params_v0.json
                                 ▼
            ┌────────────────────────────────────────────┐
            │ (D) 部署：opencv_renderer.py（纯 NumPy）   │
            │   读 JSON → 在生产环境跑同样 57 维 θ       │
            │   两端像素级一致（误差 ≤ 1/255）           │
            └────────────────────────────────────────────┘
```

---

## A. 预处理：crop_align.py

LR 通常会先 crop 再调色，所以 `original.png` 的视野往往比 `edited.png` 大。直接对齐会让 fitter 把"裁掉的部分"也试图调色，把 loss 推到错误极小值。

| 步骤 | 实现 |
|------|------|
| 特征点 | `cv2.SIFT_create(nfeatures=3000)` 在两张灰度图上提取 |
| 匹配 | BF + Lowe ratio (0.7) |
| 几何 | `cv2.findHomography(..., RANSAC, 5.0)` 把 edited→orig 坐标 |
| 提取 box | 把 edited 四个角投影到 orig，取 axis-aligned bbox |
| 兜底 | 匹配少于 10 → 返回 None，跳过 crop |

**Hard constraint**：只支持轴对齐 crop，不支持 LR 的旋转/透视校正。

输出：`PIL.Image` 已经 crop 到与 edited 同视野的 orig；下游所有阶段都基于这张图。

---

## B. 拟合器：fit.py

整个拟合器是 **per-image 的非凸优化**，无任何跨样本共享（除了可选的 amortized CNN warm-start）。

### B0. Multi-start（fit_single 外层）

`fit_single` 是一个 multi-start 包装器：默认跑 **N=2 个 _fit_attempt**，每个用不同的 `init_strategy`，最后按 **full-res 上的 default-loss** 选最优。

| init_strategy | 含义 |
|---------------|------|
| `warm`    | 标准初始化：identity → LR params → amortized CNN（同上一版默认） |
| `zero`    | 全零（identity），不喂 warm/CNN — 防止 CNN 把 Adam 锁进错误 basin |
| `perturb` | warm + amortized + Gaussian 扰动（curve/cg σ=0.05、temp σ=200、其余 σ=5） |

20-sample benchmark 上 7/16 样本 `zero` 比 `warm` 胜出，证明 amortized init 经常陷入错误 basin。`perturb` 默认不开启（`n_restarts=2` 时只用 warm + zero）。

**ranking metric**：每个 attempt 结束后用 *未* 加权重 boost 的 `CompositeLoss` 在 full-res 上算一次 loss，最低者获胜。这确保 micro-fit 不会因为它的 boosted-loss 而"骗过"最终的选择。

```python
fit_single(..., n_restarts=2, restart_strategies=None)
  for ri, strategy in enumerate(restart_strategies):
      torch.manual_seed(1234 + 7919 * ri)        # reproducible
      renderer, history = _fit_attempt(strategy=strategy, ...)
      full_loss = CompositeLoss(...)(renderer(eval_orig), eval_target)
      attempts.append((renderer, history, full_loss))
  return min(attempts, key=lambda a: a.loss)
```

### B1. 三层级初始化（precedence 低 → 高，`_apply_init_strategy`）

```
1. identity            : 所有 57 个参数 = 0（=渲染器恒等映射）
2. warm_start          : 解析 combined_details.json 的 LR 滑块 → init_from_lr_params
3. amortized_init      : 128px 缩略图喂给训练好的 CNN → 预测 57 维 θ
4. (perturb 时) +noise : 在 (3) 之后施加高斯扰动，保留 amortized basin 但跳出局部极小
```

`zero` strategy 跳过 (2)、(3)、(4)，纯 identity。

**注意**：(2) 和 (3) 都不可信复现 LR 真实输出（LR 在 RAW 域，本管线在 sRGB 域），它们只是把 fitter 的起点拉进 basin of attraction。

### B2. 启发式 pre-init

| 启发式 | 触发条件 | 动作 |
|--------|----------|------|
| `detect_bw_level == 'full_bw'` | 99-percentile chroma < 0.04 | `bw_mode=True`、`saturation/vibrance=-100`、清空所有色度参数（`mixer_hue/mixer_sat/temp/tint/cg_*`），并对它们置 `requires_grad=False` 让 optimiser 完全无视 |
| `detect_bw_level == 'near_bw'` | 99-percentile chroma < 0.10 | `saturation=-40, vibrance=-30`（不到 -50 边界以保留梯度） |
| `estimate_vignette` | 角/中心亮度比 → vignette ≤ -10 且比 CNN 低 5+ | 覆盖 CNN 预测，强制 `vignette=heuristic` |
| 细节正向 init | `bw_mode=False` 且 CNN 预测的 clarity/texture 偏弱 | `clarity ≥ 20, texture ≥ 8`（防止 Adam 通过 negative 细节去 game lin_l1） |

**BW shortcut**：当 `bw_mode=True` 时，`_fit_attempt` 把 orig 也按灰度加载（`load_image(grayscale=True)`），让 fitter 解决纯灰度→灰度问题，避免死掉的 chroma 通道给 luminance-only 算子的曲率引入噪声。saturation 已被 pin 在 -100，输出与彩色 orig 完全等价。

### B3. 渲染器前向（DifferentiableRenderer.forward）

每次 fitter 调用 loss 都会跑一遍这个 forward。**所有 57 个参数都参与梯度图**。Stage 顺序：

| Stage | 算子 | 关键数学 | 参数 |
|-------|------|----------|------|
| 1a | `_exposure` | `linear_to_srgb(srgb_to_linear(x) · 2^EV)`，linear 域 clamp 32 | `exposure ∈ [-5, 5]` |
| 1b | `_tone_curve` | 8-pt PCHIP（Fritsch-Carlson 单调），作用在 lum，chroma 按 `new_lum/lum` 比例缩放 | `tone_curve_dy[8] ∈ [-100, 100]` |
| 1c | `_whites` | mask = `clip((lum-0.75)/0.25)`，加 `mask · w/100 · 0.40` | `whites ∈ [-100, 100]` |
| 1c | `_blacks` | mask = `clip((0.25-lum)/0.25)`，对称 | `blacks ∈ [-100, 100]` |
| 1c | `_contrast` | Photoshop 公式 `factor·(x-0.5)+0.5`，c=contrast·2，奇点 c≈259 | `contrast ∈ [-100, 100]` |
| 1d | `_highlights_op` | mask = `clip((lum-0.5)/0.5)` → `guided_filter(lum, mask, r=8, ε=0.01²)` → 加性偏移 | `highlights ∈ [-100, 100]` |
| 1d | `_shadows_op` | mask = `clip((0.5-lum)/0.5)` → guided filter → 加性偏移 | `shadows ∈ [-100, 100]` |
| 2 | `_white_balance` | `temp_K = temp·50 + 5500` → Bradford CAT (D65 → target 白点) | `temp ∈ [-80, 250], tint ∈ [-200, 200]` |
| 3 | `_vibrance` | `boost = vib/100 · 1.5 · (1-sat)` 加性 | `vibrance ∈ [-100, 100]` |
| 3 | `_saturation` | `lum + factor·(x-lum)`，`factor.clamp(min=0)` | `saturation ∈ [-100, 100]` |
| 3 | `_color_mixer` | 8 通道 × (H/S/L)，cosine soft mask 在 hue 周向；`H_adj += eff·hue·0.30` | `mixer_hue/sat/lum[8]` 各 ∈ [-100, 100] |
| 3b | `_color_grading` | 3 个 luminance zone (shadow/midtone/highlight 软 mask) × RGB offset | `cg_shadow/midtone/highlight[3]` 各 ∈ [-100, 100] |
| 4 | `_vignette_op` | radial mask `1 + v/100 · dist · 0.6` | `vignette ∈ [-100, 100]` |
| 5 | `_dehaze_op` | `mean + (x-mean)·(1+d/80)` | `dehaze ∈ [-100, 100]` |
| 5 | `_sharpen` | unsharp σ=0.5，强度 `s/150 · 1.5` | `sharpening ∈ [0, 150]` |
| 5 | `_noise_reduction` | α-blend 与 σ=2 blur | `noise_red ∈ [0, 30]` |
| 后处理 | `_detail_transfer` | **Additive luminance-only 频率分离**：`out = rendered + (boost·Y_orig_hf - Y_rend_hf)`，`boost = 1 + (clarity+texture)/200`，σ 按 `short_side/512` 缩放 | 复用 clarity/texture |
| 终 | `.clamp(0, 1)` | | |

**全部 57 个参数**：

```
scalar (16):    exposure, contrast, highlights, shadows, whites, blacks,
                temp, tint, vibrance, saturation, texture, clarity, vignette,
                dehaze, sharpening, noise_red
mixer (24):     mixer_hue[8], mixer_sat[8], mixer_lum[8]
grading (9):    cg_shadow[3], cg_midtone[3], cg_highlight[3]
curve (8):      tone_curve_dy[8]   ← 直接当 8 维向量参与优化
                                     (x 位置固定 0,1/7,..,1)
```

**关键设计要点**：

- 所有阶段都在 sRGB [0,1] 空间，只有 exposure 和 white_balance 临时 round-trip 到 linear。
- 每个算子都没有 `if param == 0: return`——梯度永远流。
- 所有空间滤波（guided filter / Gaussian blur）半径都按 `short_side / 512` 比例缩放，保持感知一致性而不是像素级一致性。
- `_detail_transfer` 是 **additive luminance-only**（旧版是 multiplicative full-RGB）：`ΔY = boost·Y_orig_hf - Y_rend_hf`，再以标量加到三通道。三个理由：(1) 不需要 `Y_rend.clamp(min=ε)` 的除零守卫，L-BFGS strong-Wolfe 可以稳跑；(2) BW 路径上加性=乘性，不破坏黑白；(3) 旧版会把 orig 的预编辑色度泄露到边缘，灰度目标也会有红/绿余色 — 改 luminance-only 后边缘色度全部来自 rendered，正色。

### B4. 多尺度 Adam（`adam_phase`）

| 维度 | 默认值 |
|------|---|
| Scales | **`[256, 512, 1024]`**（在 long_side 维度 resize；之前是 [256, 512]） |
| Steps/scale | **800**（`--adam_iters`；之前是 300） |
| LR groups | wb 0.5, color 0.3, hsl 0.5, curves 0.1, local 0.3 |
| LR decay | `0.2 ^ scale_idx`（第 N 个 scale 的 LR = 0.2^N × base） |
| LR schedule | 5%→100% 线性 warmup（首 10% 步数），后 cosine 退火 |
| Adam eps | **1e-3**（不是默认 1e-8） |
| Grad clip | norm=1.0 全局 |
| NaN reset | `clamp_params()` + 重建 optimizer，每 scale 最多 3 次，超过弃 |
| Best track | 每步 forward 后比较 `log['total']`，更新 `best['sd']` |
| **Divergence reset** | 连续 40 步 `loss > 1.5×in_phase_min` → restore best + LR×0.5 + 重建 Adam state；最多 2 次 |
| **LR annealing** | 连续 30 步 best 没改善 0.1% → LR×0.5 + 重建 Adam state；最多 4 次 |

**Divergence detection**：50-sample benchmark 显示 26% 的拟合在 Adam 中段会"飞出去"（loss 突涨数百步才回落）。监测连续 `divergence_streak=40` 步 `loss > divergence_ratio×min`，触发 restore best + LR halve + 重建 Adam state（清掉触发飞出的 m, v）。`max_resets=2` 之后再触发就 break 出整个 phase（曲面对当前 LR 太陡，继续也是浪费 GPU）。

**LR annealing**：另一个失败模式是 Adam 在 plateau 上不断 oscillate。监测连续 `anneal_patience=30` 步 best loss 改善 < 0.1%，halve LR + 重建 Adam state（旧 m, v 是给原 LR 标定的，留着会越过新 LR 的 step size）。`max_anneals=4` 让 LR 最多降到原来的 1/16。

**为什么 eps=1e-3**：参数贴边时（如 `texture=0`）梯度长期为 0，`v_hat` 衰减到 0，下一次 non-zero gradient 会让 `lr·m_hat/√v_hat` 爆炸。eps=1e-3 floor 住分母。

**为什么 LR decay 0.2（不是 0.5）**：第 2/3 个 scale 应该 *refine* 不是 *re-explore*。0.5 的 LR 配 ms_ssim weight=3.0 会在 50 步内炸掉收敛解。

**Scale transition robustness**：进入新 scale 前 `_rebaseline_best`（在新分辨率重测 best）+ `_restore_if_worse`（如果当前 renderer 比 best 差，回退到 best），消除"在 256 上 best 是 1.2，在 512 上是 1.5"这种跨 scale 比较错误。

### B5. 块坐标下降 L-BFGS（`coordinate_descent`）

5 个语义 block：

```python
'wb'     : [temp, tint]
'color'  : [vibrance, saturation, cg_shadow/midtone/highlight]
'hsl'    : [mixer_hue, mixer_sat, mixer_lum]
'curves' : [exposure, contrast, whites, blacks, tone_curve_dy]
'local'  : [highlights, shadows, clarity, texture, vignette,
           dehaze, sharpening, noise_red]
```

默认 **`cd_sweeps=5`**（之前 2），每 sweep 跑全部 5 个 block，每 block `L-BFGS(strong_wolfe, max_iter=25)`。L-BFGS 只对它收到的 params 求梯度，其他 param 自然冻结。

`parameter_blocks()` 会过滤掉 `requires_grad=False` 的参数（bw_mode 冻结的色度参数），并把空 block 整体丢弃。否则 L-BFGS 的 `_gather_flat_grad` 会在 None grad 上崩。

**已知坑**：L-BFGS strong-Wolfe 在 closure 里**不**调 `clamp_params()`，可能探出边界。`_contrast` 在 `c≈259` 是奇点，`_strong_wolfe` 在非光滑曲面偶尔 `IndexError: bracket[low_pos]`。整个 `optim.step(closure)` 被 `try/except (RuntimeError, ValueError, IndexError)` 包住，捕获后 `clamp_params()` + `_restore_if_worse()` 把 renderer 拽回 best 再跳过该 block（否则后续 block 会继承被破坏的 params 级联失败）。

### B6. 全局 L-BFGS polish

默认 **`polish_iters=200`**（之前 60），所有 57 个参数联合优化。同 B5 的 try/except 兜底。`_track_best` **在 closure 内部**调用，而不是 step 之后 — strong-Wolfe 在一个 step 内会评估多次 candidate，最佳点经常出现在 line search 中段，只在 step 后采样会漏掉。这单项把"54% 的样本最终 loss > 历史最佳"修到接近 0%。

### B7. Chroma + Detail Micro-fit（`POLISH_BOOST_WEIGHTS`）

针对用户反馈"渲染图发灰、不够锐"的专门末段，默认 **`micro_fit_iters=200`**：

| 损失键 | 主流水权重 | Micro-fit 权重 | 作用 |
|--------|-----------|---------------|------|
| `chroma_pa` | 5.0 | **20.0** | per-pixel `relu(target_chroma − pred_chroma).mean()`，强推饱和度 |
| `detail_pa` | 5.0 | **20.0** | one-sided HF 能量差，强推 clarity/texture |
| `param_reg` | 0.5 | **0.05** | 放松"反高光爆灯/反细节破坏"的正则，允许极端参数值 |
| `hue_de` | 0.05 | **0.02** | 减半，给 chroma_pa 让出空间 |

实现要点：
1. 先 `_restore_if_worse` 拽回主流水的 best，再 `_rebaseline_best(loss_fn=boost_loss_fn)` 在 boosted surface 重测一次（否则 micro-fit 的 in-phase divergence/restore 会用旧的 default-loss best 当锚点，apples-to-oranges）。
2. Adam group LR 降到 main 流水的 **5%**（`group_lrs * 0.05`），refine-only 不 explore。
3. 复用最大 scale (1024) 的 orig/target，不再换分辨率。
4. micro-fit 跑完之后 `_rebaseline_best(loss_fn=loss_fn)` 重新切回 default loss，让 B8 的 final restore 对得上 multi-start 的 ranking metric。

### B8. Best-state restore

`_fit_attempt` 全程维护 `best = {'loss': ..., 'sd': renderer.state_dict()}`：
- Adam 每步 forward 后 `_track_best`
- L-BFGS 在 closure 内部 `_track_best`（不是 step 后）
- Scale 切换、CD/polish/micro-fit 之间用 `_restore_if_worse` 兜底

收尾时：如果当前 renderer 的 loss > best['loss']（比如 polish 过冲了），`load_state_dict(best['sd'])`。`bw_mode` 是 Python 属性而不是 state_dict 张量，`load_state_dict` 不会动它。

---

## Loss：CompositeLoss

每次 forward 的 prediction 和 target 都跑 11 个 component，加权和反传。

| 权重键 | 权重 | 物理含义 |
|--------|------|----------|
| `lin_l1` | 1.0 | linear-RGB L1（曝光/WB 的传感器域信号） |
| `lab_ab` | 0.5 | LAB ab L1（只罚色度，避免与亮度损失重复） |
| `de2000` | 0.05 | mean ΔE2000（值域 0-30+，权重小） |
| `ms_swd` | 0.25 | LAB 多尺度 Sliced Wasserstein（分布匹配，对 mis-alignment 容忍） |
| `hue_de` | 0.05 | 8 桶分色相 ΔE2000（与 mixer 几何对齐） |
| `hl_over` | 2.0 | 高光过冲的非对称 L1（在 clamp 杀梯度前给反向信号） |
| `param_reg` | 0.5 | 参数空间正则：罚 (a) 高光爆灯方向 (b) 反细节方向（texture<0、clarity<0、noise_red>0） |
| `lum_cdf` | 1.0 | 9 个分位点的 luminance 直方图匹配 |
| `ms_ssim` | 3.0 | 多尺度 SSIM（需要 short_side ≥ 161；否则 fallback 到 win_size=7） |
| **`chroma_pa`** | **5.0** | one-sided per-pixel chroma parity：`relu(t_chroma − p_chroma).mean()`，反 lab_ab "用整体降饱和换错位 win" 的偏好 |
| **`detail_pa`** | **5.0** | one-sided HF 能量 parity：`relu(\|HF_t\| − \|HF_p\|)`，反 Adam 把 clarity/texture park 在 0 |

**why param_reg 不是 0.01**：lin_l1 偏好"模糊+略错色"超过"清晰+正色"（边缘对不齐时模糊 L1 更低），0.01 罚不住，0.5 让"主动破坏细节"的 L1 成本占总 loss ~12%。

**why ms_ssim 不是 5.0**：5.0 在 scale 256 会单步爆炸（梯度方向相邻迭代剧烈摆动，gradient clipping 拦不住），3.0 贡献 ~20%。

**why chroma_pa per-pixel relu (而不是 mean-then-relu)**：mean-then-relu 在饱和度 mean 接近 target 时 = 0（即使一半像素严重欠饱和），per-pixel 让每个欠饱和像素都贡献梯度。20-sample benchmark 验证：per-pixel 把 sat 范围从 [-100, 15] 扩到 [-100, 40]。

**why detail_pa**：`param_reg` 会罚 clarity<0 / texture<0，但 0 是免费的。`detail_pa` 让 0 也有成本（只要 target 比 pred 多 HF 能量），逼 Adam 在合适时机推到正向。

---

## 输出文件 (C)

`--all_versions` 时，每个 sample 的**每个 version** 输出到独立子目录，镜像数据集结构：

```
{output_dir}/{sample_id}/version_{n}/
  params.json   rendered.png   compare.png   history.json   renderer.pt   fit_meta.json
```

| 文件 | 内容 | 用途 |
|------|------|------|
| `params.json` | 57 个参数（LR 中文键名 + tone_curve_dy / cg_* 等）+ `_crop_aligned` 标记 | 喂给 opencv_renderer.py |
| `rendered.png` | full-res `R(orig, θ_fit)` | 视觉验收 |
| `compare.png` | 横向三联 `[orig \| rendered \| target]` | 一眼诊断 |
| `history.json` | 每步 loss 各分量 + scale + iter | 离线分析收敛曲线 |
| `renderer.pt` | `renderer.state_dict()`（lossless，包含 tone_curve_dy 等所有张量） | 复用、避免 JSON round-trip 损失 |
| `fit_meta.json` | 质量元数据：`final_loss` / `param_l1` / `degenerate` / `loss_components` 等 | **过滤退化样本** |

> 不带 `--all_versions`（只拟合 `--version`）时也是同样的 `version_{n}/` 子目录布局，只是该样本仅落一个 version 子目录。

### fit_meta.json 与退化样本过滤

部分 target（典型如**局部打光 / 渐变滤镜 / AI 蒙版**等空间局部效果）超出本渲染器 57 个**全局**参数的表达能力。这类样本拟合时，任何全局调整都会让 loss 比 identity 更高，于是 multi-start + best-restore 会忠实地选中 **θ≈0（identity）**——参数几乎没动、loss 仍高。**这种「退化到 identity」的 (sample, version) 作为 SFT 监督是有害的**（等于教模型「这种输入不要调整」），应当从训练集剔除。

`fit_meta.json` 记录判别信号：

| 字段 | 含义 |
|------|------|
| `final_loss` | full-res default-loss 的 total（不是 history 末条的 boosted loss） |
| `param_l1` | `Σ\|θ\|`，identity 严格为 0；越小越接近「没调」 |
| `n_active_params` | `\|θ_i\| > 1e-3` 的参数个数 |
| `degenerate` | `final_loss > --degenerate_loss(默认5.0)` **且** `param_l1 < --degenerate_param_l1(默认5.0)` |

主程序打印时也会对退化样本标注 `⚠️ DEGENERATE`。后续构建 SFT 数据时，扫描所有 `fit_meta.json`、丢弃 `degenerate=true` 即可（原始 `final_loss` / `param_l1` 都已落盘，可用自定义阈值重新判定）。

> ⚠️ `history.json` 的最后一条来自 micro-fit phase（用的是 boosted loss，权重不是默认）。**不要直接拿 `history[-1]['total']` 当作 default loss 比较** — 应读 `fit_meta.json` 的 `final_loss`，或主程序打印的 `full-res: total=...`。

---

## D. 部署：opencv_renderer.py

把 `params_v{v}.json` 喂给纯 NumPy + `opencv-contrib-python` 的渲染器。

- 数学公式与 PyTorch 端**完全相同**（含 guided filter radius/eps、PCHIP 单调钳位、Bradford CAT 系数、additive luminance-only detail_transfer 等）。
- 不依赖 PyTorch / CUDA，可以打镜像跑生产。
- 实测在 2008×1339 真实数据图上，与 PyTorch 端最大像素差 = **1/255**（纯 float→uint8 舍入）。

---

## CLI 速查

```
python -m differentiable_renderer.fit \
    --data_dir try_new_dataset1/train \
    --output_dir fit_results \
    --amortized_init amortized_init.pt \
    --warm_start \
    --all_versions \             # 拟合每个样本的所有 version_*（输出镜像数据集结构）
    --scales 256 512 1024 \      # 默认；去掉 1024 可少跑 ~30% 时间
    --adam_iters 800 \           # 每个 scale 的 Adam 上限步数
    --cd_sweeps 5 \              # 块坐标下降轮数
    --cd_block_iters 25 \        # 每 block 的 L-BFGS max_iter
    --polish_iters 200 \         # 全局 L-BFGS 步数
    --micro_fit_iters 200 \      # 末段 chroma+detail boost；0 关闭
    --n_restarts 2 \             # multi-start 次数；1=不 multi-start
    --device cuda \
    --degenerate_loss 5.0 \      # fit_meta.json 退化判定：loss 阈值
    --degenerate_param_l1 5.0 \  #   且 Σ|θ| 阈值（两者同时满足→degenerate=true）
    --limit 20                   # 仅跑前 N 个样本做 smoke test
```

20-sample benchmark 实测：`n_restarts=2 + scales=[256,512,1024] + adam_iters=800 + cd_sweeps=5 + polish_iters=200 + micro_fit_iters=200`，平均 ~12 min/张，相对 Plan A baseline mean loss -10.6%（3.59 → 3.21），8 wins / 5 ties / 3 regressions（regression 全部是已知 loss>5 的硬样本）。

---

## 关键不变量（debug 时优先核验）

1. **57 参数**：`renderer.param_count() == 57`
2. **identity**：`DifferentiableRenderer()(img) ≈ img`（θ 全 0 时是恒等）
3. **梯度全连通**：任何 θ 分量的微扰都会改变 forward 输出（无 dead path）
4. **clamp 顺序**：`clamp_params()` 必须先 `nan_to_num_` 再 `clamp_`（clamp 会把 NaN 传播）
5. **scale 不变性**：guided filter / Gaussian blur 的有效 σ 按 short_side 缩放，所以 256 训出来的 θ 在 2K 上视觉一致
6. **deploy 一致性**：`opencv_renderer.render(orig, params)` 与 PyTorch `renderer(orig)` 像素差 ≤ 1/255
7. **detail_transfer 加性**：`R(orig, θ=0) ≈ orig`（boost=1 时 ΔY = Y_orig_hf - Y_rend_hf = 0）
8. **best 跨 phase 单调**：每个 phase 结束后 `best['loss']` 永远 ≤ phase 开始时的值（`_restore_if_worse` 保证）
9. **Multi-start ranking**：`fit_single` 返回的 renderer 在 full-res default-loss 上等于 `min` over attempts

---

## 当前已知的"打不到 0 loss"的根本原因

这是**渲染器 capacity 上限**，不是 fitter 实现问题：

1. **没有局部蒙版**：57 参数全是全局 scalar / 全局向量。LR 的径向/线性渐变滤镜、AI 蒙版、人脸提亮等都映射不进来。
2. **没有非线性 RAW 处理**：LR 在 demosaic 之后的 RAW 域工作，而本管线在 sRGB-native，曝光的高光 roll-off、shadow 的 noise floor 无法等价。
3. **detail_transfer 是 hack**：clarity/texture 实际作用范围只是高频幅度乘子（1×~2×），不能生成原图没有的细节，也无法压制（clamp 在 [0,100]）。luminance-only 加性版避免了色度泄露，但也仍然不能替代真正的 detail-synth。
4. **频率分离两段式**：rendered 的低频 + orig 的高频亮度，强边缘可能有可见的亮度阶跃（rendered 的 low-freq 与 orig 的 high-freq 不来自同一信号）。
5. **离散参数空间被 bypass**：很多 LR 滑块（如 detail panel 的 mask、效果 panel 的 grain）这里没有。

下次想"更准确地复现 target"，需要决策的是 **(a) 接受这些 capacity 上限并把 fitter 调到极致**（multi-start 数提到 3-4、micro-fit 加权重）还是 **(b) 扩展渲染器**（局部参数、神经残差等），而不是堆 fitter。
