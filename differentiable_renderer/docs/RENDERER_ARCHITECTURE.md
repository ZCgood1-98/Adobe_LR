# Renderer 完整架构讲解

## 核心概念：为什么需要 Differentiable Renderer？

### 问题陈述

你有：
- **原图**：PNG sRGB（经过 Lightroom RAW→sRGB 渲染后保存）
- **编辑后图**：PNG sRGB（用户在 Lightroom 编辑后的结果）
- **编辑参数记录**：JSON（Lightroom 记录的参数值）
  - ⚠️ **但这些参数是针对 RAW 线性域优化的**

### 核心问题

RAW 和 PNG 是两个完全不同的域：

```
RAW 线性域              PNG sRGB 非线性域
(Lightroom 工作域)     (我们的 PNG 数据)

曝光 = +1 EV
  → 像素值 × 2.0       曝光 = +1 EV
    (物理倍增)           → 像素值 × 2^(1/2.2) ≈ 1.37×
                         (感知倍增)

高光区域处理：          高光区域处理：
  R=1.5 → 直接乘        R=1.0 (饱和)
  (线性可行)              → 溢出! (非线性失效)

结论：
LR 的参数值 ≠ PNG 上的最优参数值
```

### 解决方案：Differentiable Renderer

```
最优化问题：
    θ_png* = argmin_θ  L( Render(img_orig, θ), img_edited )
                         ↑ 可微函数        ↑ 目标

使用：
1. 搭建一个 PNG 域上的 sRGB-native 渲染器
   → 与 step3_assr_cot.py 中的 NumPy 版本一致
2. 加上 PyTorch，使其全可微
3. 用梯度下降找到最优参数 θ_png
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    DifferentiableRenderer                    │
│                      (nn.Module)                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  40个可学习参数 (nn.Parameter)                               │
│  ├─ 16个标量参数 (Stage 1-5)                                │
│  └─ 24个颜色混合器参数 (8 channels × 3 axes: H/S/L)        │
│                                                              │
│  5个处理阶段 (严格按 Lightroom 顺序)                         │
│  ├─ Stage 1: 基础曝光 (6个参数)                             │
│  ├─ Stage 2: 白平衡   (2个参数)                             │
│  ├─ Stage 3: 色彩调整 (3个参数 + 24个 mixer)               │
│  ├─ Stage 4: 效果     (1个参数)                             │
│  └─ Stage 5: 细节增强 (5个参数)                             │
│                                                              │
│  预计算的注册缓冲 (nn.register_buffer)                       │
│  ├─ 4个高斯核 (为 texture/clarity/sharpen/denoise)         │
│  └─ mixer 几何参数 (中心、宽度)                             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## 设计原则 (5个核心)

### 1. sRGB-Native (不做 Gamma 转换)

```python
# ❌ 错误做法（会伪色）
linear = srgb ** 2.2
adjusted = linear * 2.0
srgb_out = adjusted ** (1/2.2)

# ✓ 正确做法（step3 已证实有效）
srgb_out = srgb * (2.0 ** (ev / 2.2))
```

**为什么**：
- PNG 已是 sRGB，不需往返转换
- 亮度 mask 在 [0, 1] 范围天然有效
- 一次 gamma 即可，无伪色

### 2. 所有操作都可微 (Never Break Gradients)

```python
# ❌ 坏的做法
if param == 0:
    return img  # 断裂梯度！

# ✓ 好的做法
return img * (2.0 ** (param / 2.2))
# 当 param=0 时，2^0 = 1，恒等
# 梯度链条完整
```

**关键约束**：没有分支控制，所有路径都计算

### 3. 频率分解设计 (Unsharp Mask 族)

```
img = blur(img, σ) + (img - blur(img, σ))
       └─ 低频 ─┘    └────── 高频 ───────┘

不同 σ 提取不同频率：
  σ=2.0  → 纹理 (极细节)
  σ=10.0 → 清晰度 (中调)
  σ=0.5  → 锐化 (边缘)

叠加时各自作用在不同频率，互不干扰
```

### 4. 颜色空间选择：RGB ↔ HSL (软约束)

```
为什么选 HSL 而非 LAB / HSV？

HSL:
  ✓ 直观 (H=色相, S=饱和度, L=亮度)
  ✓ 与 Lightroom mixer 对齐
  ✓ 梯度计算简单
  ✗ L≠感知亮度 (但在 sRGB 域也不需要)

LAB:
  ✗ 需要 sRGB → XYZ 转换 (违反原则 1)
  ✗ 增加计算量

HSV:
  ✗ 饱和度定义不同 (不与 LR 兼容)
```

### 5. Gaussian Soft Mask (平滑色相边界)

```
颜色混合器中，为 Red 通道 (center=0°) 设计的 mask：

    mask(H)
      1.0 │     ╱╲
          │    ╱  ╲
     0.5  │   ╱    ╲
          │  ╱      ╲
      0.0 ├──┴───────┴────── H
         -40°  0°  +40°

设计：
  1. Gaussian（不是 step 函数）→ 平滑边界，无色块
  2. 色相是环形 (360° wrapping) → 处理跨越 0°/360° 的色相
  3. 饱和度保护 → 灰色 (S≈0) 像素排除在外（防伪色）
```

## 前向推理详细流程

### Stage 1: 基础曝光（6个参数）

```
输入: img sRGB [0,1]

Step 1.1: 曝光
  img' = img * 2^(exp / 2.2)  [指数缩放]

Step 1.2: 白色色阶
  lum = 0.2126*R + 0.7152*G + 0.0722*B  [Rec.709]
  mask = ((lum - 0.75) / 0.25).clamp(0, 1)  [只作用在最亮区]
  img'' = img' + mask * (whites / 100) * 0.20

Step 1.3: 黑色色阶
  mask = ((0.25 - lum) / 0.25).clamp(0, 1)  [只作用在最暗区]
  img''' = img'' + mask * (blacks / 100) * 0.20

Step 1.4-5: 高光 & 阴影 (亮度分区)
  hi_mask = ((lum - 0.5) / 0.5).clamp(0, 1)  [>0.5 的区域]
  img'''' = img''' + hi_mask * (highlights / 100) * 0.30
  
  sh_mask = ((0.5 - lum) / 0.5).clamp(0, 1)  [<0.5 的区域]
  img''''' = img'''' + sh_mask * (shadows / 100) * 0.30

Step 1.6: 对比度 (Photoshop 公式)
  c = contrast * 1.27  [-100,100] → [-127,127]
  factor = (259(c+255)) / (255(259-c))
  output = factor * (img''''' - 0.5) + 0.5  [围绕 0.5 的线性拉伸]
  
结果: img_stage1 ∈ [0,1]，已 clamp
```

### Stage 2: 白平衡（2个参数）

```
Step 2.1: 色温
  相对值: (K - 5500) / 50 ∈ ℝ
  
  R' = R * (1 + temp / 500)    [高温 > 偏暖]
  G' = G                        [不变]
  B' = B * (1 - temp / 500)    [高温 < 减弱蓝]

Step 2.2: 色调 (绿-品红轴)
  shift = tint / 800  [-150,150] → [-0.1875, 0.1875]
  
  R' = R * (1 + shift * 0.3)
  G' = G * (1 - shift)          [主要轴]
  B' = B * (1 + shift * 0.3)

结果: img_stage2，色温-色调均调好
```

### Stage 3: 色彩调整（26个参数）

#### Step 3.1: 自然饱和度

```
max_c = max(R, G, B)
min_c = min(R, G, B)
sat = (max_c - min_c) / max_c  # 原始饱和度 ∈ [0,1]

boost = (vibrance / 100) * (1 - sat)  # 反相权重
        ↑ 低饱和像素 boost 大
        ↓ 高饱和像素 boost 小

lum = 0.2126*R + 0.7152*G + 0.0722*B
delta = img - lum_broadcast

output = img + boost * delta  # 加色彩距离
```

**直观理解**：低饱和的灰色区域推离亮度轴，提升彩度

#### Step 3.2: 全局饱和度

```
lum = luminance(img)
factor = 1 + saturation / 100  # 缩放因子

output = lum + factor * (img - lum)
         ↑      ↑               ↑
         亮度  缩放            偏离量
```

#### Step 3.3: 颜色混合器 (最复杂)

```
# 转为 HSL 空间
H, S, L = rgb_to_hsl(img)  # H∈[0,360°], S,L∈[0,1]

# 初始化调整
H_adj, S_adj, L_adj = H.clone(), S.clone(), L.clone()

# 饱和度保护（无意义的灰色色相不调）
sat_guard = (S / 0.10).clamp(0, 1)
#  S≤0.10 → guard 线性从 0→1
#  S>0.10  → guard = 1 (完全调)

# 对 8 个色相通道循环
for i in range(8):
    center = CENTERS[i]  # e.g., Red=0°, Green=120°
    sigma  = WIDTHS[i] / 2  # e.g., Red σ=20°
    
    # 色相距离 (环形)
    dist = |H - center|
    dist = min(dist, 360 - dist)  # 取环形最短路
    
    # Gaussian soft mask
    hue_mask = exp(-0.5 * (dist / sigma)^2)
    eff_mask = hue_mask * sat_guard
    
    # 色相调整：±100 → ±30°
    H_adj = (H_adj + eff_mask * mixer_h[i] * 0.30) % 360
    
    # 饱和度调整 (比例推进)
    s_shift = eff_mask * (mixer_s[i] / 100)
    S_adj = if s_shift ≥ 0:
               S_adj + s_shift * (1 - S_adj)  # 推向 1
            else:
               S_adj + s_shift * S_adj        # 推向 0
    
    # 亮度调整 (同理)
    l_shift = eff_mask * (mixer_l[i] / 200)  # 注意系数 200
    L_adj = if l_shift ≥ 0:
               L_adj + l_shift * (1 - L_adj)
            else:
               L_adj + l_shift * L_adj

# 转回 RGB
output = hsl_to_rgb(H_adj, S_adj, L_adj)
```

### Stage 4: 效果（1个参数）

```python
# 晕影：四角压暗
yy = linspace(-1, 1, H)  # 纵向坐标
xx = linspace(-1, 1, W)  # 横向坐标
dist = sqrt(xx^2 + yy^2)  # 距离场 ∈ [0, √2]

mask = 1 + vignette / 100 * dist * 0.6
       ↑            ↑               ↑
       初值        参数强度       衰减系数

vignette < 0 → mask < 1 → 压暗
vignette > 0 → mask > 1 → 提亮

output = img * mask.clamp(min=0)
```

### Stage 5: 细节增强（5个参数）

**都用反锐化蒙版 (Unsharp Mask) 原理**：

```python
delta = img - gaussian_blur(img, σ)

# 三种频率
纹理(σ=2):   output = img + (texture/100) * 0.5 * delta
清晰度(σ=10): output = img + (clarity/100) * 0.6 * delta
锐化(σ=0.5):  output = img + (sharp/150) * 0.8 * delta

# 去朦胧 (特殊，不是反锐化)
mean = img.mean()
output = mean + (img - mean) * (1 + dehaze/150)

# 降噪 (Alpha blend)
blurred = gaussian_blur(img, σ=2)
alpha = noise_red / 100
output = img * (1-alpha) + blurred * alpha
```

## 可微性保证

### 关键设计选择

```python
# ✓ 使用 torch.where 而非 if-else
mask = (x > 0)
y = torch.where(mask, f(x), g(x))
# PyTorch 计算两个分支，反向传播时自动处理 mask 的选择

# ✓ 所有操作都是内置可微
F.conv2d  → 内置 autograd
torch.where → 支持梯度（mask 作为选择器）
torch.exp, torch.clamp → 都可微

# ✓ RGB ↔ HSL 转换可微
所有计算都用 torch 操作，支持自动求导
```

### 梯度流路径

```
loss
  ↓ (从某个像素的误差开始)
rendered_img
  ↓ (通过 clamp)
stage5_output (sharpening / denoise / ...)
  ↓
stage4_output (vignette)
  ↓
stage3_output (color_mixer, saturation, vibrance)
  ↓
stage2_output (temperature, tint)
  ↓
stage1_output (exposure, contrast, highlights, ...)
  ↓
input_img (无梯度，用户输入)

每一层都计算 ∂layer_out / ∂layer_param
最后通过链式法则得到 ∂loss / ∂param
```

## 参数优化

### 优化目标函数

```
L(θ) = L1(rendered, target) + λ * SSIM_loss(rendered, target)

其中：
  L1 = mean(|rendered - target|)           # 像素级绝对误差
  SSIM_loss = 1 - SSIM(rendered, target)   # 感知相似度

推荐 λ = 0.5
```

### 优化算法

```
θ_{t+1} = θ_t - lr * ∇L(θ_t)

使用：
  优化器: Adam (自适应学习率)
  学习率: 0.05 (在 [-100, 100] 参数范围内合理)
  调度: Cosine annealing (逐步降低 lr)
  迭代: 600 (两个尺度 × 300 iters)
```

### 多尺度策略

```
Scale 1: 256px
  ├─ 快速收敛到粗粒度特征
  ├─ 200-300 iterations
  └─ 主要调整：亮度、对比度、饱和度

↓ (切换到更大尺度时，保持参数)

Scale 2: 512px
  ├─ 精细化细节特征
  ├─ 200-300 iterations
  └─ 精调：细节、清晰度、锐化
```

## Cold vs Warm Start 的科学解释

### Cold Start (从 θ=0 开始)

```
初值: 所有参数 = 0 (完全恒等)
     loss = 原始 L1(original, target)

优化方向: 完全自由，梯度指向全局最优

最终: 找到在 sRGB 域上的最优参数组合
      loss ≈ 0.034
```

### Warm Start (从 LR 参数初始化)

```
初值: 参数 = LR 在 RAW 线性域的最优值
     loss ≈ 0.170 (已接近，但不是 sRGB 最优)

优化方向: 受初值约束，在初值附近探索

最终: 被 LR 参数设定的"局部最优陷阱"困住
      loss ≈ 0.037 (比 cold start 更差)
```

### 数学解释

```
设 L(θ) 为损失函数，θ* 为全局最优

情况 1: Cold Start
  ∇L(0) 指向 θ*
  → 梯度下降直线到达

情况 2: Warm Start
  θ_LR ≠ θ*（域转换导致）
  ∇L(θ_LR) 可能指向局部最优 θ_local
  → 被困在 θ_local，无法到达 θ*

原因: RAW 线性域和 sRGB 非线性域的优化目标完全不同
      参数的"最优性"依赖于定义域
```

## 总结

| 概念 | 解释 |
|------|------|
| **Differentiable** | 所有操作可微，支持反向自动求导 |
| **sRGB-Native** | 不做 gamma 往返，直接在 sRGB 域操作 |
| **Pipeline** | 5 个阶段严格按 Lightroom 顺序 |
| **40 Parameters** | 16 个标量 + 8×3 混合器 |
| **Frequency Decompose** | 纹理/清晰度/锐化 用不同 σ 的反锐化 |
| **Gradient Flow** | 无分支断裂，所有路径均可导 |
| **Cold Start > Warm Start** | sRGB 域最优 ≠ RAW 域最优 |

---

**关键思想**：用梯度下降在 PNG 的 sRGB 域找到最优参数，而不是直接迁移 RAW 域的参数。
