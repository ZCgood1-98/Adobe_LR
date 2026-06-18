# DifferentiableRenderer 快速参考

## 快速概览

```
原图 PNG  ──→  [Differentiable Renderer (40个参数)]  ──→  输出 PNG
                        ↓ 梯度反向传播 ↓
              参数优化：θ = θ - lr · ∇L
              目标：最小化 L(output, target_image)
```

## 40个参数速查表

### Stage 1: 曝光调整 (Tone Curve & Luminance)

| 参数 | 范围 | 作用 | 系数 | 类型 |
|------|------|------|------|------|
| 曝光度 | -5.0 ~ +5.0 EV | 整体亮度 (指数) | 2^(ev/2.2) | float |
| 白色色阶 | -100 ~ +100 | 高光点定义 | ±0.20 加法 | int |
| 黑色色阶 | -100 ~ +100 | 黑点定义 | ±0.20 加法 | int |
| 高光 | -100 ~ +100 | 明亮区域调整 | ±0.30 加法 | int |
| 阴影 | -100 ~ +100 | 暗部区域调整 | ±0.30 加法 | int |
| 对比度 | -100 ~ +100 | S 形曲线调整 | Photoshop 公式 | int |

### Stage 2: 白平衡 (Color Temperature & Tint)

| 参数 | 范围 | 作用 | 计算 | 类型 |
|------|------|------|------|------|
| 色温 | ±100 (相对) | 冷暖色调 | R *= (1 + rel/500), B *= (1 - rel/500) | 相对 |
| 色调 | -150 ~ +150 | 绿-品红轴 | G *= (1 - tint/800), R/B *= (1 + tint*0.3/800) | int |

**色温标准**：
```
2000K  3000K  5500K(中性)  7000K  10000K
  ←────────────────────────────────────→
  冷(蓝)  暖     日光      多云    冷(蓝)
```

### Stage 3: 色彩调整 (Color Saturation & Mixer)

| 参数 | 范围 | 作用 | 原理 | 类型 |
|------|------|------|------|------|
| 自然饱和度 | -100 ~ +100 | 自适应彩度 | boost = (1 - sat) 的权重 | int |
| 饱和度 | -100 ~ +100 | 全局彩度 | factor = 1 + sat/100 | int |
| color_mixer (8×3) | ±100 each | 按色相调 H/S/L | Gaussian soft mask | int |

**color_mixer 的 8 个色相通道**：
```
Red (0°, ±20° width)
Orange (30°, ±15° width)
Yellow (60°, ±17.5° width)
Green (120°, ±25° width)    ← 最宽（绿色范围大）
Aqua (180°, ±17.5° width)
Blue (225°, ±20° width)
Purple (280°, ±17.5° width)
Magenta (320°, ±17.5° width)
```

**H/S/L 调整范围**：
- H: ±100 → ±30°
- S: ±100 → ±100% (加法，向 0 或 1 推进)
- L: ±100 → ±0.5 (加法，向 0 或 1 推进)

### Stage 4: 效果 (Effects)

| 参数 | 范围 | 作用 | 原理 |
|------|------|------|------|
| 晕影 | -100 ~ 0 | 四角压暗 | mask = 1 - (|v|/100) * dist * 0.6 |

### Stage 5: 细节增强 (Detail Enhancement)

| 参数 | 范围 | 高斯σ | 系数 | 用途 |
|------|------|-------|------|------|
| 纹理 | -100 ~ +100 | 2.0 | 0.5 | 表面微观细节 |
| 清晰度 | -100 ~ +100 | 10.0 | 0.6 | 中调局部对比 |
| 去朦胧 | -100 ~ +100 | - | 1 + x/150 | 对比度增强 |
| 锐化 | 0 ~ +150 | 0.5 | 0.8 | 边缘清晰度 |
| 噪点消除 | 0 ~ +100 | 2.0 | α-blend | 平滑降噪 |

**频率分解**：
```
原图 = 低频 (blur) + 高频 (delta)

纹理:   delta × 0.5  (细微细节)
清晰度: delta × 0.6  (中等细节)
锐化:   delta × 0.8  (边缘)

这些操作在不同频率，可以组合
```

## 核心算法速查

### 亮度分区 Mask 公式

```python
# 对于亮度范围 [0, 1]
lum = 0.2126*R + 0.7152*G + 0.0722*B  # Rec.709

# 高光 (lum > 0.5)
hi_mask = ((lum - 0.5) / 0.5).clamp(0, 1)
#  lum=0.5  → mask=0   (不受影响)
#  lum=0.75 → mask=0.5 (半影响)
#  lum=1.0  → mask=1.0 (完全影响)

# 阴影 (lum < 0.5)
sh_mask = ((0.5 - lum) / 0.5).clamp(0, 1)
#  lum=0.5  → mask=0
#  lum=0.25 → mask=0.5
#  lum=0.0  → mask=1.0

# 应用
img_new = img + hi_mask * (highlights / 100.0) * 0.30
```

### 颜色混合器的 Soft Mask

```python
# 对于 Red 通道 (center=0°, sigma=20°)
dist = min(|H - 0|, 360 - |H - 0|)  # 圆形距离
hue_mask = exp(-0.5 * (dist / 20)^2)

# 饱和度保护（防止灰色伪色）
sat_guard = (S / 0.10).clamp(0, 1)
#  S=0.00 → guard=0    (完全灰，不调）
#  S=0.05 → guard=0.5  (有点灰，半调)
#  S≥0.10 → guard=1.0  (彩色，全调)

effective_mask = hue_mask * sat_guard
H_new = H + effective_mask * (mixer_h / 100) * 0.30
```

### Unsharp Mask 族

```python
# 反锐化蒙版原理
delta = original - gaussian_blur(original, sigma)
sharpened = original + strength * delta

# 三种应用（频率不同）
纹理(σ=2.0):   output = img + (texture/100) * 0.5 * delta
清晰度(σ=10):  output = img + (clarity/100) * 0.6 * delta
锐化(σ=0.5):   output = img + (sharp/150) * 0.8 * delta
```

## 参数初始化

### Cold Start (推荐)
```python
renderer = DifferentiableRenderer().to('cuda')
# 所有参数默认初始化为 0（恒等操作）
```

### Warm Start (从 Lightroom 参数)
```python
renderer = DifferentiableRenderer().to('cuda')
renderer.init_from_lr_params(lr_params_dict)
# 必须从 step1_parse_params.extract_params() 获取
```

**性能对比**：
| 方式 | 初始loss | 最终loss | 训练时间 | 推荐 |
|------|---------|---------|---------|------|
| Cold | 0.2616 | **0.0342** | 正常 | ✓ |
| Warm | 0.1697 | 0.0374 | 正常 | 快速演示用 |

⚠️ **注意**：Cold start 反而更优，因为 LR 参数来自 RAW 线性域，在 sRGB 非线性域下是局部陷阱。

## 前向推理

```python
# 推理
img = torch.rand(1, 3, H, W).to('cuda')  # 原图 PNG
with torch.no_grad():
    output = renderer(img)
# output: (1, 3, H, W) ∈ [0, 1]
```

## 反向传播

```python
# 优化循环
optimizer = torch.optim.Adam(renderer.parameters(), lr=0.05)

for epoch in range(300):
    optimizer.zero_grad()
    
    # 前向
    rendered = renderer(img_original)
    
    # 损失（L1 + 0.5 * SSIM_loss）
    loss = L1_loss(rendered, img_target) + 0.5 * ssim_loss(...)
    
    # 反向
    loss.backward()  # 自动计算 ∂loss/∂param for all 40 params
    
    # 更新
    optimizer.step()
```

## 梯度流诊断

```python
# 检查所有参数都收到梯度
for name, param in renderer.named_parameters():
    if param.grad is None:
        print(f"❌ {name} 无梯度！")
    else:
        print(f"✓ {name}: grad_norm={param.grad.norm():.2e}")
```

**常见问题**：
- `clarity/texture/sharpen` 梯度很小？
  → 正常。在随机图像上高频成分接近 0
  → 用真实图像时会显著增加

## sRGB 域的设计权衡

### ✓ 优点
- 输入已在 [0, 1]，无需 gamma decode
- 所有 mask 天然限于 [0, 1]，无溢出
- 乘法因子永不为负
- 无精度损失、无伪色

### ✗ 劣势
- LR 参数（RAW 线性域）数值无法直接迁移
- 感知上的 2× 亮度 ≠ 物理上的 2× 辐照度
- 高动态范围内精度略低

### 决策逻辑
```
PNG 已是 sRGB 编码输出
      ↓
从 sRGB 解码→线性 (需要 gamma)
      ↓
应用参数
      ↓
再编码回 sRGB (又一次 gamma)
      ↓
两次 gamma 往返 = 精度损失 + 伪色风险

❌ 不值得！

直接在 sRGB 域工作：
  ✓ 少 2 次 gamma 往返
  ✓ 感知域上的调整更直观
  ✓ 无需额外计算
```

## 文件结构

```
differentiable_renderer/
├── renderer.py           # DifferentiableRenderer 核心类
├── fit.py                # 参数优化脚本
├── __init__.py
├── test_output/          # 拟合结果示例
│   └── {sample_id}/
│       ├── params_v0.json       # 最终参数
│       ├── rendered_v0.png      # 渲染结果
│       ├── compare_v0.png       # [Orig|Rendered|Target]
│       ├── history_v0.json      # 损失曲线
│       └── renderer_v0.pt       # 模型检查点
└── loss_comparison.png   # 冷/热启动对比
```

## 使用示例

### 单样本拟合
```bash
python differentiable_renderer/fit.py \
    --data_dir scraped_data_v4/train \
    --sample_id 000cc99f-66fa-498f-8e02-c3735a5b90a2 \
    --version 0 \
    --output_dir results/ \
    --scales 256 512 \
    --iters 300
```

### 批量拟合（恢复安全）
```bash
python differentiable_renderer/fit.py \
    --data_dir scraped_data_v4/train \
    --output_dir results/ \
    --scales 256 512 \
    --iters 300
# 已拟合的样本自动跳过
```

### 热启动 (Warm Start)
```bash
python differentiable_renderer/fit.py \
    --data_dir scraped_data_v4/train \
    --sample_id 000cc99f-... \
    --warm_start  # 从 LR 参数初始化
```

## 验证可微性

```python
# 快速检查梯度流
img = torch.rand(1, 3, 256, 256, requires_grad=False)
renderer = DifferentiableRenderer()

# 前向
out = renderer(img)

# 损失与反向
loss = out.mean()
loss.backward()

# 验证
total_params = sum(p.numel() for p in renderer.parameters())
grads = sum(1 for p in renderer.parameters() if p.grad is not None)
print(f"✓ {grads}/{total_params} 参数有梯度")
```

---

**最后提示**：
- 所有操作都是 fully differentiable
- 梯度自动计算，无需手动实现
- PyTorch 处理数值稳定性和反向模式 AD
- 放心优化！
