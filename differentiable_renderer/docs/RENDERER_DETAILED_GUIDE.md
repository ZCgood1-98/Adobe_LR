# Differentiable Renderer 详细讲解

## 一、核心设计理念

### 1.1 为什么要做 Differentiable Renderer？

**问题背景**：
- Lightroom 在 **RAW 线性域** 进行编辑（科学值）
- 但数据集中只有 **PNG sRGB 域** 的图片（已gamma编码）
- 直接应用 LR 参数到 PNG 效果不同

**解决方案**：
```
原图 (PNG sRGB)  →  [可微渲染器 (40个参数)]  →  编辑后图
                      ↓
                    梯度下降找最优参数组合
                      ↓
               θ_png = argmin L(render(原图, θ), 目标图)
```

### 1.2 为什么运作在 sRGB 域？

**不采用线性域的原因**：
```
线性域方案的问题：
  高亮 exposure +3 EV
    → 像素 * 2^(3/2.2) ≈ 5× 倍增
    → 1.0 → 5.0 （溢出！）
    → 高光 mask 变为 0（全黑）
    → 严重伪色

sRGB 原生方案优点：
  ✓ 输入已在 [0,1]，无需 gamma decode
  ✓ 所有 mask 的亮度输入天然限于 [0,1]
  ✓ 乘法因子永不为负
  ✓ 无精度损失、无伪色
```

## 二、40个参数详解

### 2.1 参数分类与流水线顺序

```
Render Pipeline (5个阶段，顺序严格按 Lightroom 工作流)
│
├─ Stage 1: 基础曝光 (Tone)
│  ├─ 曝光度       exposure      [float: -5.0 ~ +5.0 EV]
│  ├─ 白色色阶     whites        [int:   -100 ~ +100]
│  ├─ 黑色色阶     blacks        [int:   -100 ~ +100]
│  ├─ 高光        highlights     [int:   -100 ~ +100]
│  ├─ 阴影        shadows        [int:   -100 ~ +100]
│  └─ 对比度       contrast       [int:   -100 ~ +100]
│
├─ Stage 2: 白平衡 (White Balance)
│  ├─ 色温        temp           [相对值: (K-5500)/50]
│  └─ 色调        tint           [int:   -150 ~ +150]
│
├─ Stage 3: 色彩调整 (Color)
│  ├─ 自然饱和度   vibrance       [int:   -100 ~ +100]
│  ├─ 饱和度       saturation     [int:   -100 ~ +100]
│  └─ color_mixer  [8 channels × (H, S, L)]
│     ├─ Red       mixer_hue[0], mixer_sat[0], mixer_lum[0]
│     ├─ Orange    mixer_hue[1], mixer_sat[1], mixer_lum[1]
│     ├─ Yellow    mixer_hue[2], mixer_sat[2], mixer_lum[2]
│     ├─ Green     mixer_hue[3], mixer_sat[3], mixer_lum[3]
│     ├─ Aqua      mixer_hue[4], mixer_sat[4], mixer_lum[4]
│     ├─ Blue      mixer_hue[5], mixer_sat[5], mixer_lum[5]
│     ├─ Purple    mixer_hue[6], mixer_sat[6], mixer_lum[6]
│     └─ Magenta   mixer_hue[7], mixer_sat[7], mixer_lum[7]
│
├─ Stage 4: 效果 (Effects)
│  └─ 晕影        vignette       [int:   -100 ~ 0]
│
└─ Stage 5: 细节 (Details)
   ├─ 纹理        texture        [int:   -100 ~ +100]
   ├─ 清晰度      clarity        [int:   -100 ~ +100]
   ├─ 去朦胧      dehaze         [int:   -100 ~ +100]
   ├─ 锐化        sharpening     [int:    0 ~ +150]
   └─ 噪点消除    noise_red      [int:    0 ~ +100]
```

## 三、核心处理算法详解

### 3.1 亮度调整 (Stage 1)

#### 曝光度 (Exposure)
```python
def _exposure(self, x):
    return (x * (2.0 ** (self.exposure / 2.2))).clamp(0.0, 1.0)
```

**原理**：
- EV (Exposure Value) 是摄影标准
- EV = 1 → 亮度翻倍
- sRGB 感知域里，EV = 1 → 2^(1/2.2) ≈ 1.37× 感知亮度
- 系数 1/2.2 是 sRGB gamma 指数

**例子**：
```
曝光度 = +1.0 EV
    → img * 2^(1.0/2.2) ≈ img * 1.37
    → 亮度适度提升，不过曝
```

#### 高光 & 阴影 (Highlights & Shadows) - 亮度分区
```python
def _highlights_shadows(self, x):
    lum = _luminance(x)  # Rec.709: 0.2126R + 0.7152G + 0.0722B
    
    # 高光处理：lum > 0.5 的区域
    hi_mask = ((lum - 0.5) / 0.5).clamp(0.0, 1.0)  # [0, 1] 渐变
    x = (x + hi_mask * (highlights / 100.0) * 0.30).clamp(0.0, 1.0)
    
    # 阴影处理：lum < 0.5 的区域  
    sh_mask = ((0.5 - lum) / 0.5).clamp(0.0, 1.0)
    x = (x + sh_mask * (shadows / 100.0) * 0.30).clamp(0.0, 1.0)
    
    return x
```

**设计原理**：
- 使用 **亮度 mask**（不是乘法，而是加法）
- 避免 sRGB 压缩域的色相偏移
- ±100 参数 → ±0.30 亮度加减（相对 [0,1] 范围）

**效果示意**：
```
lum = 0.2 (暗): sh_mask = (0.5-0.2)/0.5 = 0.6  → 受阴影影响强
lum = 0.5 (中): sh_mask = 0, hi_mask = 0       → 不受影响
lum = 0.8 (亮): hi_mask = (0.8-0.5)/0.5 = 0.6  → 受高光影响强
```

#### 白色/黑色色阶 (Whites & Blacks)
```python
def _whites(self, x):
    """只影响最亮区域 (lum > 0.75)"""
    lum = _luminance(x)
    mask = ((lum - 0.75) / 0.25).clamp(0.0, 1.0)
    return (x + mask * (whites / 100.0) * 0.20).clamp(0.0, 1.0)

def _blacks(self, x):
    """只影响最暗区域 (lum < 0.25)"""
    lum = _luminance(x)
    mask = ((0.25 - lum) / 0.25).clamp(0.0, 1.0)
    return (x + mask * (blacks / 100.0) * 0.20).clamp(0.0, 1.0)
```

**作用**：
- 精细控制纯黑和纯白的定义
- whites↑ 让高光更纯净
- blacks↑ 让阴影保持黑色

#### 对比度 (Contrast)
```python
def _contrast(self, x):
    c = self.contrast * 1.27  # [-100,100] → [-127,127]
    # Photoshop/Lightroom 经典公式
    factor = (259.0 * (c + 255.0)) / (255.0 * (259.0 - c) + 1e-8)
    return (factor * (x - 0.5) + 0.5).clamp(0.0, 1.0)
```

**原理**：
- 围绕 sRGB 中灰 (0.5) 对称
- contrast > 0: 亮变亮、暗变暗（S形曲线）
- contrast < 0: 压平对比（反S形）

### 3.2 色彩调整 (Stage 2-3)

#### 色温 (Temperature)
```python
def _temperature(self, x):
    rel = self.temp  # 无量纲相对值
    # 红通道：高温(>5500K) 加强 → 偏暖
    # 蓝通道：高温 减弱 → 相对偏暖
    r = (x[:, 0:1] * (1.0 + rel / 500.0)).clamp(0.0, 1.0)
    g = x[:, 1:2]
    b = (x[:, 2:3] * (1.0 - rel / 500.0)).clamp(0.0, 1.0)
    return torch.cat([r, g, b], dim=1)
```

**色温标准**：
```
2000K   3000K   5500K (中性)   7000K   10000K
←─────────────────────────────────────────→
  冷(蓝)  暖光    日光      多云    阴天(蓝)
```

**参数映射**：
```
self.temp = (实际K - 5500) / 50
例：
  temp = -50 (5500-50*50=2000K)  → 很冷
  temp = 0   (5500K)            → 中性
  temp = +50 (5500+50*50=8000K)  → 很暖
```

#### 色调 (Tint)
```python
def _tint(self, x):
    shift = self.tint / 800.0  # [-150,150] → [-0.19, 0.19]
    r = (x[:, 0:1] * (1.0 + shift * 0.3)).clamp(0.0, 1.0)
    g = (x[:, 1:2] * (1.0 - shift)).clamp(0.0, 1.0)  # 主要轴
    b = (x[:, 2:3] * (1.0 + shift * 0.3)).clamp(0.0, 1.0)
    return torch.cat([r, g, b], dim=1)
```

**色调轴**：
```
tint < 0 (绿):    减少品红，增加绿  → 相机AF纠正
tint > 0 (品红):  减少绿，增加品红  → 暖肤色
```

#### 自然饱和度 & 全局饱和度
```python
def _vibrance(self, x):
    """自然饱和度：优先增强低饱和区域"""
    max_c = torch.max(x, dim=1, keepdim=True).values
    min_c = torch.min(x, dim=1, keepdim=True).values
    sat = (max_c - min_c) / max_c.clamp(min=1e-6)  # [0,1]
    
    # boost 与已有饱和度反相：低饱和→大boost，高饱和→小boost
    boost = (self.vibrance / 100.0) * (1.0 - sat)
    lum = _luminance(x)
    return (x + boost * (x - lum)).clamp(0.0, 1.0)

def _saturation(self, x):
    """全局饱和度：均匀拉伸所有颜色"""
    lum = _luminance(x)
    factor = 1.0 + self.saturation / 100.0
    return (lum + factor * (x - lum)).clamp(0.0, 1.0)
```

**差异**：
```
vibrance = +50:
  灰色像素    → 彩度增加很多  (boost = 0.5)
  饱和颜色    → 彩度增加很少  (boost ≈ 0)
  ⇒ 保护已有色彩，提升平淡区域

saturation = +50:
  所有像素 → 彩度均匀增加 50%
  ⇒ 全局放大效果，可能过饱和
```

### 3.3 颜色混合器 (Color Mixer) - 最复杂的部分

#### RGB ↔ HSL 转换
```python
def _rgb_to_hsl(img):
    """将 sRGB 转为 HSL (H: [0,360]°, S/L: [0,1])"""
    max_c, min_c = torch.max(img, dim=1), torch.min(img, dim=1)
    delta = max_c - min_c
    
    # L (Lightness): 平均最大最小值
    L = (max_c + min_c) / 2.0
    
    # S (Saturation): 相对于亮度
    S = delta / (max_c + min_c)      if L < 0.5
    S = delta / (2.0 - max_c - min_c) if L ≥ 0.5
    
    # H (Hue): 根据哪个通道最大分段计算
    if max == R:   H = 60 * ((G - B) / delta) % 6
    if max == G:   H = 60 * ((B - R) / delta + 2)
    if max == B:   H = 60 * ((R - G) / delta + 4)
    
    return H, S, L
```

**逆转换** RGB ← HSL:
```python
def _hsl_to_rgb(H, S, L):
    """反向转换"""
    # 色度 Chroma
    C = (1 - |2L - 1|) * S
    
    # 基于 H 段 (0-5) 分配 RGB
    seg = floor(H / 60) % 6
    X = C * (1 - |((H / 60) % 2) - 1|)
    
    if seg == 0:   (R', G', B') = (C, X, 0)
    if seg == 1:   (R', G', B') = (X, C, 0)
    if seg == 2:   (R', G', B') = (0, C, X)
    ...
    
    # 亮度匹配
    m = L - C/2
    return (R', G', B') + m
```

#### 颜色混合器核心逻辑
```python
def _color_mixer(self, x):
    """对 8 个色相通道分别调整 H/S/L"""
    
    H, S, L = _rgb_to_hsl(x)  # 转为 HSL 域
    H_adj, S_adj, L_adj = H.copy(), S.copy(), L.copy()
    
    # 饱和度保护：低饱和像素（灰色）H值无意义，不应被调整
    sat_guard = (S / 0.10).clamp(0.0, 1.0)
    #  S = 0.00 → guard = 0    (完全灰，不调整)
    #  S = 0.05 → guard = 0.5  (有点灰，半影响)
    #  S ≥ 0.10 → guard = 1.0  (彩色，完全影响)
    
    for i in range(8):  # 8 个色相通道
        center = MIXER_CENTERS[i]      # e.g., Red=0°, Green=120°
        sigma = MIXER_WIDTHS[i] / 2.0  # e.g., Red=20° (±20° 范围)
        
        # 圆形色相距离 → 高斯软遮罩
        dist = abs(H - center)
        dist = min(dist, 360 - dist)   # 环形距离
        hue_mask = exp(-0.5 * (dist / sigma)^2)
        # e.g., dist=0° → mask=1.0 (完全影响中心)
        #      dist=20° → mask≈0.6  (边界衰减)
        #      dist=40° → mask≈0.05 (基本无影响)
        
        eff_mask = hue_mask * sat_guard  # 综合遮罩
        
        # 色相调整：±100 → ±30°
        H_adj = (H_adj + eff_mask * mixer_hue[i] * 0.30) % 360.0
        
        # 饱和度调整：比例推进（正向推向 1.0，负向缩向 0.0）
        s_shift = eff_mask * (mixer_sat[i] / 100.0)
        S_adj = S_adj + s_shift * (1.0 - S_adj)  if s_shift >= 0
        S_adj = S_adj + s_shift * S_adj           if s_shift <  0
        
        # 同理亮度调整
        l_shift = eff_mask * (mixer_lum[i] / 200.0)
        L_adj = L_adj + l_shift * (1.0 - L_adj)  if l_shift >= 0
        L_adj = L_adj + l_shift * L_adj           if l_shift <  0
    
    return _hsl_to_rgb(H_adj, S_adj, L_adj)
```

**关键设计**：
```
1. 色相软遮罩 (Gaussian)
   ✓ Red 通道不只影响 0°，±40° 范围都有影响
   ✓ 平滑边界，避免色块分割

2. 饱和度保护
   ✓ 灰色 (S≈0) 的像素没有有效色相
   ✓ 强制 guard=0 防止伪色

3. 比例调整（非线性）
   ✓ S=0.2, S_shift=+0.5 → S_new = 0.2 + 0.5*(1-0.2) = 0.6
   ✓ S=0.9, S_shift=+0.5 → S_new = 0.9 + 0.5*(1-0.9) = 0.95
   ✓ 高饱和像素不会被轻易推到 1.0 (撞顶)
```

### 3.4 细节增强 (Stage 5)

#### Unsharp Mask 族 (纹理/清晰度/锐化)
```python
def _texture(self, x):
    """小半径 (σ=2) 反锐化蒙版 → 增强表面细节"""
    blurred = _gaussian_blur(x, kernel_sigma=2.0)
    delta = x - blurred  # 高频分量
    return (x + (texture / 100.0) * 0.5 * delta).clamp(0.0, 1.0)

def _clarity(self, x):
    """中等半径 (σ=10) 反锐化蒙版 → 增强中间调局部对比"""
    blurred = _gaussian_blur(x, kernel_sigma=10.0)
    delta = x - blurred
    return (x + (clarity / 100.0) * 0.6 * delta).clamp(0.0, 1.0)

def _sharpen(self, x):
    """极小半径 (σ=0.5) 反锐化蒙版 → 增强边缘清晰度"""
    blurred = _gaussian_blur(x, kernel_sigma=0.5)
    delta = x - blurred
    return (x + (sharpening / 150.0) * 0.8 * delta).clamp(0.0, 1.0)
```

**频率分解**：
```
原图  =  低频(blurred) + 高频(delta)

纹理 (+):  低频不变，高频×0.5 → 表面细节增强
清晰度(+): 低频不变，高频×0.6 → 中调定义增强
锐化  (+): 低频不变，高频×0.8 → 边缘强化

组合时的叠加：
  纹理 + 清晰度 > 单独都有效
  （因为作用在不同频率）
```

#### 去朦胧 (Dehaze)
```python
def _dehaze_op(self, x):
    """围绕全图均值增强局部对比"""
    mean = x.mean()
    return (mean + (x - mean) * (1.0 + dehaze / 150.0)).clamp(0.0, 1.0)
```

**原理**：
```
朦胧原因：亮区被拉暗，暗区被提亮（失去对比）
去朦胧：
  x_new = mean + (x - mean) × (1 + dehaze/150)
  
例子：
  mean = 0.5, x = 0.3 (暗), dehaze=+50
  x_new = 0.5 + (0.3-0.5) × (1 + 50/150)
        = 0.5 + (-0.2) × 1.33
        = 0.5 - 0.267
        = 0.233  (更暗，对比增强)
```

#### 噪点消除 (Noise Reduction)
```python
def _noise_reduction(self, x):
    """α-blend 与高斯模糊"""
    blurred = _gaussian_blur(x, kernel_sigma=2.0)
    alpha = (self.noise_red / 100.0).clamp(0.0, 1.0)
    return (x * (1 - alpha) + blurred * alpha).clamp(0.0, 1.0)
```

**从 noise_red=0 到 100 的渐变**：
```
noise_red=0:   原图，无平滑
noise_red=50:  50% 原图 + 50% 模糊
noise_red=100: 100% 模糊，细节完全消失
```

#### 晕影 (Vignette)
```python
def _vignette_op(self, x):
    """四角压暗"""
    B, C, H, W = x.shape
    # 生成距离场：中心 0，四角 √2
    yy = linspace(-1, 1, H)
    xx = linspace(-1, 1, W)
    dist = sqrt(xx² + yy²)
    
    # vignette<0 压暗四角，vignette>0 提亮四角
    mask = 1.0 + vignette / 100.0 * dist * 0.6
    return (x * mask.clamp(min=0.0)).clamp(0.0, 1.0)
```

**距离场示意**：
```
     0.0   0.5   1.0   0.5   0.0
   ┌───────────────────────────┐
0.0│                           │
0.5│     0.7   0.9   0.7       │
1.0│     0.9   1.4   0.9       │
0.5│     0.7   0.9   0.7       │
0.0│                           │
   └───────────────────────────┘

vignette = -50:
  mask = 1 + (-0.5) * dist * 0.6 = 1 - 0.3*dist
  → 中心 mask=1.0 (不变)
  → 四角 mask≈0.4 (压暗 60%)
```

## 四、可微性与梯度流

### 4.1 为什么所有操作都可微？

**设计原则**：
- ❌ 不使用 `if param == 0: return` guards（会断裂梯度）
- ✓ 所有操作始终执行（即使参数为0，梯度也流动）

**例子**：
```python
# 不好的做法
if self.exposure == 0:
    return x  # 断裂梯度！

# 好的做法
return x * (2.0 ** (self.exposure / 2.2))
# 当 exposure=0 时，2^0 = 1, x×1 = x（恒等）
# 但梯度链条完整
```

### 4.2 Gaussian Blur 的可微实现

```python
def _gaussian_blur(img, kernel_1d):
    """分离式 2D 高斯卷积 → 高效且可微"""
    
    # kernel_1d: 1D 高斯核（预计算）
    # e.g., σ=2: [0.02, 0.05, 0.09, 0.14, 0.19, 0.14, 0.09, 0.05, 0.02]
    
    # 扩展为 (C, 1, 1, ks)，组卷积保持通道分离
    kh = kernel_1d.view(1, 1, 1, ks).expand(C, -1, -1, -1)
    kv = kernel_1d.view(1, 1, ks, 1).expand(C, -1, -1, -1)
    
    # 反射填充 + 水平卷积
    out = F.pad(img, [ks//2, ks//2, 0, 0], mode='reflect')
    out = F.conv2d(out, kh, groups=C)  # groups=C → 每通道独立卷积
    
    # 反射填充 + 竖直卷积
    out = F.pad(out, [0, 0, ks//2, ks//2], mode='reflect')
    out = F.conv2d(out, kv, groups=C)
    
    return out
```

**为什么 PyTorch 能自动求导**：
- `F.conv2d` 是内置可微操作
- 反向传播时自动计算卷积的梯度
- 所有参数（如 clarity）到模糊结果的路径畅通

### 4.3 RGB ↔ HSL 的可微性陷阱

**潜在问题**：
```python
# ❌ 这里有分支
mask_r = (delta > eps) & (max_c == r)
if mask_r:
    H = h_r
else:
    H = h_g  # 或其他

# 问题：梯度只通过一个分支
```

**解决方案**：使用 `torch.where` 而非 if-else
```python
# ✓ 所有分支都计算，再用 where 选择
H = torch.where(mask_r, h_r,
    torch.where(mask_g, h_g,
    torch.where(mask_b, h_b, ...)))

# PyTorch 会对所有计算路径求导，只是在反向传播时
# 将无关分支的梯度置零（通过 mask 的 0 倍数）
```

## 五、前向推理示例

### 5.1 一张图片的完整流程

```
输入: img (1, 3, 256, 256) sRGB [0, 1]

Stage 1 (曝光):
  img → exposure(+1.0) → img' (更亮)
  img' → whites(+20) → img'' (高光更纯)
  ...继续所有 Stage 1 参数...

Stage 2 (白平衡):
  img → temperature(-200K) → img' (偏冷)
  img' → tint(+30) → img'' (偏品红)

Stage 3 (色彩):
  img'' → vibrance(-10) → img''' (低饱和区变淡)
  img''' → saturation(-20) → img'''' (全局降饱和)
  img'''' → color_mixer(Red.H=+10, Red.S=-5, ...) → img'''''

Stage 4 (效果):
  img''''' → vignette(-30) → img'''''' (四角压暗)

Stage 5 (细节):
  img'''''' → texture(+5) → 增强表面细节
  ... → clarity(+10) → 增强中调
  ... → sharpen(+20) → 增强边缘
  ... → noise_red(+30) → 平滑噪声

最终输出: img_rendered (1, 3, 256, 256) sRGB [0, 1]
```

### 5.2 梯度回传

```
Loss = L1(rendered, target) + 0.5 * (1 - SSIM)

loss.backward()

梯度链条：
  ∂loss/∂rendered ← computed
  ∂rendered/∂noise_red (最后一层)
  ∂rendered/∂sharpen
  ...
  ∂rendered/∂mixer_lum[3]  (Green 通道 L)
  ...
  ∂rendered/∂exposure (第一层)

所有 40 个参数都收到梯度更新：
  param_new = param_old - lr * ∂loss/∂param
```

## 六、与 NumPy 版本的对应关系

| NumPy (step3_assr_cot.py) | PyTorch (renderer.py) | 用途 |
|---------------------------|----------------------|------|
| `apply_exposure(img, ev)` | `_exposure(x)` | 曝光 |
| `apply_shadows_highlights()` | `_highlights_shadows()` | 高光/阴影 |
| `_rgb_to_hsl()` | `_rgb_to_hsl()` | RGB→HSL |
| `gaussian_filter(sigma=10)` | `_gaussian_blur(_k_clarity)` | 清晰度 Blur |
| `apply_color_mixer()` | `_color_mixer()` | 8通道颜色调整 |

**核心相同**：
- 算法完全一致（系数、逻辑、顺序）
- NumPy 数组 ↔ PyTorch 张量的映射
- 可微性是唯一的新增约束

## 七、常见疑问

### Q: 为什么梯度有时很小？
**A**：高频操作（texture/clarity/sharpen）在随机图像上高频分量可能接近 0。
→ 解决：用真实图像时梯度会明显

### Q: 为什么 warm start 反而更差？
**A**：LR 参数是 RAW 线性域最优，sRGB 非线性域下是局部最优陷阱。
→ 解决：Cold start 能逃离这个陷阱，找到全局更优点

### Q: 色温参数为什么用相对值？
**A**：绝对值 [2000, 10000] 范围太大，优化器难以学习。
→ 相对化：(K - 5500) / 50 → [-100, 100]，优化器友好

### Q: HSL 转换中为什么需要 sat_guard？
**A**：灰色像素 (S≈0) 没有定义色相，强制调整会产生伪色。
→ guard 权重：S < 0.10 时权重线性从 0 升到 1，避免伪色

