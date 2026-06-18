# Renderer 文档索引

本文档索引提供了对 Differentiable Renderer 的完整讲解。根据你的需求选择合适的文档：

## 📚 文档导航

### 1. **快速开始** ⚡
**文件**: [`QUICK_REFERENCE.md`](QUICK_REFERENCE.md)

**内容**：
- 40个参数速查表
- 核心算法公式
- 前向推理/反向传播示例
- 使用命令速查

**何时阅读**：
- 想快速了解参数含义
- 需要调用 API
- 想看具体的数学公式

---

### 2. **详细教程** 📖
**文件**: [`RENDERER_DETAILED_GUIDE.md`](RENDERER_DETAILED_GUIDE.md)

**内容**：
- 五大设计理念详解
- 40个参数的**逻辑分类**与**工作原理**
- 每个操作的**数学推导**和**直观解释**
- RGB ↔ HSL 转换与色相软遮罩的原理
- 高斯反锐化蒙版族的频率分解
- Cold vs Warm Start 分析

**何时阅读**：
- 想深入理解每个算法的工作原理
- 在调试或改进时需要理论支持
- 对渲染管线的设计感兴趣

**主要章节**：
```
├─ 第一部分：核心设计理念 (sRGB 域，可微性，etc.)
├─ 第二部分：40 个参数详解 (分类、范围、作用)
├─ 第三部分：核心处理算法 (亮度调整、色彩调整、细节增强)
├─ 第四部分：可微性与梯度流
├─ 第五部分：前向推理示例
├─ 第六部分：与 NumPy 版本对应
└─ 第七部分：常见疑问解答
```

---

### 3. **架构讲解** 🏗️
**文件**: [`RENDERER_ARCHITECTURE.md`](RENDERER_ARCHITECTURE.md)

**内容**：
- 为什么要做 Differentiable Renderer (问题陈述)
- 5个核心设计原则的详细论证
- 完整的前向推理流程（逐步、逐参数）
- 可微性保证与梯度流路径
- Cold vs Warm Start 的数学解释
- 参数优化策略

**何时阅读**：
- 想完全理解这个项目为什么这样设计
- 需要从第一性原理理解每个选择
- 想看前向推理的详细步骤
- 对为什么 Cold Start > Warm Start 感兴趣

**主要观点**：
- RAW 线性域 ≠ sRGB 非线性域 → 需要重新优化
- sRGB-Native 设计避免了 gamma 往返带来的伪色和精度损失
- 40个参数都可微，但没有分支控制
- Gaussian Soft Mask 在颜色混合器中平滑边界

---

## 🎯 根据问题选择文档

### "怎么使用 Renderer？"
→ [`QUICK_REFERENCE.md`](QUICK_REFERENCE.md) 的**使用示例**部分

### "这个参数是干什么的？"
→ [`QUICK_REFERENCE.md`](QUICK_REFERENCE.md) 的**参数速查表**
或 [`RENDERER_DETAILED_GUIDE.md`](RENDERER_DETAILED_GUIDE.md) 的对应章节

### "曝光度怎么实现的？"
→ [`RENDERER_DETAILED_GUIDE.md`](RENDERER_DETAILED_GUIDE.md) 的**3.1 曝光度**
或 [`RENDERER_ARCHITECTURE.md`](RENDERER_ARCHITECTURE.md) 的**Stage 1**

### "为什么 sRGB 域而不是线性域？"
→ [`RENDERER_ARCHITECTURE.md`](RENDERER_ARCHITECTURE.md) 的**设计原则 1**
或 [`RENDERER_DETAILED_GUIDE.md`](RENDERER_DETAILED_GUIDE.md) 的**一、核心设计理念**

### "梯度怎么流动的？"
→ [`RENDERER_ARCHITECTURE.md`](RENDERER_ARCHITECTURE.md) 的**可微性保证**
或 [`QUICK_REFERENCE.md`](QUICK_REFERENCE.md) 的**梯度流诊断**

### "为什么 Cold Start 比 Warm Start 更优？"
→ [`RENDERER_ARCHITECTURE.md`](RENDERER_ARCHITECTURE.md) 的**Cold vs Warm Start 的科学解释**
或阅读对比分析文档（见下文）

### "颜色混合器怎么工作的？"
→ [`RENDERER_DETAILED_GUIDE.md`](RENDERER_DETAILED_GUIDE.md) 的**3.3 颜色混合器**
或 [`RENDERER_ARCHITECTURE.md`](RENDERER_ARCHITECTURE.md) 的**Stage 3.3**

---

## 📊 补充分析文档

### Cold vs Warm Start 对比
**文件**: `differentiable_renderer/loss_comparison.png` (可视化)

**内容**：
- 损失曲线对比（总损失/L1/SSIM）
- 数值对比（初值/最终值/改进量）
- 收敛速度分析

**生成命令**：
```bash
# 已自动生成，位置:
cat differentiable_renderer/loss_comparison.png
```

---

### 管线演示图
**文件**: `differentiable_renderer/pipeline_demo.png` (可视化)

**内容**：
- 顶行：Stage-by-Stage 累积效果（从原图到最终）
- 底行：参数扫描对比（同一参数的不同值）

**展示参数**：
- Stage 1-5 的逐步效果
- 曝光度范围：-2, 0, +2
- 对比度范围：-50, 0, +50

---

### 数据流图
**文件**: `differentiable_renderer/dataflow_graph.png` (可视化)

**内容**：
- 5 个处理阶段的流程图
- 每个阶段的参数列表
- Color mixer 的独特处理

---

## 📁 核心代码文件

### 主要实现
- [`differentiable_renderer/renderer.py`](differentiable_renderer/renderer.py) — 核心 40 个参数的实现
- [`differentiable_renderer/fit.py`](differentiable_renderer/fit.py) — 参数优化脚本

### 测试结果
```
differentiable_renderer/test_output/
└── 000cc99f-66fa-498f-8e02-c3735a5b90a2/  (样本 ID)
    ├── params_v0.json          — 最终拟合参数
    ├── rendered_v0.png         — 渲染结果 PNG
    ├── compare_v0.png          — 对比图 [Original|Rendered|Target]
    ├── history_v0.json         — 损失曲线数据
    └── renderer_v0.pt          — 模型检查点
```

---

## 🔍 深度对比阅读建议

### 如果你想**完全理解**这个项目：

**推荐阅读顺序**：
1. [`RENDERER_ARCHITECTURE.md`](RENDERER_ARCHITECTURE.md) — **一遍完整讲解**
   - 从问题陈述开始，看为什么要这样设计
   - 理解 5 个核心设计原则
   - 看完整的前向推理流程
   
2. [`RENDERER_DETAILED_GUIDE.md`](RENDERER_DETAILED_GUIDE.md) — **详细公式与例子**
   - 重点学习你感兴趣的参数（如颜色混合器）
   - 看直观解释与可视化示意图
   
3. [`QUICK_REFERENCE.md`](QUICK_REFERENCE.md) — **快速查阅**
   - 把常用公式复制到自己的笔记本
   - 记住参数范围和系数

**预计阅读时间**：2-3 小时

---

### 如果你想**快速上手**使用：

**推荐阅读顺序**：
1. [`QUICK_REFERENCE.md`](QUICK_REFERENCE.md) 的**使用示例**
2. 跑一遍 `fit.py`
3. 查看 `test_output` 中的结果
4. 遇到问题时查阅**参数速查表**

**预计时间**：30 分钟

---

### 如果你想**修改 Renderer 的算法**：

**推荐阅读顺序**：
1. [`RENDERER_ARCHITECTURE.md`](RENDERER_ARCHITECTURE.md) 的**设计原则**
2. 相关参数的完整实现代码 (`renderer.py`)
3. 对应的测试结果 (`test_output`) 和可视化
4. [`RENDERER_DETAILED_GUIDE.md`](RENDERER_DETAILED_GUIDE.md) 的数学推导

**关键提示**：
- 修改后确保仍然可微（避免条件分支）
- 修改后重新拟合测试样本，检查 loss 变化
- 如果 loss 上升，回到原始设计

---

## 🎓 学习重点概括

### 为什么需要这个 Renderer？
```
问题：LR 参数是基于 RAW 线性域优化，但我们只有 PNG sRGB
答案：用梯度下降在 sRGB 域重新优化参数
```

### 为什么用 sRGB 域？
```
优点：
  ✓ 无 gamma 往返损失
  ✓ 亮度 mask 天然有效
  ✓ 无伪色
  
劣势：
  ✗ LR 参数值无法直接用
  ✗ 需要重新优化
  
权衡：得不偿失，但能规避伪色风险
```

### 为什么 Cold > Warm？
```
Warm Start 看似更聪明（初值更接近），但：
  RAW 域最优 ≠ sRGB 域最优
  → LR 参数是 RAW 域的局部陷阱
  → Cold Start 能逃离这个陷阱，找到更优解
```

### 关键数字
```
参数总数：40
  16 标量 + 24 color_mixer (8×3)

阶段数：5
  Stage 1-5 严格按 Lightroom 顺序

高斯核数：4
  texture/clarity/sharpen/denoise

收敛耗时：~6s per sample (GPU)

最终 Loss：
  Cold start: 0.034
  Warm start: 0.037
  改进: 9.5% (Cold 更优)
```

---

## 📝 更新日志

- **2026-04-16** — 完成 Renderer 实现、拟合脚本、测试
- **2026-04-16** — 生成完整文档库（3个主要文档 + 3个可视化）

---

**本索引最后更新**：2026-04-16

需要帮助？查看对应文档或运行 `python differentiable_renderer/fit.py --help`
