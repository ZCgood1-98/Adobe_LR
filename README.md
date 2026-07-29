# adobe_lr — Lightroom 修图智能体项目

围绕 Lightroom 修图数据，做四件事：**采集数据 → 构建 SFT 训练集 → 评测/蒸馏 LLM → 用可微渲染器反解修图参数**。
代码按这四条线分到四个文件夹，数据与产出目录留在根目录共享。

> ⚠️ **所有命令都从仓库根目录运行**。脚本里的数据路径（如 `scraped_data_v4`）是相对当前工作目录的默认值，`cd` 进子文件夹再运行会找不到数据。

---

## 目录结构

```
adobe_lr/
├── 1_data_collection/        # 线1：数据采集与预处理
├── 2_sft_pipeline/           # 线2：SFT 训练数据流水线
├── 3_llm_eval/               # 线3：LLM 评测 / 奖励模型
├── differentiable_renderer/  # 线4：可微渲染器（包名固定，勿改）
├── 5_training/               # 线5：模型训练（SFT 数据构建 + LLaMA-Factory 配置）
│
├── scraped_data_v4/          # 原始数据集 train/(2277) + test/(1000)
├── all_combined_details.json # 合并后的全量修图参数
├── amortized_init.pt         # 渲染器 CNN 暖启动权重
├── sft_data/                 # 线2 产出（旧 ASSR CoT 方案，已弃用）
├── sft_data_v2/              # 线5 产出（逐维直出 SFT 数据，以 fit_results 为源）
├── eval_results*/            # 线3 评测产出
├── fit_results_*/            # 线4 拟合产出
├── aesthetic_output/         # 美学优化产出
├── Ziyiwang_filter/ ...      # 人工筛选结果（多批）
└── requirements.txt
```

---

## 线 1 · 数据采集与预处理 `1_data_collection/`

| 脚本 | 作用 |
|---|---|
| `scrape_details.py` | Playwright 抓取 Lightroom 修图数据，每图 1 主图 + 最多 3 个无 Mask 版本 |
| `cut_original_images.py` | OpenCV 连通域分析裁掉原图左右黑边 |
| `merge_details.py` | 合并各样本参数 → `all_combined_details.json`（注：内含旧机器硬编码路径，运行前需改 `BASE_DIR`）|

数据清洗辅助脚本在 `scraped_data_v4/` 内（依赖自身位置，勿移动）：
`find_crop_mismatch.py`（尺寸异常）、`find_different_subjects.py`（pHash 找不同主体）、`find_duplicate_versions.py`（去重复版本）。

## 线 2 · SFT 训练数据流水线 `2_sft_pipeline/`

主入口 `run_data_pipline.py` 串联 Step1→4，产出 LLaMA-Factory ShareGPT 格式训练集：

| 脚本 | 作用 |
|---|---|
| `step1_parse_params.py` | 解析修图参数（Unicode 负号、百分号、color_mixer 等）|
| `step2_filter.py` | 按修改幅度 magnitude 过滤排序（色温以偏离 5500K 归一化，带视觉重要性权重）|
| `step3_assr_cot.py` | 生成 ASSR 框架 CoT 标注（规则模板 + LLM）。**内含 NumPy 渲染器**，是线4渲染器的源头 |
| `step4_build_dataset.py` | 组装成 ShareGPT 多模态格式 → `train.json` / `val.json` |
| `filter_dataset.py` | 独立的 Flask 网页人工筛选工具（y/n 键操作，断点续筛）|

```bash
python 2_sft_pipeline/run_data_pipline.py --force_overwrite     # 完整运行
python 2_sft_pipeline/run_data_pipline.py --llm_ratio 0.0       # 只用规则模板，快速测试
python 2_sft_pipeline/filter_dataset.py                         # 启动网页筛选工具
```

## 线 3 · LLM 评测 / 奖励模型 `3_llm_eval/`

| 脚本 | 作用 |
|---|---|
| `evaluate_llms.py` | 多视觉 LLM 在 4 个量化维度上评测修图感知力（走 OpenRouter，需 `.env` 里的 `OPENROUTER_API_KEY`）|
| `distill_gemini_scorer.py` | 把 Gemini 成对偏好蒸馏成本地 CLIP+MLP 打分器（GRPO 奖励模型），依赖 `evaluate_llms.py` |
| `aesthetic_optimizer.py` | 两阶段 MLLM 美学优化：五维诊断 → 逐维度候选 → 本地 OpenCV 渲染选优 |

```bash
python 3_llm_eval/evaluate_llms.py --samples 50 --output eval_results/
python 3_llm_eval/distill_gemini_scorer.py generate --dataset_root scraped_data_v4 --split train --output gemini_pairs.json
python 3_llm_eval/aesthetic_optimizer.py --sample <uuid>
```

## 线 4 · 可微渲染器 `differentiable_renderer/`

目标：`θ = argmin L(Render(原图, θ), 修图后图)`，反解出 Lightroom 修图参数（49 个）。

| 模块 | 作用 |
|---|---|
| `renderer.py` | 核心 PyTorch 可微渲染器（step3 NumPy 版的 1:1 翻译，支持 49 参数）|
| `fit.py` | **主入口**：CNN 暖启动 → Adam → 分块坐标下降(L-BFGS) → 全局抛光 |
| `losses.py` | 复合感知损失（线性 L1 + LAB-ab + ΔE2000 + MS-SWD + 逐色相 ΔE2000）|
| `color_spaces.py` | sRGB↔linear↔XYZ↔LAB 转换、ΔE2000、Bradford 色适应 |
| `amortized_init.py` | 孪生 CNN 预测 θ 初值（产出 `amortized_init.pt`）|
| `crop_align.py` | SIFT 特征匹配做裁剪对齐 |
| `guided_filter.py` / `local_laplacian.py` | 可微导向滤波 / 局部拉普拉斯滤波 |
| `opencv_renderer.py` | `renderer.py` 的 NumPy/OpenCV 逐位等价实现，用于验证 |
| `docs/` | 架构、详细指南、参数速查、索引 |

```bash
# 注意：包用包限定导入，需用 -m 方式运行
python -m differentiable_renderer.fit --data_dir scraped_data_v4/train \
    --output_dir fit_results_demo --amortized_init amortized_init.pt --device cuda --limit 50
```

> **不要重命名 `differentiable_renderer/` 文件夹**：内部用 `from differentiable_renderer.X import ...` 包限定导入，改名会导致全部 import 失效。

## 线 5 · 模型训练 `5_training/`

按 `TRAINING_PLAN.md` 最新拍板落地：任务A「逐维直出」SFT（Qwen3-VL-8B-Instruct + LoRA）。
数据源为 **fit_results 拟合参数**（非爬取参数），详见 `5_training/README.md`。

```bash
python 5_training/build_sft_dataset.py --selfcheck 30   # 构建 sft_data_v2/
bash 5_training/setup_env.sh                            # 独立 conda env lf
bash 5_training/train_sft.sh                            # LoRA SFT
```

---

## 渲染器的三个版本（公式严格一致）

```
step3_assr_cot.py 的 NumPy 渲染器  (最早)
        │ 1:1 翻译
        ▼
differentiable_renderer/renderer.py  (PyTorch 可微，用于梯度拟合)
        │ 1:1 翻译
        ▼
differentiable_renderer/opencv_renderer.py  (NumPy/OpenCV，逐位验证)
```

## 环境

```bash
pip install -r requirements.txt
# .env 需配置 OPENROUTER_API_KEY（线3 用）
```
