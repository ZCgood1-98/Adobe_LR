# 线5 GRPO 阶段 · 离线分析脚本 + 日志存档

从 **SFT-v2 收尾 → GRPO 探索实验**期间的**离线分析脚本**(纯 CPU、读现有预测/数据、几秒出结果)与**训练/诊断日志**。
原在会话临时目录(scratchpad),此处为持久存档。完整研究叙事见 `TRAINING_PLAN.md` ★2026-07-24 节 与 memory `training-line5-grpo.md`。

> ⚠ 所有脚本**从仓库根目录**运行(内部用相对路径 + `sys.path` 挂 `5_training/`)：`python 5_training/analysis/<脚本>.py`
> 结果均为**终端 print、未存盘**——重跑即复现。数据源在仓库内持久:
> `sft_data_v2/diag_*/predictions.jsonl` + `fit_results/<组>/version_<n>/params.json`

## 分析脚本(按实验层)

| 脚本 | 层 | 回答的问题 | 关键结果 |
|---|---|---|---|
| `dir_vs_mag.py` | 0·SFT-v2 诊断 | 病是感知还是校准? | **校准病**:高杠杆维方向全对(sat r0.95 / vib 0.96 / exposure 0.71 / contrast 0.85)、普遍欠调;temperature r≈−0.02 感知洞、vignette 超调 2–3×。→ 定调转 GRPO |
| `gate2_variance.py` | 1·GRPO 门② | 组内 rollout 有方差吗(GRPO 有得选吗)? | **有**:组内 reward std 中位 ≈2.5、最优 rollout 超组均 → 门②过。(曾踩"给 ms_swd 定 seed 污染采样 RNG"的坑,已修) |
| `wb_clamp_test.py` | 2A·WB | WB 真在拖 light 吗? | **是**:色温清中性 → light −0.044→−0.004、+色调 →+0.000。模型 WB 编辑净有害。(→ 我提 clamp、用户否:WB 是目标) |
| `temp_bin_analysis.py` | 2B·WB | 是分桶太粗吗(用户假设)? | **不是**:5500→桶31 / 5600→桶28 分得开、往返误差 14K、GT 用 36/63 桶、**桶空间 r 0.159 ≈ 值空间 r 0.171**。真病=感知弱+保守(模型只用 8/63 桶) |
| `explore_priority.py` | 2C·WB | 该逼谁多探索? | contrast/texture/clarity/exposure→**B 档·该加强**;sat/vib→**A 档·保护**;temperature(WB)→**C 档·弱信号探索难救**(r0.17,后被 run1/2 实测坐实) |
| `compare_report.py` | 工具 | 两个 diag 报告卡并排比 | 用法:`python 5_training/analysis/compare_report.py <基线 diag 目录> <新 diag 目录>` |
| **`dbg_ctx.py`** | **3·揪出致命 bug** | rollout 到底有没有在看图? | 🔴 **`train()`+梯度检查点 ⇒ `use_cache=False` ⇒ `past_key_values=None` ⇒ 增量解码丢掉 prompt+图像**。铁证:亮/暗两图输出逐位相同。修后:亮图→曝光 −0.7、暗图→+1.1。**任何生成路径都该配这种「换输入→输出必须变」的不变量测试** |
| `dbg_chain2.py` | 3·全链归因 | 全链崩塌是 exposure bias 还是参数组合? | **A(自渲) ≈ B(GT中间图) ⇒ exposure bias 不是主因**；病在参数组合本身(color 阶段伤害最大、晕影双峰失控) |
| `dbg_chain.py` | 3·全链初查 | prompt 拼装是否与 SFT 一致 + D 轨迹 | input_ids 逐 token 相同(len 1123)✓；D 轨迹暴露 A/B 恒等 → 引出 `dbg_ctx.py` |

## 结果存档（results/）
`chain_eval.json`（全链评测主判据，各模型汇总）与各轮 `diag_*_report.json`（per-stage 报告卡）。
⚠ per-stage 报告卡**有 GT 上下文托底、会美化模型**，主判据以 `chain_eval.json` 的全链改善为准（见 TRAINING_PLAN ★2026-07-28/29）。

## 日志(logs/)

| 日志 | 内容 |
|---|---|
| `grpo_explore.log` | run1 全探索训练(greedy 峰值 step90 后崩) |
| `grpo_explore2.log` | run2 收窄探索训练(单调、稳定) |
| `grpo_smoke.log` / `grpo_smoke2.log` | 探索温度 + KL 机制 smoke 验证 |
| `diag_explore100.log` / `diag_explore2.log` | run1(step_100)/ run2 报告卡诊断 |
| `serve.log` | 诊断复审页 SSH 转发 |

## 其他(SFT-v2 期开发工具,留档)

`sim_curriculum.py`(2/1/2 课程表位置-span 模拟)、`test_resume.py`(断点续跑测试)、`dbg_gen.py`(受限解码生成调试)。

---
## 落盘结果对照(不在本目录、在仓库别处)

- 报告卡诊断:`sft_data_v2/diag_grpo`(基线 grpo_tune)、`diag_explore100`(run1)、`diag_explore2`(run2)——各含 `report.json`/`predictions.jsonl`/`scores.jsonl`
- 训练输出:`5_training/output/grpo_tune`、`grpo_explore`、`grpo_explore2`——LoRA adapter + step_50/100/150 checkpoint
