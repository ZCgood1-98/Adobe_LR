# 线 5 · 模型训练 `5_training/`

落地 `TRAINING_PLAN.md` 拍板方案（2026-06-23 / 07-04 / 07-12 修订）的 **Phase 1-2**：
任务 A「逐维直出」SFT —— 数据源为 **`fit_results/` 拟合参数**（渲染器参数空间，与
GRPO 的渲染/奖励天然同一空间；非爬取的 Lightroom 原始参数），`fit_annotations.json`
提供类别 / 黑白 / keep-discard 过滤。

与 TRAINING_PLAN 的对应关系（防迷路）：

| plan 概念 | 落在哪 |
|---|---|
| Phase 1 任务A 数据（直出、无 CoT、类别条件） | `build_sft_dataset.py` → `sft_data_v2/` |
| Phase 2 SFT（Qwen3-VL-8B + LoRA，LLaMA-Factory） | `sft_lora_qwen3vl8b.yaml` + `train_sft.sh` |
| 「逐维渲染一批候选→选最优」循环 · 训练侧 | = **Phase 3 GRPO 本身**（组内 N rollout = N 候选，参考距离 = 选优），未落地 |
| 同上 · 推理侧 | = 推理编排器（beam-2 + 末端 N 轨迹 selection），未落地；老参考实现 `3_llm_eval/aesthetic_optimizer.py` |
| Phase 1 任务B（montage selection） | 未落地，依赖任务A模型能采样 |

## 流水线总览与当前状态

```
fit_results/ + fit_annotations.json
   │
   ├─ ① build_sft_dataset.py ──────────► sft_data_v2/（样本+中间图）      ✅ 2026-07-13
   ├─ ② scan_render_consistency.py ───► render_consistency.json           ✅ 全量已扫
   ├─ ②b rescue_geometry.py（救回几何错位 version，见质量门②）           ✅ 救回 422
   ├─ ③ 重跑 ① 套一致性门（图片幂等，val 已钉死）                         ✅ 已重组
   ├─ ④ setup_env.sh（conda env lf + 下载模型 17GB）                      ✅ 已就绪
   ├─ ⑤ 冒烟训练（max_samples=64）                                        ✅ 已通过
   ├─ ⑥ bash train_sft.sh（正式 SFT v1，实跑 13.3h）                      ✅ 2026-07-16
   ├─ ⑦ diagnose_stage_error.py（分 stage 图像空间诊断）                  ✅ v1 不及格 → 立项 v2
   ├─ ⑧ SFT-v2（值离散化 bin token + 合成机制课）                         ⬜ 已立项，见下节
   └─ ⑨ 任务B 数据 / GRPO / 推理编排器                                    ⬜ 线5 后续
```

| 文件 | 作用 |
|---|---|
| `build_sft_dataset.py` | fit_results + 标注 → 逐维 ShareGPT 样本 + 中间图渲染 → `sft_data_v2/` |
| `scan_render_consistency.py` | 全量扫「累积渲染 vs rendered.png」差异 → `render_consistency.json`（质量门②依据） |
| `rescue_geometry.py` | 救回被门②过滤的「输入取景错位」version：强制生成对齐原图 → 复测 → 过则收回、不过则回滚 |
| `make_refit_list.py` | 生成重拟合清单（fit.py `--only` 格式），自动跳过已重拟合的（fit_meta 含 crop_box） |
| `sft_lora_qwen3vl8b.yaml` | LLaMA-Factory SFT 配置（bf16 LoRA r32，冻结 ViT） |
| `setup_env.sh` | 建独立 conda env `lf`（不动 Chen 环境）；`DOWNLOAD_MODEL=1` 顺带拉权重 |
| `train_sft.sh` | 训练启动，支持追加 `key=value` 覆盖参数 |
| `diagnose_stage_error.py` | SFT 后分 stage 图像空间诊断（定「要不要拆大 stage」） |

## 数据构建

### Stage 分组（v1 拍板，4 阶段）

顺序 = TRAINING_PLAN 四·优先级表（感知杠杆从高到低）。选大 stage 而非细拆的理由：
stage 内参数在**同一次自回归输出里互为条件**（先吐 contrast，后面的 tone_curve 条件
在它上生成），组内协调保留在条件链里；细拆则协调只能隔着渲染图弱耦合传递，且轨迹
变长加剧误差累积（plan 坑2）、GRPO 渲染成本线性涨。要不要拆由 ⑦ 诊断的数据决定。

| Stage | 模型输出 JSON 键 | 数字个数 |
|---|---|---|
| 1 light | exposure, temperature(K), tint | 3 |
| 2 tone | contrast, highlights, shadows, whites, blacks, tone_curve(8), dehaze | 14 |
| 3 color | vibrance, saturation, color_mixer(8通道×hsl), color_grading(3区×rgb) | 35 |
| 4 detail | texture, clarity, sharpen, noise_reduction, vignette | 5 |

改分组只动 `extract_stages` + `STAGE_PROMPTS` 两处；构建器幂等，重跑只补渲受影响的中间图。

### 样本形态

每个 keep version 出 4 条样本（`s1..s4.jpg` 为各 stage 输入图，s1=原图缩放，长边 1024）：

```json
{"messages": [
   {"role": "system",    "content": "You are a professional photo retouching agent. ...4 stages..."},
   {"role": "user",      "content": "<image>\nCategory: portrait. Black & white target: no.\nStage 2/4 (tone): ...\nOutput only the JSON object."},
   {"role": "assistant", "content": "{\"contrast\": -8.2, ..., \"tone_curve\": [...], \"dehaze\": -36.0}"}],
 "images": ["images/train/<gid>/v0/s2.jpg"],
 "_meta": {"group": "train/<gid>", "version": "0", "stage": "tone", "category": "portrait", "bw": false}}
```

- 中间图 = 用「已决定阶段的累积参数」从原图**一次性**渲染（渲染器管线顺序固定，参数
  子集渲染即得一致中间态；**禁止链式渲染**——会重复叠加 detail_transfer、打乱算子序）；
- 输入图：version 目录有 `original_cropped.png` 用它（2510 个裁剪对齐 version），否则组级
  `original_captured_cropped.png`；
- 渲染与 JSON 标签共用同一份舍入值（曝光 2 位小数、色温整数、其余 1 位），教师强制态
  与标签严格一致；
- 类别 token（7 类英文）+ 黑白 flag 注入 user prompt（plan「类别=条件信号，非独立 LoRA」）；
  黑白 version 的 color 阶段标签自然是 saturation/vibrance = -100；
- 温度/晕影等方向性语义在 prompt 里**故意中性**（渲染器 temp 方向与 LR 滑块相反，
  没验证过的语义不写，靠数据学）。

### 双质量门

| 门 | 规则 | 2026-07-13 实测 |
|---|---|---|
| ① 拟合质量 | `fit_meta.final_loss > 8` 弃用（`--max_final_loss`） | 过滤 205 个 version（2.2%） |
| ② 渲染一致性 | 累积渲染 vs `rendered.png` 平均\|Δ\| > 8/255 弃用（`--max_render_diff`，依赖先跑 `scan_render_consistency.py`） | 初扫 p50=1.66、>8 共 826 个（9.25%），成因 100% 与 crop 对齐相关。其中 596 个「拟合对齐过但没生成 `original_cropped.png`」的纯几何错位已由 `rescue_geometry.py` 修复：强制 SIFT 补齐对齐原图 → 复测 → **救回 422 个**（diff 中位 13.4→3.1），172 个复测未过回滚（病根=拟合质量）、2 个 SIFT 无框。**终态过滤 404 个（4.5%）**（172 回滚 + 230 有对齐图的差拟合 + 2 无框）；这批的救法只剩重拟合，暂不做（明细 `rescue_geometry_log.json`） |

**为什么「累积渲染 vs rendered.png」会有非零差异**（诊断读数的构成，不是 bug）：
两张图有四个变量不同——① 输入图几何（rendered.png 渲自拟合时 crop 对齐的原图；无
`original_cropped.png` 时构建器退回整幅组原图 → 错位，**10~30/255，超标的唯一主因**）；
② 渲染器浮点差（torch f32 GPU vs numpy f64 CPU，公式一致但数值路径不同，1~3/255）；
③ 分辨率 2048 vs 1024 的重采样与空间算子近似（0.5~2/255）；④ 参数舍入（≪1/255）。
基线 p50≈1.7 = ②+③+④ 的固有底噪，消不掉也无需消；门槛 8 正是用来把底噪与几何错位
分开的。**底噪不进训练闭环**：训练标签、中间图、GRPO 渲染与 reward 统一走
opencv_renderer + 舍入参数（自洽），rendered.png 只是拟合期快照、仅用于本诊断。

### 当前数据集（`sft_data_v2/`，5.7GB）

| split | 组 | version | 样本(=version×4) | 用途 |
|---|---|---|---|---|
| train | 2466* | 6972 | 27888 | SFT 训练 |
| val | 49 | 137 | 548 | eval_loss / 早停 / 诊断；**下游 RL 也须排除**（清单 `val_groups.json`，构建器自动钉死沿用） |
| test | 491 | 1418 | 5672 | 只导出不参训；SFT 阶段离线评测用 |

\* train+val 共 2515 组按组切分（几何救援后 2026-07-13 终态）。注意：磁盘上
`fit_results/test` 的 **500 组固定评测集不变**，task_a_test.json 只是其中过了质量门的
部分（GRPO/终评仍以 500 组为准）。
其余产物：`build_report.json`（构建统计）、`render_consistency.json`（一致性明细）、
`val_groups.json`（隔离清单）、`dataset_info.json`（LLaMA-Factory 注册）。

## 环境与硬件

- GPU：**RTX 5000 Ada 32GB**（bf16 LoRA 实测峰值 26.8GB，不需 QLoRA；OOM 兜底：yaml 里
  解开 `quantization_bit: 4`）；
- conda env `lf`（python 3.11）：llamafactory **0.9.5** + transformers **5.6.0** + accelerate
  **1.11.0**（llamafactory 的钳制上限，**装完别单独 -U 升级这俩**，5.13/1.14 会破坏兼容）；
  `hf download Qwen/Qwen3-VL-8B-Instruct` 已缓存 17GB（`huggingface-cli` 已废弃，用 `hf`）；
- ⚠️ CLI 覆盖参数走 YAML 解析：`eval_strategy=no` 会被解析成布尔 `False` 直接报错
  （YAML 1.1 把 no/yes 当布尔）。这类值要么改 yaml 文件，要么别覆盖。

## 训练

```bash
# 冒烟（已通过：4 步 loss 2.78→1.49，eval_loss 1.49，显存 26.8GB）
bash 5_training/train_sft.sh max_samples=64 num_train_epochs=1 output_dir=5_training/output/smoke

# 正式（27888 样本 × 2 epoch / 有效batch16 ≈ 3486 步 × ~20s ≈ 19h）
bash 5_training/train_sft.sh
```

配置要点（`sft_lora_qwen3vl8b.yaml`）：LoRA r32/α64 挂全部线性层、冻结 ViT、
lr 1e-4 cosine、bf16 + grad ckpt + sdpa、cutoff 3072、`image_max_pixels` 1048576
（与构建的 max_side=1024 匹配）、每 200 步存 checkpoint（留 3 个）+ 跑 val 评估。
产出 `5_training/output/sft_v1/`（LoRA adapter，供诊断/推理加载）。

## SFT 后诊断：分 stage 图像空间误差 `diagnose_stage_error.py`

回答「模型掌握了哪个 stage、要不要拆分大 stage」——token loss 因参数多解冗余
（同一视觉效果多种参数组合）有噪声下限、不可比，图像空间距离才是公平尺子
（与拟合/GRPO reward 同一把 `CompositeLoss`，黑白样本自动换 BW 权重档，param_reg=0，
SWD 随机投影已固定种子）：

```bash
python 5_training/diagnose_stage_error.py all               # gen 推理(GPU ~35min) + score 评分(CPU ~8min)
python 5_training/diagnose_stage_error.py gen --no_adapter  # 裸基座对照组（产出 diag_base/）
```

- 对 val 每条样本渲三张：`render(前序GT ∪ 预测)` / `render(前序GT ∪ GT)` / `render(前序GT ∪ 全零)`；
- **改善中位 = 1 − L_pred/L_noop**：≈1 学得好，≈0 不比「这步全不动」强——排除 detail 这类
  GT 常为零的 stage 靠输出全零蹭低距离的假好成绩；
- 附每参数 MAE（定位 stage 内谁没学会）+ 黑白子集单列；断点续跑；
  产出 `sft_data_v2/diag/{predictions.jsonl,scores.jsonl,report.json}`；
- 机械自检已验证：GT 当预测 → 改善=1.0；全零预测 → 改善=0；
- **决策规则**：某 stage 改善中位显著落后 → 只拆那个 stage（候选嫌疑：color，35 个数）。
  四个都不差 → 大 stage 之忧被数据排除，直接进 GRPO。

## 重拟合流程（修复终态过滤的 404 个，~9h GPU，Chen 环境）

```bash
# ① 生成清单（分 train/test 两份；自动跳过已重拟合的 → 中断后重跑此步即续跑）
python 5_training/make_refit_list.py

# ② 重拟合（GPU 空闲时；nohup 后台，日志落盘。--force 必须：旧 params.json 存在）
nohup python -m differentiable_renderer.fit --data_dir try_new_dataset1/train \
    --output_dir fit_results/train --only sft_data_v2/refit_train.txt \
    --force --amortized_init amortized_init.pt --device cuda \
    > 5_training/refit_train.log 2>&1 &
# train 跑完后同样跑 test：
nohup python -m differentiable_renderer.fit --data_dir try_new_dataset1/test \
    --output_dir fit_results/test --only sft_data_v2/refit_test.txt \
    --force --amortized_init amortized_init.pt --device cuda \
    > 5_training/refit_test.log 2>&1 &

# ③ 重扫一致性（全量 ~25min；重拟合的 version 会拿到新 diff，应≈底噪 2/255）
python 5_training/scan_render_consistency.py

# ④ 重建数据集（过期检测自动重渲受影响中间图；val 已钉死不漂移）
python 5_training/build_sft_dataset.py
```

中断续跑：kill 后重新执行 ①（清单自动缩短）再执行 ②。重拟合后仍过不了双门的
version（拟合真的表达不了的编辑）继续被过滤——此时过滤名单已是"渲染器表达力"
的真实边界，而非几何事故。

## 数据隔离约定（防泄露）

- **test 树四不进**：SFT / GRPO / 任务B 标签生成 / 线路2 偏好对，一律不得使用 test 组；
- `val_groups.json` 的 49 组：下游 RL/任务B **同样排除**，否则 val 失去跨阶段比较意义。
  SFT 与 RL 共用其余 train 组**不算泄露**（标准做法；reward 是图像空间固定度量、无学习型
  RM，复现参考=方案设定的最优点而非作弊）；
- `sft_data/`、`eval_results/` 是旧划分（2026-07-12 resplit 前）产物，不得与新划分模型比数字。

## SFT-v2（值离散化 bin token）—— 已立项 2026-07-16

v1 图像空间诊断不及格（`python 5_training/analyze_diag.py`：改善中位 light 0.0 / tone 0.011 /
color 0.0 / detail −0.964；病根 = 数字串上的 CE 概率 → 保守塌缩 + ±100 模式坍缩；
格式本身 588/588 全对 → **不拆 stage**）。v2 方案：**57 键 token + 双桶组值 token**（曝光专属 <ev_> 127 桶 + 其余共享 <v_> 63 桶，共 247 个，奇数桶中性落桶心，
键均值初始化接语义先验、训练时当锚点/推理时受限解码强制注入；bin 按每参数中性点锚定归一化到 [0,1]、
μ-law 非均匀分桶；全部映射进词表保留区 151669+，peft `trainable_token_indices` 只训新增行）
+ **有序软标签 CE** + **合成机制课**
（均匀随机参数 × 库存原图 → 渲染 → 逆向任务，喂饱 bin 尾部并教「参数→效果」）
+ **审美诊断辅助课**（2026-07-20 拍板 day-1 加入：老师 VLM 按 version 产「问题/气质/调整方向」诊断文本、
方向一致性过滤器机械拦幻觉、只陈述改前图可见事实、独立辅助样本不进参数输出路径 → rollout 零增量；
把 图→参数 因子化为 图→诊断→参数，教「为何这么修」）。
超参 2026-07-19/20 拍板（共享63桶+曝光127桶/μ=8、软标签 σ=1.0、中性桶降权 0.15 封底、色温 mired 轴、键 token 屏蔽 loss、
机制课 70%组合+30%单参隔离、课程表 20%纯机制→1:1→收尾品味加倍），先按此跑、首诊后按数据调。
完整拍板 / 工作分解（5 项 ~1 周）见 `TRAINING_PLAN.md`「★ 设计修订（2026-07-16 立项）」。
~~Phase 0 方差实测~~ 已于 2026-07-20 取消：**拍板 GRPO 等 v2，直接开工**（未拍板清单清空）。

**v2 执行 pipeline（2026-07-20 拍板，师/生 prompt 已定稿）**：
```
A v2_codec.py(分桶/247token/初始化, 门=全量GT标签round-trip渲染底噪级)─┐
B v2_direction_probe.py(方向校准)─► D v2_diagnosis_gen.py             ├─► E v2_train.py ─► F 正式训练 ─► G 诊断对比
  (D1=50条试产👤用户翻阅 → D2=全量7474, 老师gemini-3.1-flash-lite)    │   (独立HF Trainer,  (~20h)      (GRPO放行双门:
C v2_build_dataset.py(品味课换格式复用s1-s4图 + 机制课合成)───────────┘    冒烟门=只动247行)              改善>0+采样方差)
```
用户出场点：OpenRouter key（D1 前）→ 翻阅 50 条试产 → 冒烟后发令正式训练 → 看 G 报告拍板 GRPO。
风险与后手：自研训练器（transformers 5.6 原生 Qwen3-VL processor 兜底）、机制课双图显存（降 batch/缩 768）、老师质量（试产即门）。

### ev 组信号稀疏：已做的缓解 + 升级路径「层级离散化」（未实现，防遗忘存档）

**问题**：曝光专属 `<ev_0..126>` 只由曝光标签训练（品味课 ~1.5 万 + 机制课 ~1.2 万次/epoch，
摊到 127 个 token），而共享 `<v_*>` 由 56 参数的 ~52 万次/epoch 共同打磨——分组词表切断统计共享。

**已实现的缓解（2026-07-20）**：
① σ 按桶密度缩放（`v2_train.build_soft_labels`：σ = 1.0 × 桶数/63 → ev 组 σ≈2）——值域平滑与共享组
对齐（一致性修正），且每次发射训 ~13 个邻域 token，等效信号密度翻倍；
② 机制课 stage 采样加权 light=0.4（`v2_build_dataset._mech_worker` 的 `stage_p`）——曝光发射
7.8k→12.4k/遍，且 bin 空间均匀采样把 127 桶尾部系统性喂满（品味课曝光 70% 挤在中性区）。
两项叠加后每 ev token 等效 ~2800 次/epoch，与共享组差距从 20 倍缩到个位数。

**升级路径：层级离散化（coarse + fine 编码）**——真正「删除」问题而非缓解：

```
现在:  <exposure><ev_55>                    127 个曝光专属 token
升级:  <exposure><v_41><f_3>                粗桶复用共享 <v_*>(63) + 细分 <f_0..4>(仅 5 个新 token)
解码:  值 = 粗桶中心 + (fine−2)/5 × 粗桶宽
```

- **为什么有效**：曝光的粗结构直接骑上共享组 52 万次/epoch 的梯度流（`<v_41>` 对曝光和对比度
  是同一个 token）；曝光专属词表只剩 5 个细分 token、每个 ~4600 次发射，很肥；
  分辨率 63×5=315 级 > 127 级；`<ev_*>` 的 127 个 token 整组退役。
- **代价**：codec 编解码改双 token；构建器标签重生成；损失函数细分 token 的软标签在**粗桶边界**
  处有拼接麻烦（粗桶 41 的 fine_4 与粗桶 42 的 fine_0 是值域邻居但 token 不相邻——近似做法：
  粗桶用 63 桶高斯软标签、fine 在真值粗桶内条件式软化、跨界不平滑）；每条 light 标签 +1 token。
- **触发条件**：步骤 G 诊断中曝光维 MAE / 图像空间改善显著弱于其他维，且归因于 ev 组学不好
  （而非模型整体能力）时上马；改动局部（codec + 构建器重跑 25min + 损失一处），~1 天。
- 同机制也适用于单向参数（texture/clarity/sharpen/NR 目前 per-param 32 级——评估为足够，
  它们是感知杠杆最低的维度；若 G 诊断翻案，同一套 coarse+fine 直接套用）。

## 后续路线（线5 未落地部分）

1. **任务 B（montage selection）**：任务A SFT 模型能采样后，渲 N 候选终图 → 参考距离标
   「最优编号」→ 只 SFT 蒸馏、不进 GRPO（plan §D）；
2. **Phase 3 GRPO（仅任务A）**：奖励 = `losses.py` 复合距离对真人版本，v* 按轨迹锁定
   （**`try_new_dataset1/` 是参考图来源，不可删**）；逐 stage 稠密 reward = 距离改善量；
   曝光/白平衡 beam-2 训练时用当步 min-over-versions 剪枝。已知观察项：同图 rollout 组内
   reward 方差若塌缩（SFT 记忆解主导），备选留部分 train 组专供 RL，默认不拆池；
3. **推理编排器**：逐维直出 + beam-2 + 末端 N 轨迹 montage → 任务B 选优；骨架可参考
   `aesthetic_optimizer.py`。**未拍板**：推理时（无参考图）beam-2 的剪枝依据——砍掉轨迹内
   beam 只留末端 selection，或让任务B 模型学会中间态两选一；
4. 类别不平衡上采样（样本已带 `_meta.category`，训练侧后续处理）。

## 已知边角

- 几何救援给 422 个 version 新增了 `fit_results/.../version_N/original_cropped.png`
  （加上原有 2510 个共 2932 个），输入约定不变；救援明细在 `sft_data_v2/rescue_geometry_log.json`；
- **终态过滤的 404 个为什么"crop 也救不回"**：根因是 fit.py 当年只存了
  `crop_aligned` 布尔、**没存实际用的 crop box** → 现在只能事后估计；弱纹理/重编辑样本
  SIFT 估计不稳（回滚的 172 个中：104 裁后改善但卡 8~12、37 没变、31 反而变差），
  且这批 final_loss 中位 4.1——当年拟合就是对歪输入硬拟的，标签本身次级。
  **彻底解 = 重拟合 404 个（流程已就绪，2026-07-13）**：fit.py 已改造——
  ① fit_meta 记录 `crop_method/crop_box/align_coverage` 等完整几何；② 拟合实际用的
  对齐输入直接写入 `version_N/original_cropped.png`（`--no_save_aligned_input` 可关）
  → 训练输入 == 拟合输入**按构造成立**，几何类问题永久消失；③ 新增 `--only 清单` 定向
  重拟合。构建器同步加了过期检测（params.json/输入图比中间图新 → 自动重渲）。
  完整指令见「重拟合流程」一节；
- `train/e67314d5-…` v1/v2：源缺 main_image、无拟合，构建时自动跳过（该组 v0/v3 可用）；
- 被质量门过滤的 version 的 s1..s4 图片留在 `sft_data_v2/images/` 无害（放宽门槛重跑可复用）；
- 构建器全部路径相对仓库根目录，**必须从根目录运行**（与全仓库约定一致）。
