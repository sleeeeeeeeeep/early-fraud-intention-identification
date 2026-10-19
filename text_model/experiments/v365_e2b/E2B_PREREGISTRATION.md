# E2-B 预登记实验方案

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-10-12T10:15:00+08:00
- Verification Status: UNVERIFIED
- Version Label: code_plan_v1

## 实验目标

验证“总体类别先验校正 + tE 相对 tE-1 的相邻边界约束”能否在保留诈骗前期识别能力的同时，降低 E2-A 的 tE 前抢报和正常通话过程误报。

## 假设

- H1：E2-B 在 V365 Dev 的连续2轮策略下，tE 后检测率不低于 M1，并降低 E2-A 的 normal 曾误报率。
- H2：E2-B 不增加完整文本 normal 误报，也不损害原 V361 长文本能力。
- H3：若只改善单轮策略而不能改善连续2轮策略，则视为概率尖峰，不视为稳定收益。

## 固定控制变量

- 仅使用 V365 Train 构造训练数据；ambiguous 不参与训练。
- V365 Test 与 restricted Test 标签保持封存，构建、训练、报告脚本均按路径硬拒绝。
- 从 M0 `runs/repro_v361b_b320_head96/best_model.pt` 独立初始化，不从 E2-A 继续训练。
- 主训练集仍为6500条：3900条 V365 派生前缀 + E1-B 实际使用的同一批2600条 V361 回放。
- Route B、`head_context_enhanced_chunks`、chunk=320、overlap=64、head=96、max_chunks=12。
- batch=1、gradient accumulation=8、epoch=1、seed=202、AMP、冻结 backbone、task LR=2e-6，其余参数与 E1-B/E2-A 相同。
- 只训练二分类检测目标；13分类头不参与损失并保持冻结。
- 训练阶段不读取任何 Dev；训练后一次性在固定 Dev 上评测。
- 阈值固定0.5，不在 Dev 上调阈值；连续2轮为主策略，连续1轮仅作诊断。

## E2-B 数据与目标函数

### 主样本构成

复用 E2-A 已审计的1200条唯一 V365 前缀池和同一批2600条 replay。replay 的类别构成固定为诈骗1909、正常691。为使6500条物理训练实例总体严格平衡为3250/3250，V365 3900次抽样固定为：

- fraud tE 前缀：1341次；
- normal 匹配前缀：2559次；
- 所有600个 fraud call 和600个 normal call 至少出现一次；
- 抽样与最终洗牌均由 seed=202 固定。

1341次 fraud 抽样采用“全部600通各2次 + 141次按 decision×scenario 的 Hamilton 比例补充”。normal 的2559次由两部分组成：1341条与 fraud draw 同场景、同累计轮数、同结束说话人并按token长度全局匹配的控制前缀，以及1218条覆盖全部600个normal call的广覆盖前缀。scenario_tag只用于配对，不拼入模型文本。

因此 E2-B 是“总体类先验校正 + 同场景正常控制 + 相邻边界辅助约束”的复合候选实验，而不是单因素消融。结果只能判断该组合方案是否值得保留，不能把变化单独归因于其中某一项。

### 相邻边界辅助视图

每个 tE>1 的 V365 fraud tE 主样本附带同一通话的 tE-1 累计前缀。已知600个唯一 fraud call 中有27个 tE=1，它们没有合法的非空 tE-1 前缀，因此只使用主CE并明确记录 pair mask=0。该辅助视图：

- 不计入6500条主样本数量；
- 不被硬标为 normal；
- 只用于约束二分类诈骗分数的相邻边界关系；
- 不含 tE 及其后的轮次。

该约束只覆盖局部的 `tE-1 → tE` 边界，不能宣称已经约束所有更早的 pre-tE 概率尖峰；若仍有早轮抢报，后续应单独做 hard-negative/轨迹约束实验。

### 损失

令二分类 logit margin 为 `m=z_fraud-z_normal`：

- 主损失：6500条主样本的加权二分类交叉熵 `L_ce`；
- 排序损失：`L_rank=max(0, 0.5-(m_tE-m_pre))`；
- tE前报警上限损失：`L_cap=max(0, sigmoid(m_pre)-0.5)`，只惩罚 tE-1 分数超过0.5的部分；
- 总损失：`L=L_ce+0.20*L_rank+0.10*L_cap`。

不做权重搜索，不根据 Dev 修改上述常数。

## 评测输入与比较对象

- V365 Dev 固定完整通话与固定累计前缀；
- 原 V361 long Dev 240条；
- 比较 M0、M1、E2-A、E2-B；
- 所有模型使用同一推理程序与同一分块参数。

## 预登记硬门槛

### A. 实验有效性门

- 固定 V365 Dev 完整文本与累计前缀输入哈希必须与已审计基线一致；累计前缀共4492条。
- full 与同通话最终累计前缀的 threshold flip=0，且最大概率差≤1e-4。
- 推理 coverage=1；checkpoint strict load 的 missing/unexpected keys 均为空。
- 推理代码和预登记文档 SHA 写入运行清单；Test/restricted 读取数=0。

任一项失败记为 `INVALID`，不能解释为模型优劣。

### B. M1 非劣保持门

E2-B 必须同时满足以下全部条件，才算至少保持现有稳定模型 M1：

1. V365 Dev 完整文本：early_fraud ≥13/15、全部 fraud ≥78/80、normal FP ≤6/80。
2. V365 Dev 连续2轮：early tE后检测 ≥14/15、全部 fraud tE后检测 ≥79/80、explicit tA前检测≥63/65、全部 fraud 有效检测延迟中位数=0且P75≤1轮。
3. V365 Dev 连续2轮：normal 曾误报 ≤11/80、全部 fraud tE前抢报 ≤57/80。
4. V365 Dev 连续2轮：无抢报且最终有效检测的 clean-effective ≥22/80。
5. 原 V361 long Dev：fraud 命中120/120、normal FP ≤2/120。
6. 不出现数据/权重加载异常，训练与推理均正常退出，检查点可 strict load。

### C. 真正改进门

只达到 B 代表与 M1 持平，不足以称为改进。若要成为后续盲测冻结候选，还必须同时满足：

- 连续2轮 normal 曾误报≤10/80；
- 连续2轮全部 fraud tE前抢报≤56/80；
- 上述两项的事件数之和≤65（M1为68，至少净减少3个报警负担事件）；
- 连续2轮 clean-effective≥23/80；
- A、B 全部通过。

连续1轮结果、AUC和概率轨迹全部报告，但不替代上述硬门槛。报告 Wilson 95%区间、同一call的离散转移、描述性的 exact McNemar/binomial 差异；小样本1条变化不得表述为稳定提升。任何 B 项失败记为 `FAIL`；B 通过但 C 未通过记为 `HOLD`；A/B/C 全通过才记为 `PASS_FOR_FREEZE`。即使通过，也只表示“可冻结候选”，V365 Test 仍需在模型、阈值和报警策略冻结后另行授权进行一次盲评。

## 预期产物

- `data/processed/v365_e2b_boundary_balanced_60new_40replay_s202/`
- `runs/v365_e2b_boundary_balanced_s202/best_model.pt`
- `runs/v365_e2b_dev_compare_20261012/`
- 数据审计、训练清单、strict-load 检查、四模型 Dev 对比与门槛判定报告。

## 停止条件

- 输入路径、源哈希或构成不符合合同：训练前停止。
- GPU/训练进程崩溃：保留日志，不静默重试。
- 磁盘空间不足以安全写入完整检查点：训练前停止。
- 任意 Test/restricted 路径被传入：立即拒绝。
