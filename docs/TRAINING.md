# 训练与评测流程

## 数据主线

1. V361：长文本训练 6500 条，Long-Dev 240 通，另有历史 stress 对照；用于保留长文本能力并形成 M0。
2. V362：严格前缀标注和筛选，得到 E1-B 的前缀来源；3900 次前缀采样 + 2600 条长文本回放构成 6500 次训练实例。采样次数不等于唯一通话数。
3. V365：合成通话经独立复核、裁定和重建，最终目录名 `v365_2000_final_20260923` 在历史脚本中保留。训练集用于 E2 构建，固定 Dev 用于对照；Test 与受限标签封存。

E2 的 tE 派生目标属于弱监督，不是每个前缀都经过独立盲标。不能把 tE 前的所有文本简单视为「明确正常」，也不能在构造当前前缀特征时看后续文本。

## 训练版本

| 版本 | 数据构造 | 训练入口 | 关键约束 |
| --- | --- | --- | --- |
| M0 | V361 长文本 | `text_model/inference/train_longdoc.py`、V361 配置与 suite 脚本 | Route B，头部上下文 96 |
| M1 | `build_v362_strict_prefix_inputs.py`、`build_v362_e1b_mixed_train.py` | `train_v362_e1b_binary.py` | M0 初始化；60/40 混合；骨干冻结 |
| E2-A | `build_v365_e2a_te_mixture.py` | `train_v365_e2a_binary.py` | tE 边界前缀与匹配正常前缀 |
| E2-B | `build_v365_e2b_boundary_mixture.py` | `train_v365_e2b_binary.py` | 平衡类别、匹配正常、局部排序和分数上限 |
| E2-C | `build_v365_e2c_k2_pair_mixture.py` | `train_v365_e2c_binary.py` | 移除 E2-B 排序/上限损失，改为相邻前缀约束 |
| E2-D | `scripts/e2d/build_e2d_train_inputs.py` | `scripts/e2d/train_e2d_gate.py` | 冻结 M1，门控只拦截既有 k2 报警 |

上述 `build_*.py` 位于 `text_model/scripts/data/`，`train_*.py` 位于 `text_model/scripts/training/`，E2-D 的文件单独位于 `scripts/e2d/`。E2-A/B/C 均从 M0 独立初始化，不是串行微调。

M1 归档启动参数：chunk=320，overlap=64，head=96，batch=1，梯度累计=8，训练 1 轮，骨干冻结 1 轮，encoder LR=0，task LR=2e-6，seed=202。原启动器还设置完整文本 accuracy floor=0.99、prefix FPR ceiling=0.05；这些训练筛选条件与 V365 的最终联合替换验收不是同一件事。

E2-B 为多因素组合实验，不能将效果归因于某一个损失项。E2-D 使用前后两轮的分数 margin、长度、轮次和块数变化等 10 项特征，网络为 `10 → 16 → 1`，共 193 参数。

## 评测顺序

1. 固定配置、权重和数据版本；保存清单和哈希。
2. 分别评估 V365 Dev 全文、逐轮累计前缀以及原 Long-Dev。
3. 全文分数固定为 `p_binary[1]`，阈值 0.5；累计前缀主口径固定 k2。
4. 同时检查 early 有效检测、normal 曾误报、tE 前抢报、tA 前有效报警及延迟。
5. 依据预登记和联合验收判断是否替换 M1，不能只挑某个更好的指标。

E2-D 最终报告采用 20261016 schema 修复后的汇总。其全文结果是 M1 原输出复用；修复报告不等于重训模型。最终结论见 [实验汇总](../reports/实验结果汇总.md)。

V365 Test 在本次归档中没有读取内容或评测。历史 V361 文档中可能出现 Test-B / stress 的既有结果，它们不等于 V365 Test 被解封。
