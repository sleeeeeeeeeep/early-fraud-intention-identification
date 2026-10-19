# E2-C 预登记实验方案

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-10-12T16:30:00+08:00
- Verification Status: UNVERIFIED
- Version Label: code_plan_v1

## 实验目标与唯一新概念

E2-C 检验一个新概念：训练目标与现网“连续2轮超过阈值才报警”的 k2 策略直接对齐。模型在相邻两个累计前缀上都超过0.5时才受到辅助惩罚；单轮尖峰不惩罚。目标是在保持 tE 后检测能力和原长文本能力的同时，减少诈骗 tE 前抢报及正常通话过程误报。

E2-C 从 M0 独立初始化，不从 E2-B 续训。它固定复用 E2-B 的6500条物理主样本；移除所有新增 `e2c_*` 字段后，行顺序、sample_id、主文本、标签、权重及 replay 必须逐行恢复为冻结的 E2-B 文件。相对 E2-B，唯一训练机制变化是：移除 E2-B 的 tE-1 排序/上限损失，换成 k2 相邻前缀损失。相对 M1，E2-C 仍包含 E2-B 已引入的类别平衡和同场景正常控制，因此不能把最终变化完全归因于单一因素。

## 数据与安全边界

- 构造和训练只读取 V365 Train、冻结 E2-B 混合文件及其清单；ambiguous 不参与训练。
- V365 Dev 仅在训练完成并固定唯一 checkpoint 后用于一次评测。
- V365 Test、restricted Test labels 和其他 Test 答案全程封存；所有入口按路径硬拒绝。
- 物理主样本固定6500条：V365派生3900条、同一批V361 replay 2600条，总体fraud/normal为3250/3250。
- replay 不附辅助pair。V365每个物理draw最多附一个pair，辅助视图不计入6500条，也不赋任何诈骗/正常硬标签。

### fraud pair

相邻pair为同一通话的累计前缀 `1..e-1 → 1..e`，其中 `e < tE`，因此两端都严格位于首次证据充分边界之前。每个call的物理draw按sample_id排序，候选终点固定按“最早、最靠近tE、再向中间交替”的顺序 `[2, tE-1, 3, tE-2, ...]` 分配；候选耗尽才循环。审计得到1279个fraud物理draw有合法pair，62个tE=1 draw无合法非空pair，只使用主CE。

### normal pair

相邻pair来自同一条 V365 Train 正常完整通话，每个辅助视图自身只含 `1..endpoint`，严格因果。固定优先顺序为 `[2, 3, n_full, 4, n_full-1, ...]`，兼顾最早的连续误报与通话后段误报。2559个normal物理draw全部附pair。部分pair会晚于与它并列的物理主前缀，但仍只来自同一Train正常通话，不涉及Dev/Test或标签泄漏；清单必须报告具体数量，不能声称全部位于主前缀内。

## 固定训练目标

令每个文档二分类 margin 为 `m=z_fraud-z_normal`。主损失是6500条物理行上的原加权二分类交叉熵。对辅助相邻pair：

`L_k2 = max(0, min(m_prev, m_curr))`

只有当两个margin均大于0，即两个前缀都达到 `p_fraud>0.5` 时，该损失才非零。总损失固定为：

`L = L_ce + 0.20 * L_k2`

margin buffer固定为0；不保留E2-B的rank/cap损失；不搜索权重。

## 固定控制变量

- M0初始化；Route B；`head_context_enhanced_chunks`。
- chunk=320、overlap=64、head context=96；`max_chunks=12`沿用项目诊断语义，不做硬截断。
- batch=1、gradient accumulation=8、epoch=1、seed=202、AMP、冻结backbone、task LR=2e-6。
- 只训练二分类检测目标；13分类相关头冻结且不参与损失。
- 单次固定训练；不根据Dev挑epoch、checkpoint、阈值或报警策略。
- 阈值固定0.5，k2为主策略，k1只作诊断。

## 评测与硬门槛

训练后仅对固定 V365 Dev 完整文本、固定4492条累计前缀及 V361 long Dev 240条评测。比较 M0、M1、E2-A、E2-B、E2-C；历史模型可复用已审计预测，E2-C做三次新推理。

实验有效性要求：Dev完整通话200、累计前缀4492、long Dev 240；full与最终累计前缀无阈值翻转且最大概率差不超过1e-4；三项覆盖率均为1；checkpoint strict load无missing/unexpected key；Test/restricted读取数为0。

相对当前冠军M1，E2-C必须同时满足：完整文本early命中≥13/15、explicit=65/65、全部fraud≥78/80、normal FP≤6/80；k2 early有效检测≥14/15、全部fraud有效检测≥79/80、normal曾误报≤11/80、抢报≤57/80、clean-effective≥22/80、explicit在tA前检测≥63/65、延迟中位数≤0且P75≤1；long Dev fraud=120/120、normal FP≤2/120且相对M1无新增漏报。

若要替代M1，还须同时满足：k2 normal曾误报≤10/80、诈骗抢报≤56/80、二者事件数之和≤65、clean-effective≥23/80。

任一有效性失败记为 `INVALID`；非劣门失败记为 `FAIL`；非劣通过但改进门未全过记为 `HOLD`；全部通过才记为 `PASS_FOR_FREEZE`。即使通过，本实验也不得自动读取Test；只能冻结checkpoint、阈值0.5与k2策略，等待另行授权的一次盲评。

## 停止条件

- 输入路径、哈希、数据构成或pair因果断言不符合合同：训练前停止。
- GPU或训练进程异常退出：保留日志，不静默重跑。
- 磁盘不足以原子写入完整checkpoint：训练前停止。
- 任意Test/restricted路径被传入：立即拒绝。
