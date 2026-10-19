# V361-C Route-B head96 最终对照评测

- 成功运行：`runs/r_260912_013127_V361C_RouteBHead96_vs_V341D1_TestB_LongStress_gpu23_a1`
- 评测时间：2026-09-12
- 性质：冻结 checkpoint、纯推理、只报告结果，不参与选模或阈值调整
- Route-B：V361 `b320_head96` epoch 4，320-token chunk、overlap 64、head context 96、门控注意力聚合
- Baseline：V341-D1 epoch 3，单段 256-token 截断，不使用滑动窗口

## Test-B（1580 条）

| 指标 | Baseline 256 | Route-B head96 | Route-B 差值 |
|---|---:|---:|---:|
| 13 类 Accuracy | 89.68% | 89.37% | -0.32 pp |
| 13 类 Macro-F1 | 89.91% | 89.55% | -0.36 pp |
| Fraud Recall | 97.19% | 97.05% | -0.14 pp |
| Normal FPR | 5.83% | 6.67% | +0.83 pp |
| Binary Accuracy | 96.96% | 96.77% | -0.19 pp |

Test-B 的 WordPiece 长度均可落入 Route-B 的单个 320-token chunk；该结果主要衡量 320-token 完整输入与 256-token 截断输入的外部泛化差异。Route-B 与 baseline 基本持平，baseline 小幅领先。

## Long-Stress（120 条）

| 指标 | Baseline 256 | Route-B head96 | Route-B 差值 |
|---|---:|---:|---:|
| 13 类 Accuracy | 84.17% | 96.67% | +12.50 pp |
| Observed-label Macro-F1 | 79.22% | 84.61% | +5.39 pp |
| Fraud Recall | 93.33% | 100.00% | +6.67 pp |
| Normal FPR | 16.67% | 0.00% | -16.67 pp |
| Binary Accuracy | 88.33% | 100.00% | +11.67 pp |
| Full-text coverage | 截断到 256 token | 100.00% | 完整覆盖 |

Route-B 在 Long-Stress 的 60 条诈骗与 60 条正常长文本上实现二分类全对。按风险证据位置分层，Q1/Q2/Q3/Q4 的诈骗召回均为 100%；按长度分层，medium、long、very-long 的二分类准确率也均为 100%。

## 结论

Route-B head96 在 Test-B 上付出约 0.3 pp 的轻微整体精度代价，换来 Long-Stress 上 12.5 pp 的 13 类准确率提升、6.67 pp 的诈骗召回提升和 16.67 pp 的正常误报率下降。该结果支持保留 Route-B 作为长文本路线，同时将 Test-B 的小幅回退列为后续短文本兼容性优化目标。
