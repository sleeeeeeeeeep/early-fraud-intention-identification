# V361-D Route-A 最终评测结果

- 运行：`runs/r_260912_072639_V361D_RouteA_Plain_Head64_TestB_LongStress_gpu23_a1`
- 模型：冻结的 `a320_plain`（epoch 5）与 `a320_head64`（epoch 4）
- 规则：Test-B、Long-Stress 只报告，不用于模型或阈值选择

## Test-B（1580 条，13 类齐全）

| 模型 | Acc-13 | Macro-F1 | 欺诈召回 | 正常 FPR | 二分类 Acc |
|---|---:|---:|---:|---:|---:|
| baseline-256 | **89.684%** | **89.911%** | **97.192%** | **5.833%** | **96.962%** |
| Route-B b320_head96 | 89.367% | 89.552% | 97.055% | 6.667% | 96.772% |
| Route-A a320_head64 | 89.177% | 89.393% | 96.781% | 6.667% | 96.519% |
| Route-A a320_plain | 89.114% | 89.305% | 96.575% | 6.667% | 96.329% |

## Long-Stress（120 条）

Long-Stress 缺少标签 3、5，横向比较采用 observed Macro-F1。

| 模型 | Acc-13 | observed Macro-F1 | 欺诈召回 | 正常 FPR | 二分类 Acc |
|---|---:|---:|---:|---:|---:|
| Route-A a320_plain | **97.500%** | **85.265%** | **100.000%** | **0.000%** | **100.000%** |
| Route-B b320_head96 | 96.667% | 84.612% | 100.000% | 0.000% | 100.000% |
| baseline-256 | 84.167% | 79.223% | 93.333% | 16.667% | 88.333% |
| Route-A a320_head64 | 93.333% | 67.706% | 100.000% | 0.000% | 100.000% |

## 结论

- Test-B：baseline-256 最优；Route-A head64 比 plain 高 0.088 个 Macro-F1 百分点，二者均低于 Route-B。
- Long-Stress：a320_plain 最优，比 Route-B 高 0.653 个 observed Macro-F1 百分点；head64 的长文本细分类退化明显。
- 当前结果支持保留 baseline-256 作为常规短文本主线，并把 a320_plain 作为长文本路线候选。
