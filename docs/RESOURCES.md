# 资源清单与发布范围

## 原始材料

| 材料 | 大小 | GitHub 对应内容 |
| --- | ---: | --- |
| `telefraud-text-clean/` 训练工程 | 约 32.6 GB | 模型实现、训练/评测/构建代码、配置、预登记和汇总报告 |
| `电信诈骗检测部署包_20261014.zip` | 4,659,353,789 字节 | `deployment/` 中的源码、tokenizer 和小型配置；不含权重 |

这里的 GB 使用十进制。更精确的目录统计见 [source_inventory_summary.json](source_inventory_summary.json)。服务器材料保持原样，GitHub 目录是经过筛选和路径适配的导出版本。

## 需要另行取得的模型

| 模型 | 训练目录相对位置 | 用途 |
| --- | --- | --- |
| 基础 MegatronBERT 1.3B | `models/IDEA-CCNL-Erlangshen-MegatronBert-1.3B/` | 编码器配置、tokenizer、基础权重 |
| V341 检查点与同目录 tokenizer | `models/v341/best_model/pytorch_model.bin`（GitHub 适配位置） | 旧初始化与 tokenizer 来源 |
| M0 | `runs/repro_v361b_b320_head96/best_model.pt` | 长文本基线、M1/E2 初始化 |
| M1 | `runs/v362_e1b_binary_60prefix_40replay_s202_conservative/best_model.pt` | 当前交付模型 |
| E2-A/B/C | 见 `models/model_registry.json` | 候选对比 |
| E2-D 门控 | `runs/v365_e2d_temporal_gate_s202/temporal_gate.pt` | 仅与冻结 M1 配合使用 |
| 部署导出 M1 | `deployment/models/m1_e1b/model_state.pt` | 部署加载的纯状态权重；与训练 checkpoint 包装不同 |

M1 原训练检查点的已有 SHA256：`0d4a6fdd08b3a371d1d29108a00fa5b683986a819532684f4c1965a8eb95e1ad`。部署导出文件请按原部署包校验记录检查，不要把训练 checkpoint 的哈希误用到另一种包装的状态文件。

2026-10-19 检查发现原训练目录的基础模型软链接目标已不可访问；服务器另一个既有基础模型目录可访问。运行前需由资源持有人设置 `BASE_MODEL` 或修复资源映射。本次未修改服务器链接。

## 数据与产物

未上传：对话 JSONL/CSV、逐条标注、独立复核/裁定明细、受限 Test 答案、公司隐私评测、逐条预测、训练特征、中间归档、权重、音频、缓存与结果导出。数据集名称、规模、构造逻辑和少量聚合统计留在仓库用于理解实验。

完整复现至少需要 V361 训练与 Long-Dev、V362 前缀标注/构建输入、对应版本的 V365 Train/Dev，以及各实验清单要求的冻结输入。源码中的默认相对路径是原目录契约，存在路径不意味着数据已发布。Test 不属于常规复现前提。

`deployment/provenance/` 保存原部署材料的历史清单；原 manifest 引用了部分未随最终 zip 提供的历史文档。该 manifest 是来源记录，不是本次 GitHub 导出文件清单；本次清单为 `docs/export_manifest.json`。
