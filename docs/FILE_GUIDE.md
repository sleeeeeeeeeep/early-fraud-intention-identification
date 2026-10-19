# 文件导读

## 主线实现

| 文件或目录 | 作用 | 阅读时注意 |
| --- | --- | --- |
| `text_model/data/longdoc_chunker.py` | 角色标记清理、WordPiece 切块、头部上下文、全文覆盖 | chunk 上限仅诊断，尾部不能静默截断 |
| `text_model/data/longdoc_collator.py` | 组织批次、标签和分块输入 | 训练与推理应使用一致输入契约 |
| `text_model/models/shared_chunk_encoder.py` | 所有块共享 MegatronBERT 编码器 | 基础模型需另行准备 |
| `text_model/models/gated_attention_aggregator.py` | Route B 门控注意力聚合 | 当前 M0/M1 主线 |
| `text_model/models/evidence_pooling.py` | attention / top1 / top2 / last / mean 证据汇聚 | last 分支保留尾部信息 |
| `text_model/models/longdoc_multitask_classifier.py` | 多路融合与分类头 | 本项目取独立二分类头分数 |
| `text_model/models/hierarchical_transformer_aggregator.py` | Route A 历史对比 | 不代表当前交付路线 |
| `text_model/models/fixed_logit_aggregator.py` | 固定滑窗聚合对照 | 不与 Route B 结果混用 |
| `text_model/training/` | 损失、采样与指标 | 保留既有训练实现 |
| `text_model/inference/` | 长文本训练与离线推理入口 | 部分历史辅助输出仍有 13 分类字段 |

## 实验入口

- `text_model/scripts/data/`：V361 长文本构建、V362 早期前缀、E1-B 混合训练集、E2-A/B/C 混合与边界样本。
- `text_model/scripts/training/`：E1-B 和 E2-A/B/C 二分类训练。
- `text_model/scripts/evaluation/`：固定 Dev 构造和 M0/M1/E2 比较。
- `text_model/scripts/e2d/`：冻结 M1 推理、10 维因果特征、193 参数门控训练和验收。
- `text_model/experiments/`：E2-B/C/D 预登记。路径经过发布适配，原始字节哈希见导出映射。
- `text_model/scripts/cascade/`、`scripts/inference/`、部分 V361 配置：历史对照及兼容依赖，不能据此推断当前任务仍是级联或 13 分类。
- `text_model/tests/`：已有单元测试代码，和封存的业务 Test 数据集不是一回事。

## 数据来源代码

`data/external/` 仅保留生成、独立检查、重建和终审脚本，`data/processed/` 仅保留少量统计与构建辅助脚本。真实或合成的对话正文、逐条标注和封存答案不在本仓库。目录名称保留用于理解既有脚本引用，不能因为目录存在就认为数据齐全。

## 部署与实验分开阅读

`deployment/` 来自独立的 20261014 部署包，其 `src/text_model/` 是交付时冻结的推理副本，不要用训练目录同名文件随意替换。`predict.py` 负责 ASR、翻译和检测流程，`src/release_model.py` 负责交付权重加载，`run.py` 是网页和命令行入口。两套源码同时保留用于说明训练与交付关系。
