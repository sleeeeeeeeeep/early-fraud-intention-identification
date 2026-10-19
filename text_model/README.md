# 文本反诈核心代码

当前任务：正常/诈骗二分类，重点是长文本覆盖及诈骗前期证据识别。

## 阅读顺序

1. `data/longdoc_chunker.py`、`data/longdoc_collator.py`：全文覆盖，320 token块、64重叠和最多96头部上下文。
2. `models/shared_chunk_encoder.py`：共享MegatronBERT编码与位置特征。
3. `models/gated_attention_aggregator.py`、`models/evidence_pooling.py`：Route B五路证据汇聚。
4. `models/longdoc_multitask_classifier.py`：文档融合与独立二分类头。
5. `inference/run_longdoc_inference.py`：离线推理；不要使用13类联合概率代替 `p_binary[1]`。
6. `scripts/data/`、`scripts/training/`：训练数据构造与二分类训练。
7. `scripts/evaluation/`：Dev评测；`scripts/e2d/`：时序门控。
8. `tests/`：单元测试代码，不是封存Test数据集。

根目录 `README.md`、`reports/实验结果汇总.md` 和 `models/model_registry.json` 是当前版本入口。旧RoBERTa、早期13分类或级联推荐不再作为本工程的当前状态说明。
