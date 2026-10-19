# 本次归档验证

日期：2026-10-19。

## 已完成

- 对发布副本的 Python 文件执行 AST 语法解析，对 JSON 执行解析检查，对 Shell 执行 `bash -n`。
- 检查新增文档的仓库相对链接。
- 检查发布文件中的机器路径、内网地址、常见令牌/私钥格式与明文凭据；发布副本未保留连接服务器的认证信息。
- 在服务器临时目录中复制发布源码，用既有 `longtext` 环境运行以下既有测试；禁用 GPU 和 pytest 缓存，没有读取业务数据或模型权重：

```text
text_model/tests/test_longdoc_chunker_core.py
text_model/tests/test_longdoc_collator_targets.py
text_model/tests/test_longdoc_losses.py
text_model/tests/test_evidence_pooling_gather_rows.py

9 passed in 2.44s
```

- 生成 `docs/export_manifest.json`，用于通过 `python3 tools/verify_export.py` 检查发布文件大小和 SHA256。

## 验证边界

本次未重新训练、未在新机器重建完整环境、未加载交付模型运行网页/ASR、未新增统计显著性检验、未读取或评测封存 V365 Test。README 指标来自既有固定 Dev 聚合报告，部署验证描述来自原部署材料。

这些检查证明本次发布副本通过了上述静态检查和 9 项核心测试，不能替代完整实验复现、授权数据上的业务验收或现网性能测试。
