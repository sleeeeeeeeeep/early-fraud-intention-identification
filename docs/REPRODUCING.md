# 运行与复现

## 1. 校验仓库，不加载模型

```bash
python3 tools/verify_export.py
```

该工具只核对导出清单中的文件哈希，不读取数据集、不加载权重，也不触发网络请求。

## 2. 训练环境与部署环境分开

归档时服务器 `longtext` 环境：Python 3.8.20、torch 2.0.0+cu117、transformers 4.46.3、numpy 1.23.5、scipy 1.10.1、scikit-learn 1.3.2、pytest 8.3.2、tokenizers 0.20.3、huggingface-hub 0.36.2；该环境未安装 accelerate。主要版本另存于 `requirements-training.txt`，这不是完整的环境锁文件。

已有环境复现优先使用独立 `longtext` 环境。另建环境时，PyTorch 需按实际设备和 CUDA 兼容性选择；不要把部署包的 torch 2.7.1 / transformers 5.19.0 依赖装进历史训练环境。此次未在新机器验证完整训练，也未升级服务器依赖。

部署环境按 [deployment/README.md](../deployment/README.md) 单独创建，原部署说明使用 Python 3.10 或 3.11。

## 3. 运行已有单元测试

以下测试无需模型权重和业务数据，但 torch 相关测试需要训练依赖：

```bash
PYTHONPATH=.:text_model python -m pytest -q \
  text_model/tests/test_longdoc_chunker_core.py \
  text_model/tests/test_longdoc_collator_targets.py \
  text_model/tests/test_longdoc_losses.py \
  text_model/tests/test_evidence_pooling_gather_rows.py
```

其余 suite 测试可能依赖历史资源或脚本契约，不能用上述测试替代完整训练与部署验收。本次实际验证范围记录在 [VALIDATION.md](VALIDATION.md)。

## 4. 补齐资源后运行训练

从仓库根目录设置资源，路径替换为自己的授权材料位置：

```bash
export TELEFRAUD_ROOT="$PWD"
export TELEFRAUD_PYTHON=python
export BASE_MODEL=/path/to/authorized/megatronbert
export V341_CHECKPOINT=/path/to/authorized/v341/best_model/pytorch_model.bin
```

`V341_CHECKPOINT` 所在目录还要有对应 tokenizer。M0 checkpoint 和训练/Dev 输入须按 `docs/RESOURCES.md` 放置。`text_model/scripts/run_v362_e1b_binary.sh` 是 M1 的完整归档启动器，会先构建数据再训练；其默认输出已存在时主动拒绝覆盖。各 E2 启动器同样带有固定历史目录名和 GPU 编号，运行前应检查参数并为新实验选择独立输出。

Python 数据构建/训练入口提供 `--help`；默认参数适用于既有目录契约。仅有 GitHub 源码时，缺数据、缺检查点或 tokenizer 是预期的资源缺口，不应通过改用其他版本数据来假装复现。

## 5. 离线推理示例

补齐授权资源和含 `sample_id`、`text`、`label_13` 的 Train/Dev 格式输入后：

```bash
PYTHONPATH=.:text_model python text_model/inference/run_longdoc_inference.py \
  --input-jsonl /path/to/authorized/dev.jsonl \
  --model "$BASE_MODEL" \
  --v341-checkpoint "$V341_CHECKPOINT" \
  --longdoc-checkpoint runs/v362_e1b_binary_60prefix_40replay_s202_conservative/best_model.pt \
  --route B --input-mode head_context_enhanced_chunks \
  --chunk-length 320 --overlap 64 --head-context-length 96 \
  --output-dir outputs/dev_m1
```

该入口兼有评测功能；自定义无标签文本不应伪造标签来解释指标，可参考部署 `src/release_model.py` 的纯分类接口。V365 实时报警还需固定累计前缀、k2 与 tE/tA 评测流程，不能以全文指标替代。
