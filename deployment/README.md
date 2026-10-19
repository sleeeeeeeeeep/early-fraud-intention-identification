# 20261014 部署包源码

本目录来自 `电信诈骗检测部署包_20261014.zip`。保留网页、命令行、M1 推理源码、tokenizer、编码器配置和推理参数；**没有上传 M1 模型权重**，因此本目录不是原始可独立运行的完整部署包。

## 流程与文件

流程为音频上传 → Qwen3-ASR 转写 → 外语转中文 → M1 E1-B 二分类 → 页面展示与 CSV 导出。中文直接进入分类。原包保留的 `audio` 路线需额外音频分类模型，不属于本次主线。

| 文件 | 用途 |
| --- | --- |
| `启动服务.py` | 设置临时目录、地址和设备，并调用 text 路线 |
| `run.py` | Gradio 网页与命令行批量检测 |
| `predict.py` | ASR、翻译与分类流程 |
| `src/release_model.py` | 冻结 M1 状态加载及文本分类 |
| `src/text_model/` | 交付时的模型与分块实现副本 |
| `config/inference.json` | 320/64/96、Route B、阈值 0.5 等固定参数 |
| `models/tokenizer/`、`models/encoder_config/` | 小型加载资源 |
| `BUILD_PROVENANCE.json`、`provenance/` | 原交付来源、原包清单与校验记录 |

## 准备与启动

先从受控部署材料补入 `models/m1_e1b/model_state.pt`，按原包记录校验。训练 checkpoint 的包装与部署纯状态文件不同，不要只改名代替。

在 Python 3.10 或 3.11 独立环境中执行原部署包安装步骤：

```bash
cd deployment
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
python 启动服务.py --device cuda:0 --port 7860
```

浏览器访问 `http://127.0.0.1:7860`。CPU 使用对应 PyTorch CPU 安装源并指定 `--device cpu`。原包依赖版本按原样保留，这次未在全新部署环境重装或验证网页推理。

首次启动按原加载逻辑联网下载 `Qwen/Qwen3-ASR-1.7B-hf` 与 `Qwen/Qwen3-1.7B`，分别缓存到 `models/qwen3_asr_1_7b_hf` 和 `models/qwen3_1_7b`。它们不在 GitHub 中；联网与显存需求取决于实际部署。

## 已有验证与限制

原部署说明记录服务器验证过中文 WAV/MP3 的上传、转写、M1 分类和 CSV 下载，GPU 模型占用约 13 GB。此为原材料记录，不是本次新测量。原版 SoundFile 对 M4A 的兼容和异常回显待处理，优先 WAV/MP3；需要时先用 FFmpeg 转为单声道 16 kHz WAV。

运行后的音频缓存、检测输出和临时目录不应提交。部署网页是交付推理入口，未因此实现 E2-D 门控或完整的逐轮实时预警系统。

原完整部署包校验记录含未发布权重，直接对当前 GitHub 裁剪目录校验会报告缺文件。本仓库导出完整性使用根目录 `tools/verify_export.py`；恢复完整包时使用 `provenance/original_package_checksums.json` 中记录的原文件校验值。
