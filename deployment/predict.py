#!/usr/bin/env python3
"""甲方测试项目：最佳文本路线与纯音频路线统一推理入口。"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Sequence

# Apple Silicon上遇到尚未实现的MPS算子时允许回退CPU。
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


LABEL_NAMES = {
    0: "非涉诈场景",
    1: "冒充电商物流客服类",
    2: "冒充公检法及政府机关类",
    3: "刷单返利类",
    4: "贷款、代办信用卡类",
    5: "冒充军警购物诈骗",
    6: "杀猪盘",
    7: "虚假购物、服务类",
    8: "网络婚恋、交友类",
    9: "虚假征信类",
    10: "冒充领导、熟人类",
    11: "网络游戏产品虚假交易类",
    12: "其他类型诈骗",
}
IS_FRAUD = {label: label != 0 for label in LABEL_NAMES}

TASK_INSTRUCTION = (
    "请综合所有音频片段，只依据通话内容判断类型。"
    "父类编号为0到12，其中0表示正常通话，1到12表示十二类诈骗。"
    "不要转写音频；请在内部形成完整通话语义表示后完成分类。"
)


def _normalize(text: str) -> str:
    text = re.sub(r"<\|[^|]+\|>", "", str(text or ""))
    return " ".join(text.split())


def _select_device(value: str) -> str:
    if value != "auto":
        return value
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _snapshot_ready(model_dir: Path) -> bool:
    """检查本地HF快照的全部权重分片是否存在。"""
    if not (model_dir / "config.json").is_file():
        return False
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            shards = set(index.get("weight_map", {}).values())
            return bool(shards) and all(
                (model_dir / shard).is_file() and (model_dir / shard).stat().st_size > 0
                for shard in shards
            )
        except (OSError, ValueError, TypeError):
            return False
    single = model_dir / "model.safetensors"
    return single.is_file() and single.stat().st_size > 0


def _ensure_snapshot(model_dir: Path, repo_id: str) -> None:
    if _snapshot_ready(model_dir):
        return
    from huggingface_hub import snapshot_download

    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"未检测到完整的{repo_id}，现在自动下载/续传到：{model_dir}")
    snapshot_download(repo_id=repo_id, local_dir=str(model_dir), max_workers=8)
    if not _snapshot_ready(model_dir):
        raise FileNotFoundError(f"{repo_id}下载未完成：{model_dir}")


class TextM1E1B:
    """保留原版转写与翻译流程，接入 M1 E1B 长文本二分类模型。"""

    def __init__(
        self,
        asr_dir: Path,
        translator_dir: Path,
        model_dir: Path,
        device: str,
    ) -> None:
        import torch
        import sys

        self.torch = torch
        self.device = device
        self.asr_dir = asr_dir
        self.translator_dir = translator_dir
        self.translator = None
        self.translator_tokenizer = None
        _ensure_snapshot(asr_dir, "Qwen/Qwen3-ASR-1.7B-hf")
        _ensure_snapshot(translator_dir, "Qwen/Qwen3-1.7B")
        root = Path(__file__).resolve().parent
        sys.path.insert(0, str(root / "src"))
        from release_model import FraudDetector

        self.classifier = FraudDetector(device=device, package_root=model_dir.parent.parent)
        self.model = self.classifier.model
        self._load_asr()
        # 文本路线启动时一次性加载完整流水线，避免首条外语音频临时加载。
        self._load_translator()

    def _load_asr(self) -> None:
        dtype = (
            self.torch.bfloat16
            if self.device.startswith("cuda") or self.device == "mps"
            else self.torch.float32
        )
        device_map = "cuda:0" if self.device == "cuda" else self.device
        # 默认使用Transformers原生-hf格式，同时保留旧qwen-asr格式回退。
        if (self.asr_dir / "processor_config.json").is_file():
            from transformers import AutoModelForMultimodalLM, AutoProcessor

            self.asr_backend = "transformers"
            self.asr_processor = AutoProcessor.from_pretrained(
                str(self.asr_dir), local_files_only=True
            )
            self.asr = AutoModelForMultimodalLM.from_pretrained(
                str(self.asr_dir),
                local_files_only=True,
                dtype=dtype,
                device_map=device_map,
            ).eval()
        else:
            try:
                from qwen_asr import Qwen3ASRModel
            except ImportError as exc:
                raise RuntimeError(
                    "当前Qwen3-ASR权重为官方qwen-asr格式，请先安装：pip install -U qwen-asr"
                ) from exc
            self.asr_backend = "qwen_asr"
            self.asr = Qwen3ASRModel.from_pretrained(
                str(self.asr_dir),
                dtype=dtype,
                device_map=device_map,
                max_inference_batch_size=1,
                max_new_tokens=1024,
            )

    def _read_audio(self, path: str):
        """直接读取音频波形，避免Transformers文件加载隐式依赖librosa。"""
        import numpy as np
        import soundfile as sf
        from scipy.signal import resample_poly

        waveform, source_rate = sf.read(path, dtype="float32", always_2d=False)
        if getattr(waveform, "ndim", 1) > 1:
            waveform = waveform.mean(axis=1)
        target_rate = int(self.asr_processor.feature_extractor.sampling_rate)
        if int(source_rate) != target_rate:
            divisor = math.gcd(int(source_rate), target_rate)
            waveform = resample_poly(
                waveform,
                target_rate // divisor,
                int(source_rate) // divisor,
            )
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.size == 0:
            raise ValueError(f"空音频：{path}")
        return waveform, target_rate

    def transcribe(self, audio_path: str) -> tuple[str, str]:
        if self.asr_backend == "qwen_asr":
            result = self.asr.transcribe(audio=str(audio_path), language=None)[0]
            return _normalize(result.text), str(result.language or "Unknown")

        waveform, sample_rate = self._read_audio(str(audio_path))
        inputs = self.asr_processor.apply_transcription_request(
            audio=waveform,
            processor_kwargs={"sampling_rate": sample_rate},
        ).to(self.asr.device, self.asr.dtype)
        with self.torch.inference_mode():
            output_ids = self.asr.generate(**inputs, max_new_tokens=1024, do_sample=False)
        generated = output_ids[:, inputs["input_ids"].shape[1] :]
        parsed = self.asr_processor.decode(
            generated, return_format="parsed"
        )[0]
        return _normalize(parsed.get("transcription", "")), str(
            parsed.get("language") or "Unknown"
        )

    def _load_translator(self) -> None:
        if self.translator is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = (
            self.torch.bfloat16
            if self.device.startswith("cuda") or self.device == "mps"
            else self.torch.float32
        )
        self.translator_tokenizer = AutoTokenizer.from_pretrained(
            str(self.translator_dir), local_files_only=True
        )
        self.translator = AutoModelForCausalLM.from_pretrained(
            str(self.translator_dir),
            local_files_only=True,
            dtype=dtype,
            device_map="cuda:0" if self.device == "cuda" else self.device,
        ).eval()

    def translate_to_chinese(self, text: str, language: str) -> str:
        if language.strip().lower() in {"chinese", "zh", "mandarin"}:
            return text
        self._load_translator()
        messages = [
            {
                "role": "system",
                "content": (
                    "你是忠实通话翻译器。只输出简体中文译文，不解释、不总结、"
                    "不改写。必须完整保留金额、账号、验证码、链接、人名、"
                    "机构名、否定词和转账操作。"
                ),
            },
            {
                "role": "user",
                "content": f"将以下{language}通话逐句翻译成简体中文：\n{text}",
            },
        ]
        prompt = self.translator_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        encoded = self.translator_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.translator.device)
        with self.torch.inference_mode():
            output = self.translator.generate(
                **encoded,
                max_new_tokens=1024,
                do_sample=False,
                use_cache=True,
            )
        translated = self.translator_tokenizer.decode(
            output[0, encoded["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
        translated = re.sub(r"<think>.*?</think>", "", translated, flags=re.S)
        translated = _normalize(translated)
        if not translated:
            raise RuntimeError(f"{language}转中文失败")
        return translated

    def classify(self, texts: Sequence[str]) -> list[dict]:
        results = []
        for i, text in enumerate(texts):
            row = self.classifier.predict_text(text, sample_id=f"audio-{i}")
            label = int(row["prediction_binary"])
            probabilities = [row["p_normal"], row["p_fraud"]]
            results.append({
                "label_id": label,
                "confidence": float(probabilities[label]),
                "probabilities": probabilities,
            })
        return results

    def predict(self, audio_path: str) -> dict:
        source_transcript, detected_language = self.transcribe(audio_path)
        if not source_transcript:
            return {
                "audio_path": str(audio_path),
                "transcript": "",
                "source_transcript": "",
                "detected_language": detected_language,
                "label_id": -1,
                "label_name": "转写失败",
                "is_fraud": None,
                "route": "text_m1_e1b",
            }
        chinese_transcript = self.translate_to_chinese(
            source_transcript, detected_language
        )
        result = self.classify([chinese_transcript])[0]
        label = result["label_id"]
        return {
            "audio_path": str(audio_path),
            "transcript": chinese_transcript,
            "source_transcript": source_transcript,
            "detected_language": detected_language,
            "translated": chinese_transcript != source_transcript,
            "label_id": label,
            "label_name": {0: "非涉诈场景", 1: "涉诈场景"}[label],
            "is_fraud": bool(label),
            "confidence": result["confidence"],
            "probabilities": result["probabilities"],
            "route": "text_m1_e1b",
        }


class AudioV334C:
    """V334C Qwen2.5-Omni-7B纯音频分类器，不读取ASR文本。"""

    def __init__(self, base_dir: Path, checkpoint_dir: Path, device: str) -> None:
        import torch
        import torch.nn as nn
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            BitsAndBytesConfig,
            Qwen2_5OmniProcessor,
            Qwen2_5OmniThinkerForConditionalGeneration,
        )

        is_cuda = device.startswith("cuda") and torch.cuda.is_available()
        is_mps = device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if not (is_cuda or is_mps):
            raise RuntimeError(
                "V334C纯音频路线需要NVIDIA CUDA GPU，或统一内存>=32GB的Apple Silicon Mac"
            )
        if is_cuda and ":" in device:
            torch.cuda.set_device(torch.device(device))
        adapter_dir = checkpoint_dir / "adapter"
        heads_path = checkpoint_dir / "best_heads.pt"
        if not (adapter_dir / "adapter_model.safetensors").is_file():
            raise FileNotFoundError(f"缺少音频LoRA权重：{adapter_dir}")
        if not heads_path.is_file():
            raise FileNotFoundError(f"缺少音频分类头：{heads_path}")
        _ensure_snapshot(base_dir, "Qwen/Qwen2.5-Omni-7B")

        self.torch = torch
        self.nn = nn
        self.device = torch.device(device)
        self.is_cuda = is_cuda
        # Qwen2.5-Omni checkpoints are stored in BF16. Apple Silicon MPS
        # supports BF16, so preserve the checkpoint dtype and avoid an
        # expensive per-tensor BF16 -> FP16 conversion during startup.
        self.dtype = torch.bfloat16 if (is_cuda or is_mps) else torch.float32
        self.processor = Qwen2_5OmniProcessor.from_pretrained(
            base_dir, local_files_only=True, trust_remote_code=True
        )
        load_kwargs = {
            "local_files_only": True,
            "trust_remote_code": True,
            "dtype": self.dtype,
            "low_cpu_mem_usage": True,
            "attn_implementation": "sdpa",
        }
        if is_cuda:
            load_kwargs["device_map"] = {"": torch.cuda.current_device()}
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.dtype,
                bnb_4bit_use_double_quant=True,
            )
        else:
            # bitsandbytes不支持Apple MPS；Apple Silicon直接保留BF16加载。
            load_kwargs["device_map"] = {"": "mps"}
        raw = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            base_dir, **load_kwargs
        )
        if is_cuda:
            raw = prepare_model_for_kbit_training(
                raw,
                use_gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
        self.backbone = get_peft_model(
            raw,
            LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            ),
        )
        self.backbone.load_adapter(
            str(adapter_dir),
            adapter_name="v334_best_epoch2",
            is_trainable=False,
        )
        self.backbone.set_adapter("v334_best_epoch2")
        base = self.backbone.get_base_model()
        base.config.use_cache = False
        base.model.config.use_cache = False
        hidden_size = int(base.config.text_config.hidden_size)
        base.lm_head = nn.Identity()

        class Heads(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm = nn.LayerNorm(hidden_size)
                self.dropout = nn.Dropout(0.1)
                self.parent = nn.Linear(hidden_size, 13)
                self.child = nn.Linear(hidden_size, 39)
                self.binary = nn.Linear(hidden_size, 2)

            def forward(self, hidden):
                hidden = self.dropout(self.norm(hidden))
                return self.parent(hidden), self.child(hidden), self.binary(hidden)

        self.heads = Heads().to(self.device, dtype=self.dtype)
        try:
            state = torch.load(heads_path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(heads_path, map_location="cpu")
        self.heads.load_state_dict(state, strict=True)
        self.heads.eval()
        self.backbone.eval()
        self.sample_rate = int(self.processor.feature_extractor.sampling_rate)

    def _read_audio(self, path: str):
        import numpy as np
        from scipy.signal import resample_poly
        try:
            import soundfile as sf

            waveform, source_rate = sf.read(
                path, dtype="float32", always_2d=False
            )
        except Exception:
            import torchaudio

            tensor, source_rate = torchaudio.load(path)
            waveform = tensor.mean(dim=0).numpy()
        if getattr(waveform, "ndim", 1) > 1:
            waveform = waveform.mean(axis=1)
        if int(source_rate) != self.sample_rate:
            divisor = math.gcd(int(source_rate), self.sample_rate)
            waveform = resample_poly(
                waveform,
                self.sample_rate // divisor,
                int(source_rate) // divisor,
            )
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.size == 0:
            raise ValueError(f"空音频：{path}")
        return waveform

    def _chunks(self, waveform):
        import numpy as np

        chunk_samples = self.sample_rate * 30
        if waveform.size <= chunk_samples:
            return [waveform]
        max_start = waveform.size - chunk_samples
        starts = np.linspace(0, max_start, num=2, dtype=np.int64).tolist()
        return [waveform[int(start) : int(start) + chunk_samples] for start in starts]

    def _prepare(self, chunks):
        conversation = [
            {
                "role": "user",
                "content": [
                    *[
                        {"type": "audio", "audio": f"memory://chunk_{i}.wav"}
                        for i in range(len(chunks))
                    ],
                    {"type": "text", "text": TASK_INSTRUCTION},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        kwargs = {
            "text": prompt,
            "return_tensors": "pt",
            "padding": True,
            "sampling_rate": self.sample_rate,
            "use_audio_in_video": False,
        }
        try:
            processed = self.processor(audio=list(chunks), **kwargs)
        except TypeError:
            processed = self.processor(audios=list(chunks), **kwargs)
        tensors = {}
        for key, value in processed.items():
            if not isinstance(value, self.torch.Tensor):
                continue
            value = value.to(self.device)
            if value.is_floating_point():
                value = value.to(self.dtype)
            tensors[key] = value
        return tensors

    def predict(self, audio_path: str) -> dict:
        waveform = self._read_audio(str(audio_path))
        processed = self._prepare(self._chunks(waveform))
        autocast = (
            self.torch.autocast(device_type="cuda", dtype=self.dtype)
            if self.is_cuda
            else nullcontext()
        )
        with self.torch.inference_mode(), autocast:
            outputs = self.backbone(
                **processed,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden = outputs.hidden_states[-1]
            last_indices = processed["attention_mask"].long().sum(-1) - 1
            semantic = hidden[
                self.torch.arange(hidden.shape[0], device=self.device), last_indices
            ]
            parent_logits, _, binary_logits = self.heads(semantic)
            probabilities = self.torch.softmax(parent_logits.float(), dim=-1)[0].cpu()
            binary_probabilities = self.torch.softmax(binary_logits.float(), dim=-1)[0].cpu()
        label = int(probabilities.argmax())
        return {
            "audio_path": str(audio_path),
            "transcript": "纯音频路线：未使用ASR或文本转写",
            "label_id": label,
            "label_name": LABEL_NAMES[label],
            "is_fraud": IS_FRAUD[label],
            "confidence": float(probabilities[label]),
            "probabilities": [float(value) for value in probabilities],
            "binary_fraud_probability": float(binary_probabilities[1]),
            "route": "audio_v334c",
        }


class TeleFraudDetector:
    """保持原项目调用接口；通过route选择最佳文本或纯音频路线。"""

    def __init__(
        self,
        asr_model_path: str = "models/qwen3_asr_1_7b_hf",
        translator_model_path: str = "models/qwen3_1_7b",
        cls_model_path: str = "models/m1_e1b",
        audio_base_path: str = "models/qwen25_omni_7b",
        audio_checkpoint_path: str = "models/audio_v334c",
        device: str = "auto",
        language: str = "auto",
        route: str = "text",
    ) -> None:
        del language
        self.device = _select_device(device)
        self.route = route.strip().lower()
        root = Path(__file__).resolve().parent
        if self.route == "text":
            self.model = TextM1E1B(
                root / asr_model_path,
                root / translator_model_path,
                root / cls_model_path,
                self.device,
            )
        elif self.route == "audio":
            self.model = AudioV334C(
                root / audio_base_path,
                root / audio_checkpoint_path,
                self.device,
            )
        else:
            raise ValueError("route只允许text或audio")

    def predict(self, audio_path: str) -> dict:
        result = self.model.predict(str(audio_path))
        # Apple Silicon使用统一内存；批量长音频时PyTorch会缓存每轮ASR、
        # 翻译和Omni前向的临时张量。逐条归还缓存，防止数百条后内存累积。
        if self.device == "mps":
            import gc

            gc.collect()
            self.model.torch.mps.empty_cache()
        elif self.device.startswith("cuda"):
            self.model.torch.cuda.empty_cache()
        return result

    def predict_batch(self, audio_paths: list[str], batch_size: int = 16) -> list[dict]:
        del batch_size
        return [self.predict(path) for path in audio_paths]
