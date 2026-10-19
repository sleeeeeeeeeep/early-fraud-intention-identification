"""对话角色标识契约。

角色标识是为了后续音频合成保留的元数据，不是电话内容本身。
因此所有模型输入、WordPiece 长度和窗口位置统计都必须使用去标识文本；
``text``/``audio_text`` 则保留原始的 ``来电人``、``接电人`` 等标识。
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


# 兼容现有数据中的“来电方/接听方”，以及用户指定的“来电人/接电人”。
# 只匹配带中文冒号的角色前缀，避免删除正文中普通的“客服”“客户”等词。
ROLE_MARKER_RE = re.compile(
    r"(?m)^[ \t]*(?P<marker>"
    r"来电方|来电人|接听方|接听人|接电人|接收方|"
    r"客服|客户|用户|对方|诈骗分子|受害人|"
    r"说话人[一二三四五六七八九十]?|"
    r"caller|callee|speaker[_ ]?[0-9]+|"
    r"甲|乙|A|B"
    r")\s*[:：]",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RoleMarker:
    marker: str
    canonical_role: str
    char_start: int
    char_end: int


def canonical_role(marker: str) -> str:
    value = marker.strip().lower()
    if value in {"来电方", "来电人", "caller", "甲", "a"}:
        return "caller"
    if value in {"接听方", "接听人", "接电人", "接收方", "callee", "乙", "b"}:
        return "callee"
    if value in {"客服", "工作人员"}:
        return "service"
    if value in {"客户", "用户", "受害人"}:
        return "customer"
    if value == "对方":
        return "other"
    if value == "诈骗分子":
        return "caller"
    return "speaker"


def find_role_markers(text: str) -> list[RoleMarker]:
    """返回原文中所有带冒号的角色标识，位置以原文字符坐标表示。"""

    value = str(text or "")
    return [
        RoleMarker(
            marker=match.group("marker"),
            canonical_role=canonical_role(match.group("marker")),
            char_start=match.start(),
            char_end=match.end(),
        )
        for match in ROLE_MARKER_RE.finditer(value)
    ]


def strip_role_markers(text: str) -> str:
    """去掉角色名和冒号，但保留正文、换行和其他标点。"""

    return ROLE_MARKER_RE.sub("", str(text or ""))


def role_turns(text: str) -> list[dict[str, Any]]:
    """把带角色标识的逐行对话解析成音频侧可用的轮次元数据。"""

    value = str(text or "")
    matches = list(ROLE_MARKER_RE.finditer(value))
    if not matches:
        return []
    turns: list[dict[str, Any]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(value)
        content = value[match.end() : end].strip()
        turns.append(
            {
                "turn_index": index,
                "marker": match.group("marker"),
                "canonical_role": canonical_role(match.group("marker")),
                "text": content,
            }
        )
    return turns


def role_contract(text: str) -> dict[str, Any]:
    """生成写入 JSONL 的角色字段，明确 token 统计使用去标识文本。"""

    marked = str(text or "")
    markers = find_role_markers(marked)
    return {
        "audio_text": marked,
        "text_with_role_markers": marked,
        "model_input_text": strip_role_markers(marked),
        "role_markers_preserved": bool(markers),
        "role_marker_token_policy": "excluded_from_model_input_and_wordpiece_counts",
        "role_marker_count": len(markers),
        "role_turns": role_turns(marked),
    }
