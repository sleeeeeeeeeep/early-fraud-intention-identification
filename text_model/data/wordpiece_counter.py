"""与 BERT vocab 对齐的轻量 WordPiece 长度审计器。"""

from __future__ import annotations

from pathlib import Path
import unicodedata

from .role_markers import strip_role_markers


class BertWordPieceCounter:
    def __init__(self, vocab_path: str | Path) -> None:
        path = Path(vocab_path)
        self.vocab = {line.rstrip("\n") for line in path.open(encoding="utf-8")}
        if "[UNK]" not in self.vocab:
            raise ValueError(f"invalid BERT vocab: {path}")

    @staticmethod
    def is_chinese(char: str) -> bool:
        codepoint = ord(char)
        return (
            0x4E00 <= codepoint <= 0x9FFF
            or 0x3400 <= codepoint <= 0x4DBF
            or 0x20000 <= codepoint <= 0x2A6DF
            or 0x2A700 <= codepoint <= 0x2B73F
            or 0x2B740 <= codepoint <= 0x2B81F
            or 0x2B820 <= codepoint <= 0x2CEAF
            or 0xF900 <= codepoint <= 0xFAFF
        )

    def tokens(self, text: str) -> list[str]:
        basic: list[str] = []
        for char in strip_role_markers(str(text or "")):
            if char.isspace():
                basic.append(" ")
            elif self.is_chinese(char) or unicodedata.category(char).startswith("P"):
                basic.extend((" ", char, " "))
            else:
                basic.append(char.lower())
        pieces: list[str] = []
        for token in "".join(basic).split():
            if token in self.vocab:
                pieces.append(token)
                continue
            start = 0
            local: list[str] = []
            while start < len(token):
                end = len(token)
                found = None
                while start < end:
                    part = token[start:end]
                    candidate = part if start == 0 else "##" + part
                    if candidate in self.vocab:
                        found = candidate
                        break
                    end -= 1
                if found is None:
                    local = ["[UNK]"]
                    break
                local.append(found)
                start = end
            pieces.extend(local)
        return pieces

    def count(self, text: str) -> int:
        return len(self.tokens(text))

    def char_span_to_token_span(
        self, text: str, char_start: int, char_end: int
    ) -> tuple[int, int]:
        if not 0 <= char_start < char_end <= len(text):
            raise ValueError("invalid character span")
        start = self.count(text[:char_start])
        end = start + self.count(text[char_start:char_end])
        return start, end
