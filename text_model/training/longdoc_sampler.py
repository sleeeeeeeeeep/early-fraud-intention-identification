"""固定长文本曝光比例的确定性 sampler。"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any, Iterator

from torch.utils.data import Sampler


def is_long_row(row: Mapping[str, Any]) -> bool:
    return bool(
        row.get("longdoc_generated")
        or row.get("is_longdoc")
        or str(row.get("split") or "").lower().startswith("long")
    )


class LongExposureSampler(Sampler[int]):
    """每个 epoch 覆盖全部短样本，并重采样长样本到指定曝光比例。"""

    def __init__(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        target_fraction: float = 0.21,
        seed: int = 202,
    ) -> None:
        if not 0.0 < target_fraction < 1.0:
            raise ValueError("target_fraction must be between zero and one")
        self.short_indices = [i for i, row in enumerate(rows) if not is_long_row(row)]
        self.long_indices = [i for i, row in enumerate(rows) if is_long_row(row)]
        if not self.short_indices or not self.long_indices:
            raise ValueError("long exposure sampler requires both short and long rows")
        self.target_fraction = float(target_fraction)
        self.seed = int(seed)
        self.epoch = 0
        self.long_draws = int(
            round(
                len(self.short_indices)
                * self.target_fraction
                / (1.0 - self.target_fraction)
            )
        )

    @property
    def effective_fraction(self) -> float:
        return self.long_draws / (len(self.short_indices) + self.long_draws)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.short_indices) + self.long_draws

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + 1009 * self.epoch)
        short = list(self.short_indices)
        rng.shuffle(short)
        long_draws: list[int] = []
        while len(long_draws) < self.long_draws:
            cycle = list(self.long_indices)
            rng.shuffle(cycle)
            long_draws.extend(cycle)
        schedule = short + long_draws[: self.long_draws]
        rng.shuffle(schedule)
        return iter(schedule)
