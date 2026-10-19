#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""验证 _gather_rows 修复：多维索引时 batch 维不得被广播。

参照实现用纯 Python 逐行取值，不依赖任何广播语义，因此可以直接判定对错。

用法::

    python test_gather_rows_fix.py <evidence_pooling.py 的路径>
"""

import importlib.util
import itertools
import sys

import torch


def load(path):
    spec = importlib.util.spec_from_file_location("evidence_pooling_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference_gather(values, indices):
    """逐行取值：result[i, k] = values[i, indices[i, k]]。"""

    if indices.dim() == 1:
        return torch.stack([values[i, int(indices[i])] for i in range(values.shape[0])])
    return torch.stack(
        [
            torch.stack([values[i, int(j)] for j in indices[i]])
            for i in range(values.shape[0])
        ]
    )


def main():
    path = sys.argv[1]
    module = load(path)
    print("module:", path)
    print("covered:", module.__file__)
    print()

    torch.manual_seed(0)
    failures = []
    cases = 0

    for batch_size, num_chunks, dim in itertools.product((1, 2, 3, 5), (1, 2, 4, 7), (4,)):
        cases += 1
        values = torch.randn(batch_size, num_chunks, dim)
        scores = torch.randn(batch_size, num_chunks)
        topk = min(2, num_chunks)
        indices = scores.topk(k=topk, dim=-1).indices

        want = reference_gather(values, indices)
        got = module._gather_rows(values, indices)
        if got.shape != want.shape:
            failures.append(
                "B=%d C=%d: shape %s != %s" % (batch_size, num_chunks, tuple(got.shape), tuple(want.shape))
            )
            continue
        delta = float((got - want).abs().max())
        flag = "ok " if delta < 1e-6 else "FAIL"
        if delta >= 1e-6:
            failures.append("B=%d C=%d 2D: max|delta|=%.8f" % (batch_size, num_chunks, delta))
        print(
            "  %s _gather_rows 2D  B=%d C=%d k=%d  max|delta|=%.8f"
            % (flag, batch_size, num_chunks, topk, delta)
        )

        flat = indices[:, 0]
        want1 = reference_gather(values, flat)
        got1 = module._gather_rows(values, flat)
        delta1 = float((got1 - want1).abs().max())
        if delta1 >= 1e-6:
            failures.append("B=%d C=%d 1D: max|delta|=%.8f" % (batch_size, num_chunks, delta1))
        print(
            "  %s _gather_rows 1D  B=%d C=%d        max|delta|=%.8f"
            % ("ok " if delta1 < 1e-6 else "FAIL", batch_size, num_chunks, delta1)
        )

    print()
    print("=== evidence_pool 端到端（top1 / top2 对齐逐行参照） ===")
    for batch_size, num_chunks, dim in ((1, 6, 8), (2, 6, 8), (3, 4, 8), (4, 1, 8)):
        cases += 1
        values = torch.randn(batch_size, num_chunks, dim)
        chunk_mask = torch.ones(batch_size, num_chunks, dtype=torch.bool)
        risk = torch.rand(batch_size, num_chunks)
        attn = torch.softmax(torch.randn(batch_size, num_chunks), dim=-1)

        out = module.evidence_pool(values, chunk_mask, risk, attn)

        topk = min(2, num_chunks)
        idx = risk.topk(k=topk, dim=-1).indices
        want_top1 = reference_gather(values, idx[:, 0])
        want_top2 = reference_gather(values, idx).mean(dim=1)

        d1 = float((out["top1"] - want_top1).abs().max())
        d2 = float((out["top2"] - want_top2).abs().max())
        if d1 >= 1e-6:
            failures.append("evidence_pool B=%d C=%d top1: %.8f" % (batch_size, num_chunks, d1))
        if d2 >= 1e-6:
            failures.append("evidence_pool B=%d C=%d top2: %.8f" % (batch_size, num_chunks, d2))
        print(
            "  %s evidence_pool   B=%d C=%d  top1|delta|=%.8f  top2|delta|=%.8f"
            % (
                "ok " if (d1 < 1e-6 and d2 < 1e-6) else "FAIL",
                batch_size,
                num_chunks,
                d1,
                d2,
            )
        )

    # 同一 batch 内不同样本的 top2 必须互不相同（旧实现会让全批拿到同一个值）。
    print()
    print("=== 批内区分度（旧实现下全批 top2 相同） ===")
    batch_size, num_chunks, dim = 4, 6, 8
    values = torch.randn(batch_size, num_chunks, dim)
    mask = torch.ones(batch_size, num_chunks, dtype=torch.bool)
    risk = torch.rand(batch_size, num_chunks)
    attn = torch.softmax(torch.randn(batch_size, num_chunks), dim=-1)
    out = module.evidence_pool(values, mask, risk, attn)
    spread = float((out["top2"].unsqueeze(0) - out["top2"].unsqueeze(1)).abs().max())
    flag = "ok " if spread > 1e-6 else "FAIL"
    if spread <= 1e-6:
        failures.append("top2 在整个 batch 内完全相同，说明仍存在跨样本广播")
    for i in range(batch_size):
        for j in range(i + 1, batch_size):
            if torch.equal(out["top2"][i], out["top2"][j]):
                failures.append("top2[%d] 与 top2[%d] 完全相同" % (i, j))
    print("  %s 批内 top2 两两最大差 = %.8f" % (flag, spread))

    print()
    print("cases: %d" % cases)
    if failures:
        print("RESULT: FAIL (%d)" % len(failures))
        for item in failures:
            print("  -", item)
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
