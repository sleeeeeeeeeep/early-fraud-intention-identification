"""Causal features and frozen contract for the independent E2-D gate."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
import torch
from torch import nn

M1_SHA256 = '0d4a6fdd08b3a371d1d29108a00fa5b683986a819532684f4c1965a8eb95e1ad'
FEATURE_NAMES = [
    'previous_binary_margin', 'current_binary_margin', 'minimum_binary_margin',
    'binary_margin_change', 'log_previous_token_length', 'log_current_token_length',
    'log_current_turn', 'previous_chunk_count', 'current_chunk_count', 'chunk_count_change',
]

def guarded(path):
    p = Path(path).resolve()
    if any('test' in part.lower() or 'restricted' in part.lower() for part in p.parts):
        raise ValueError('Test/restricted access prohibited: ' + str(p))
    return p

def read_jsonl(path):
    with guarded(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def sha256_file(path):
    h = hashlib.sha256()
    with guarded(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def write_json(path, obj):
    p = guarded(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    tmp.replace(p)

def probability(row):
    p = float(row['p_binary'][1] if 'p_binary' in row else row['p_fraud'])
    if not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError('Invalid binary probability')
    return p

def turn(row):
    for k in ('prefix_turn', 'prefix_end_turn'):
        if k in row:
            return int(row[k])
    return int(str(row['sample_id']).rsplit('__P', 1)[1])

def pair_features(previous, current):
    if turn(current) != turn(previous) + 1:
        raise ValueError('Gate requires adjacent cumulative prefixes')
    def margin(row):
        p = min(1-1e-7, max(1e-7, probability(row)))
        return math.log(p/(1-p))
    a, b = margin(previous), margin(current)
    la, lb = int(previous['raw_wordpiece_length']), int(current['raw_wordpiece_length'])
    ca = int(previous.get('n_chunks', previous.get('num_chunks', 0)))
    cb = int(current.get('n_chunks', current.get('num_chunks', 0)))
    if la <= 0 or lb < la or ca < 1 or cb < ca:
        raise ValueError('Invalid prefix length/chunks')
    return [a, b, min(a,b), b-a, math.log1p(la), math.log1p(lb),
            math.log1p(turn(current)), float(ca), float(cb), float(cb-ca)]

class GateMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(10,16), nn.Tanh(), nn.Linear(16,1))
        nn.init.zeros_(self.net[2].weight)
        nn.init.constant_(self.net[2].bias, math.log(9.0))

    def forward(self, x):
        return self.net(x).squeeze(-1)

def load_gate(path):
    payload = torch.load(guarded(path), map_location='cpu')
    if payload['feature_names'] != FEATURE_NAMES or payload['m1_sha256'] != M1_SHA256:
        raise ValueError('Gate feature/model contract mismatch')
    if payload['training_used_dev'] or payload['training_used_test']:
        raise ValueError('Gate training isolation failure')
    m = GateMLP()
    m.load_state_dict(payload['state_dict'], strict=True)
    m.eval()
    return m, np.asarray(payload['mean']), np.asarray(payload['scale']), payload

@torch.no_grad()
def predict_gate(model, mean, scale, features):
    x = np.asarray(features, dtype=np.float32)
    x = np.clip((x-mean)/scale, -8, 8).astype(np.float32)
    return model(torch.from_numpy(x)).sigmoid().tolist()
