#!/usr/bin/env python3
"""从部署包目录启动服务，首次运行自动下载 ASR 与翻译模型。"""
from pathlib import Path
import argparse
import os
import subprocess
import sys

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--host', default='127.0.0.1')
parser.add_argument('--port', type=int, default=7860)
parser.add_argument('--device', default='auto')
args = parser.parse_args()
env = os.environ.copy()
for name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'):
    env.pop(name, None)
env.update(GRADIO_ANALYTICS_ENABLED='False', USE_TF='0', USE_FLAX='0')
for name, folder in [('GRADIO_TEMP_DIR', 'gradio'), ('TMPDIR', 'tmp')]:
    path = root / '.runtime' / folder
    path.mkdir(parents=True, exist_ok=True)
    env[name] = str(path)
bypass = ','.join(['localhost', '127.0.0.1', '0.0.0.0', args.host, env.get('NO_PROXY', '')])
env['NO_PROXY'] = env['no_proxy'] = bypass
raise SystemExit(subprocess.call(
    [sys.executable, str(root / 'run.py'), '--host', args.host,
     '--port', str(args.port), '--device', args.device, '--route', 'text'],
    cwd=str(root), env=env,
))
