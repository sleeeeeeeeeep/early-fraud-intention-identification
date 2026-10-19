#!/usr/bin/env python3
"""Verify the published source/document snapshot without opening datasets."""
from pathlib import Path
import hashlib
import json
import sys

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / 'docs/export_manifest.json').read_text(encoding='utf-8'))
failures = []
for entry in manifest['files']:
    path = root / entry['path']
    if not path.is_file():
        failures.append('missing: ' + entry['path'])
        continue
    data = path.read_bytes()
    if len(data) != entry['bytes'] or hashlib.sha256(data).hexdigest() != entry['sha256']:
        failures.append('changed: ' + entry['path'])
if failures:
    print('\n'.join(failures), file=sys.stderr)
    raise SystemExit(1)
print('Verified {} exported files.'.format(len(manifest['files'])))
