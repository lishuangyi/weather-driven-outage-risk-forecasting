"""Download the release assets (models and input data) and verify their SHA-256.

    python scripts/download_artifacts.py models               # frozen models (39 MB)  -> models/
    python scripts/download_artifacts.py test-window          # 2025 evaluation inputs (75 MB) -> data/inputs/input_2025/
    python scripts/download_artifacts.py training             # 2022-2024 full-year inputs (0.8 GB) -> data/inputs/
    python scripts/download_artifacts.py all

Assets and checksums are pinned in artifacts/release_v1.0.0.json. Downloads go through the GitHub CLI (`gh`) when it is
installed and logged in, otherwise plain HTTPS. Archives are cached in data/cache/ and extracted only after the
checksum matches. Inference itself never needs the network.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'artifacts' / 'release_v1.0.0.json'
CACHE = ROOT / 'data' / 'cache'


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def fetch(repo: str, tag: str, name: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + '.part')
    if shutil.which('gh'):
        done = subprocess.run(['gh', 'release', 'download', tag, '-R', repo, '-p', name, '-O', str(partial), '--clobber'])
        if done.returncode == 0:
            partial.replace(target)
            return
        print('  gh download failed, trying HTTPS', file=sys.stderr)
    url = f'https://github.com/{repo}/releases/download/{tag}/{name}'
    with urllib.request.urlopen(url) as response, partial.open('wb') as f:
        shutil.copyfileobj(response, f, 1 << 20)
    partial.replace(target)


def extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        tar.extractall(destination, filter='data')


def main() -> int:
    manifest = json.loads(MANIFEST.read_text())
    groups = sorted({a['group'] for a in manifest['assets']})
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('groups', nargs='+', choices=groups + ['all'])
    p.add_argument('--keep-archives', action='store_true', help='keep the downloaded archives in data/cache/')
    a = p.parse_args()
    wanted = set(groups) if 'all' in a.groups else set(a.groups)
    for asset in manifest['assets']:
        if asset['group'] not in wanted:
            continue
        archive = CACHE / asset['name']
        if not (archive.is_file() and sha256(archive) == asset['sha256']):
            print(f"downloading {asset['name']} ({asset['bytes'] / 1e6:.0f} MB)", flush=True)
            fetch(manifest['repository'], manifest['release'], asset['name'], archive)
        digest = sha256(archive)
        if digest != asset['sha256']:
            print(f"checksum mismatch for {asset['name']}: {digest}", file=sys.stderr)
            return 1
        extract(archive, ROOT / asset['extract_to'])
        print(f"  ok  {asset['name']} -> {asset['extract_to']}", flush=True)
        if not a.keep_archives:
            archive.unlink()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
