"""Download the small public sources, with receipts (URL, time, bytes, SHA-256, MD5, source version).

    python data_pipeline/sources/download_public.py eaglei --year 2025 --out data/raw/eaglei
    python data_pipeline/sources/download_public.py census --out data/raw/census

  eaglei   EAGLE-I county outages, Figshare article 24237376 version 4 (DOI 10.6084/m9.figshare.24237376, CC BY 4.0):
           the annual file eaglei_outages_<year>.csv, MD5-checked
  census   US Census TIGER/Line 2025 county polygons (public domain): tl_2025_us_county.zip

An existing file is kept only when its receipt names the same source and its bytes still match.
"""
import argparse
import hashlib
import json
import shutil
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

FIGSHARE_API = 'https://api.figshare.com/v2/articles/24237376/versions/4'
CENSUS = 'https://www2.census.gov/geo/tiger/TIGER2025/COUNTY/tl_2025_us_county.zip'


def digest(path: Path, name: str) -> str:
    h = hashlib.new(name)
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 22), b''):
            h.update(block)
    return h.hexdigest()


def fetch(url: str, dst: Path, md5: str | None = None, meta: dict | None = None) -> None:
    receipt = dst.with_name(dst.name + '.receipt.json')
    if dst.exists() and receipt.exists():
        old = json.loads(receipt.read_text())
        same = old.get('url') == url and old.get('md5') == md5 and all(old.get(k) == v for k, v in (meta or {}).items())
        if same and (not md5 or digest(dst, 'md5') == md5) and digest(dst, 'sha256') == old.get('sha256'):
            print(f'skip {dst} (source and checksums verified)')
            return
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + '.part')
    with urllib.request.urlopen(url, timeout=600) as response, part.open('wb') as f:
        shutil.copyfileobj(response, f, 1 << 22)
    if md5 and digest(part, 'md5') != md5:
        part.unlink()
        raise SystemExit(f'MD5 mismatch for {url}')
    part.replace(dst)
    info = {'url': url, 'retrieved_at': datetime.now(timezone.utc).isoformat(), 'bytes': dst.stat().st_size,
            'sha256': digest(dst, 'sha256'), 'md5': md5, **(meta or {})}
    receipt.write_text(json.dumps(info, indent=2) + '\n')
    print(f'downloaded {dst} ({info["bytes"]} bytes)')


def eaglei(year: int, out: Path) -> None:
    with urllib.request.urlopen(FIGSHARE_API, timeout=60) as response:
        article = json.load(response)
    if article.get('version') != 4:
        raise SystemExit(f'expected Figshare article version 4, got {article.get("version")}')
    name = f'eaglei_outages_{year}.csv'
    item = next((f for f in article['files'] if f['name'] == name), None)
    if item is None:
        raise SystemExit(f'{name} is not in Figshare article version 4')
    fetch(item['download_url'], out / name, item.get('supplied_md5') or item.get('computed_md5'),
          {'figshare_article': 24237376, 'figshare_version': 4, 'figshare_file_id': item.get('id')})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('source', choices=['eaglei', 'census'])
    p.add_argument('--year', type=int, default=2025)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    eaglei(a.year, a.out) if a.source == 'eaglei' else fetch(CENSUS, a.out / 'tl_2025_us_county.zip')
