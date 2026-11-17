"""Runtime configuration: configs/inference.json and repository paths.

Model files are resolved relative to the repository root (or OUTAGE_FORECAST_HOME when set); they are downloaded with
`python scripts/download_artifacts.py models` and verified against the SHA-256 recorded in the configuration.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

PACKAGE_ROOT = Path(os.environ.get('OUTAGE_FORECAST_HOME', Path(__file__).resolve().parents[2])).resolve()
CONFIG_FILE = PACKAGE_ROOT / 'configs' / 'inference.json'


@dataclass(frozen=True)
class Config:
    raw: dict

    @classmethod
    def load(cls, path: Path = CONFIG_FILE) -> 'Config':
        config = cls(json.loads(Path(path).read_text(encoding='utf-8')))
        config.validate()
        return config

    def artifacts(self):
        """(path, sha256) of every model file named in the configuration."""
        def walk(node):
            if isinstance(node, dict):
                for key, value in node.items():
                    if key in ('path', 'weights', 'normalizer', 'checkpoint') and isinstance(value, str):
                        yield value, node.get('sha256' if key == 'path' else key + '_sha256')
                    else:
                        yield from walk(value)
            elif isinstance(node, list):
                for value in node:
                    yield from walk(value)
        return list(walk({k: self.raw[k] for k in ('task_a', 'task_b')}))

    def validate(self) -> None:
        """Every referenced feature schema exists and every model file is present with its recorded SHA-256."""
        schemas = self.raw['features']
        def references(node):
            if isinstance(node, dict):
                for key, value in node.items():
                    if key == 'features' and isinstance(value, str):
                        yield value
                    else:
                        yield from references(value)
        unknown = {r for r in references({k: self.raw[k] for k in ('task_a', 'task_b')}) if r not in schemas}
        if unknown:
            raise RuntimeError(f'inference.json refers to unknown feature schemas {sorted(unknown)}')
        for relative, digest in self.artifacts():
            path = self.path(relative)
            if not path.is_file():
                raise RuntimeError(f'model file missing: {relative} (run: python scripts/download_artifacts.py models)')
            if digest is None or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise RuntimeError(f'model file does not match its recorded SHA-256: {relative}')

    def __getitem__(self, key):
        return self.raw[key]

    def features(self, name: str) -> list[str]:
        """Ordered feature schema by name (see "features" in inference.json)."""
        return list(self.raw['features'][name])

    def path(self, relative: str) -> Path:
        return PACKAGE_ROOT / relative

    @property
    def counties(self) -> dict[str, int]:
        """The official counties and their fixed MCC denominators."""
        return self.raw['counties']

    @property
    def county_order(self) -> list[str]:
        """The 40 training counties in encoding order (county one-hot, embedding index, regions)."""
        return self.raw['county_order']

    @property
    def denominators(self) -> dict[str, int]:
        return self.raw['denominators']

    @property
    def region_of(self) -> dict[str, str]:
        return {c: region for region, members in self.raw['regions'].items() for c in members}
