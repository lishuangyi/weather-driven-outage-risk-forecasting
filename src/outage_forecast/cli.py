"""Command line: python infer.py --input-dir /input --output /output/predictions.csv

Exit codes: 0 success, 2 missing or invalid input (nothing is written). The program reads only the package and the
input directory, needs no network, does not train, and writes only the output file (replaced atomically on success)
plus a temporary work directory next to it for library caches, removed at the end.
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Power-outage risk forecasting for AI data centers: inference')
    parser.add_argument('--input-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto', help=argparse.SUPPRESS)
    parser.add_argument('--diagnostics-json', type=Path, default=None, help=argparse.SUPPRESS)  # used by demo/run_demo.py
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', stream=sys.stderr)
    input_dir, output = args.input_dir.resolve(), args.output.resolve()
    if not input_dir.is_dir():
        logging.error('Inference cannot run: input directory not found: %s', input_dir)
        return 2
    if output.is_relative_to(input_dir):
        logging.error('Inference cannot run: the output must not be inside the input directory')
        return 2
    diagnostics_json = args.diagnostics_json.resolve() if args.diagnostics_json else None
    from .config import PACKAGE_ROOT as package
    if diagnostics_json is not None and (diagnostics_json.is_relative_to(input_dir) or diagnostics_json.is_relative_to(package)
                                         or diagnostics_json == output or not diagnostics_json.name or diagnostics_json.is_dir()):
        logging.error('Inference cannot run: the diagnostics file must be a file outside the input directory and the package, '
                      'and must differ from the output')
        return 2
    output.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='.aidc_work_', dir=output.parent))
    try:
        tempfile.tempdir = str(work)
        for name in ('TMPDIR', 'CUDA_CACHE_PATH', 'TRITON_CACHE_DIR', 'XDG_CACHE_HOME'):
            os.environ[name] = str(work)
        from .inputs import InputError  # imported after the cache locations are set (torch reads them on import)
        from .pipeline import run
        try:
            run(input_dir, work / 'predictions.csv', args.device, diagnostics_json=diagnostics_json)
        except InputError as error:
            logging.error('Inference cannot run: %s', error)
            return 2
        os.replace(work / 'predictions.csv', output)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print(f'Inference completed: {output}')
    return 0
