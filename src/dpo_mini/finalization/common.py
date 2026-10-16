import os
import time
from contextlib import contextmanager
from pathlib import Path
from ..provenance import ROOT, atomic_json, read_json, file_hash, digest, now


def execution_manifest():
    # Exact measurement dependency closure. Plan/source archives separately cover
    # orchestration, checks, tests and presentation, which may be completed later.
    paths=list((ROOT/'src/dpo_mini').glob('*.py'))
    paths += [ROOT/'src/dpo_mini/finalization'/p for p in ['__init__.py','common.py','strategy.py','worker.py']]
    paths += [ROOT/'scripts'/p for p in ['finalization_worker.py','_common.py']]
    paths += list((ROOT/'configs').glob('*'))
    files={p.relative_to(ROOT).as_posix():file_hash(p) for p in paths if p.is_file()}
    return {'files':files,'sha256':digest(files),'scope':'Measurement worker dependency closure; full source separately archived'}


class Stages:
    def __init__(self):
        self.values = {}
        self.active = None

    @contextmanager
    def span(self, name):
        if self.active is not None:
            raise RuntimeError('Nested stages would double count elapsed time')
        self.active = name
        start = time.perf_counter()
        try:
            yield
        finally:
            self.values[name] = self.values.get(name, 0.) + time.perf_counter() - start
            self.active = None


def offline_environment():
    for name in ['HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY',
                 'HF_HUB_DISABLE_IMPLICIT_TOKEN', 'PYTHONNOUSERSITE']:
        os.environ[name] = '1'
    os.environ.pop('PYTHONPATH', None)


def current_pass():
    record = read_json(ROOT/'artifacts/finalization/latest.json')
    return ROOT / (record.get('path') or record.get('finalization_path'))
