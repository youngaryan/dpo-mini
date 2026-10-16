"""Small local provenance primitives. No tracking service or machine identifiers."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(canonical(value))
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def source_manifest() -> dict:
    files = {}
    for folder in ["src", "scripts", "tests", "configs"]:
        for p in sorted((ROOT / folder).rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                files[p.relative_to(ROOT).as_posix()] = file_hash(p)
    return {"files": files, "sha256": digest(files), "git": "No Git repository; source file hashes recorded"}


def seed_all(seed: int, threads: int = 4) -> None:
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False


def state_hash(model) -> str:
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().view(__import__("torch").uint8).numpy().tobytes())
    return h.hexdigest()


def environment() -> dict:
    import importlib.metadata
    import psutil
    import torch
    mem = psutil.virtual_memory()
    versions = {}
    for name in ["torch", "numpy", "pytest", "transformers", "datasets", "peft", "trl", "accelerate", "psutil"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    cpu_name = platform.processor()
    if os.name == "nt":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
            cpu_name = winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
    return {"recorded_at": now(), "os": platform.platform(), "python": platform.python_version(),
            "cpu": cpu_name, "physical_cores": psutil.cpu_count(logical=False),
            "logical_cpus": psutil.cpu_count(), "ram_total_bytes": mem.total,
            "ram_available_bytes": mem.available, "disk_free_bytes": shutil.disk_usage(ROOT).free,
            "cuda_available": torch.cuda.is_available(), "cuda_runtime": torch.version.cuda,
            "accelerators": [{"name": torch.cuda.get_device_name(i), "memory_bytes": torch.cuda.get_device_properties(i).total_memory}
                             for i in range(torch.cuda.device_count())],
            "versions": versions, "torch_build": torch.__config__.show(),
            "exclusive_hardware": False, "driver_version": None,
            "driver_note": "nvidia-smi unavailable in initial inspection; no private process inventory collected"}
