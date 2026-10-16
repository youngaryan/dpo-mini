import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
venv_python=ROOT/(".venv/Scripts/python.exe" if os.name=="nt" else ".venv/bin/python")
if venv_python.exists() and Path(sys.prefix).resolve() != (ROOT/".venv").resolve():
    import subprocess
    raise SystemExit(subprocess.run([str(venv_python),*sys.argv],cwd=ROOT).returncode)
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "hf"))
os.environ.setdefault("HF_DATASETS_CACHE", str(ROOT / ".cache" / "datasets"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
