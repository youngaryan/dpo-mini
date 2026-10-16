"""Create/use an isolated environment, install pinned dependencies, execute study."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import venv

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument("--offline",action="store_true")
    args,_=p.parse_known_args()
    folder=ROOT/".venv"
    exe=folder/("Scripts/python.exe" if os.name=="nt" else "bin/python")
    (ROOT/"artifacts").mkdir(exist_ok=True)
    if not exe.exists():
        venv.EnvBuilder(with_pip=False).create(folder)
    probe=subprocess.run([str(exe),"-c","import torch,numpy,pytest,transformers,datasets,peft,trl,psutil,matplotlib,reportlab,fitz"],
                         cwd=ROOT,capture_output=True)
    if probe.returncode:
        uv=shutil.which("uv")
        lock=ROOT/"requirements.lock"
        requirements=lock if lock.exists() and lock.stat().st_size else ROOT/"requirements.in"
        if uv:
            command=[uv,"--system-certs","--cache-dir",str(ROOT/".cache/uv"),"pip","install","--python",str(exe),"-r",str(requirements)]
            if args.offline:command.append("--offline")
        else:
            subprocess.run([str(exe),"-m","ensurepip"],check=True)
            command=[str(exe),"-m","pip","install","-r",str(requirements),"--cache-dir",str(ROOT/".cache/pip")]
            if args.offline:command.append("--no-index")
        with (ROOT/"artifacts/bootstrap-install.log").open("a",encoding="utf8") as log:
            result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            raise SystemExit("Dependency installation failed; see artifacts/bootstrap-install.log. No global packages changed.")
    # All work stays in this synchronous child; no detached jobs.
    result=subprocess.run([str(exe),str(ROOT/"scripts/reproduce.py"),*sys.argv[1:]],cwd=ROOT)
    raise SystemExit(result.returncode)


if __name__=="__main__":main()
