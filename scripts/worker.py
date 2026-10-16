import _common
import argparse
from pathlib import Path
from dpo_mini.provenance import read_json
from dpo_mini.benchmark import run_worker

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--spec",type=Path,required=True)
    parser.add_argument("--run-dir",type=Path,required=True)
    args=parser.parse_args()
    result=run_worker(read_json(args.spec),args.run_dir)
    print({k:result[k] for k in ["status","kind","arm","training_seconds","pairs","workflow_seconds_excluding_imports"]},flush=True)
