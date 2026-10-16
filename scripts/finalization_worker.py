import _common
from pathlib import Path
import argparse
from dpo_mini.finalization.common import offline_environment
offline_environment()

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['parity','audit','benchmark','diagnose','precision'])
    parser.add_argument('output',type=Path)
    parser.add_argument('--spec',type=Path)
    args=parser.parse_args()
    if args.stage=='parity':
        from dpo_mini.finalization.parity import run
        result=run(args.output)
        print(result['status'],flush=True)
    elif args.stage=='precision':
        from dpo_mini.finalization.precision_check import run
        result=run(args.output)
        print(result['status'],flush=True)
    elif args.stage=='diagnose':
        from dpo_mini.finalization.diagnose import run
        result=run(args.output)
        print(result['status'],flush=True)
    elif args.stage=='audit':
        from dpo_mini.finalization.audit import run
        result=run(args.output)
        print(result['status'],flush=True)
    else:
        from dpo_mini.finalization.worker import run
        from dpo_mini.provenance import read_json
        result=run(read_json(args.spec),args.output)
        print(result['status'],flush=True)
