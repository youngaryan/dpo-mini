"""Offline, resumable follow-up with leaf-process budget accounting."""
import _common
import argparse
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from dpo_mini.provenance import ROOT, atomic_json, read_json, now, file_hash, digest
from dpo_mini.finalization.common import current_pass, offline_environment, execution_manifest


def execute(folder, name, command, max_hours, expected=None, cwd=ROOT):
    import shutil
    if shutil.disk_usage(ROOT).free<2*1024**3:
        raise RuntimeError('Preserve at least 2 GiB free disk space')
    inventory_path=folder/'inventory.json'
    if inventory_path.exists():
        inventory=read_json(inventory_path)
        total=0
        for base,dirs,files in os.walk(ROOT):
            for filename in files:
                try:total+=(Path(base)/filename).stat().st_size
                except PermissionError:pass
        additional=max(0,total-inventory['storage_at_start_bytes'])
        atomic_json(folder/'storage-audit.json',{'logical_bytes':total,'additional_bytes':additional,
            'limit_additional_bytes':inventory['additional_storage_limit_bytes'],'at':now(),
            'scope':'Logical file sizes; inaccessible historical pytest temporary directories skipped; linked files conservatively counted per path'})
        if additional>inventory['additional_storage_limit_bytes']:
            raise RuntimeError('Additional storage budget exceeded')
    ledger_path=folder/'ledger.json'
    ledger=read_json(ledger_path) if ledger_path.exists() else []
    spent=sum(r.get('charged_seconds',0.) for r in ledger)
    remaining=max_hours*3600-spent
    if remaining<=0: raise RuntimeError('New experiment-process budget exhausted')
    token=name+'-'+uuid.uuid4().hex[:8]
    log=folder/'logs'/ (token+'.log');log.parent.mkdir(parents=True,exist_ok=True)
    record={'id':token,'stage':name,'command':command,'started_at':now(),'status':'running',
            'source':execution_manifest()['sha256'],'log':log.relative_to(ROOT).as_posix(),
            'accounting':'Only leaf subprocess elapsed; parent waits are not charged again',
            'timeout_seconds':min(5400,remaining)}
    ledger.append(record);atomic_json(ledger_path,ledger)
    begin=time.perf_counter()
    with log.open('wb') as stream:
        child=subprocess.Popen(command,cwd=cwd,stdout=stream,stderr=subprocess.STDOUT,env=os.environ.copy())
        record['pid']=child.pid
        import psutil
        record['process_create_time']=psutil.Process(child.pid).create_time()
        atomic_json(ledger_path,ledger)
        try:
            result=child.wait(timeout=record['timeout_seconds'])
        except subprocess.TimeoutExpired:
            child.terminate();child.wait(timeout=30);result=-1
        record.update(exit_code=result,ended_at=now(),charged_seconds=time.perf_counter()-begin,
                      status='completed' if result==0 else 'failed')
        if expected is not None and expected.exists():
            record['result']={'path':expected.relative_to(ROOT).as_posix(),'sha256':file_hash(expected)}
        atomic_json(ledger_path,ledger)
    if result: raise RuntimeError(f'{name} failed; see {log.relative_to(ROOT)}')
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--offline',action='store_true')
    p.add_argument('--max-experiment-hours',type=float,default=6.)
    p.add_argument('--resume',action='store_true')
    modes=p.add_mutually_exclusive_group()
    modes.add_argument('--report-only',action='store_true')
    modes.add_argument('--verify-only',action='store_true')
    modes.add_argument('--status',action='store_true')
    p.add_argument('--stage',choices=['core','parity','audit','diagnose','precision','experiments','clean','package'])
    args=p.parse_args()
    if not 0<args.max_experiment_hours<=6: p.error('Additional experiment budget must be in (0,6] hours')
    if args.offline: offline_environment()
    folder=current_pass()
    if args.status:
        from dpo_mini.finalization.status import refresh
        state=refresh(folder)
        print({k:state[k] for k in ['stages','new_measured_leaf_process_seconds']})
        print([{'stage':r['stage'],'updates':r['progress'].get('completed_updates'),'pairs':r['progress'].get('completed_pairs')} for r in state['active']])
        return
    if args.report_only or args.verify_only:
        from dpo_mini.finalization.report import build, verify
        if args.report_only: build(folder)
        else: verify(folder)
        return
    if args.stage=='core':
        (folder/'test-tmp').mkdir(parents=True,exist_ok=True)
        if (folder/'core-tests.xml').exists():
            import shutil
            shutil.copy2(folder/'core-tests.xml',folder/'logs'/('core-previous-'+uuid.uuid4().hex[:8]+'.xml'))
        execute(folder,'core',[sys.executable,'-m','pytest','-q','tests','--basetemp',
            str(folder/'test-tmp'/uuid.uuid4().hex), '--junitxml',str(folder/'core-tests.xml')],args.max_experiment_hours,folder/'core-tests.xml')
    elif args.stage in ['parity','audit','diagnose','precision']:
        out=folder/args.stage
        if args.resume and (out/'result.json').exists():
            result=read_json(out/'result.json')
            if result.get('status')=='passed':
                print(f'{args.stage}: verified existing result; no rerun requested')
                return
        if (out/'result.json').exists():
            out=folder/(args.stage+'-attempt-'+uuid.uuid4().hex[:8])
        execute(folder,args.stage,[sys.executable,'scripts/finalization_worker.py',args.stage,str(out)],args.max_experiment_hours,out/'result.json')
    else:
        from dpo_mini.finalization.pipeline import run
        run(args,folder,execute)

if __name__=='__main__': main()
