"""Local stage orchestrator. New invocation -> new study; resume is hash-checked."""
import _common
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from dpo_mini.provenance import ROOT,atomic_json,digest,environment,file_hash,now,read_json,source_manifest


def parser():
    p=argparse.ArgumentParser()
    p.add_argument("--profile",choices=["auto","toy","compact","standard"],default="auto")
    p.add_argument("--offline",action="store_true")
    p.add_argument("--resume",action="store_true")
    p.add_argument("--report-only",action="store_true")
    p.add_argument("--pilot-only",action="store_true")
    p.add_argument("--max-experiment-hours",type=float,default=6)
    p.add_argument("--max-subprocess-minutes",type=float,default=90)
    p.add_argument("--max-download-gib",type=float,default=10)
    p.add_argument("--max-storage-gib",type=float,default=20)
    p.add_argument("--seeds",type=int,nargs="+",default=[42,123,2026])
    p.add_argument("--threads",type=int,default=4)
    p.add_argument("--device",choices=["auto","cpu","cuda","mps"],default="auto")
    return p


class Execution:
    def __init__(self,args):
        self.args=args
        ledger=ROOT/"artifacts/process_ledger.json"
        self.ledger=read_json(ledger) if ledger.exists() else []
        self.used=sum(r["wall_seconds"] for r in self.ledger if r["experiment_budget"])
        self.id=now().replace(":","").replace(".","-")+"-"+uuid.uuid4().hex[:6]
        self.root=ROOT/"artifacts/studies"/self.id
        self.stages={k:{"status":"planned"} for k in ["A_core","B_upstream","C_pilot_and_freeze","D_benchmark","E_learning","F_report"]}
        self.runs=[]
        if args.resume and (ROOT/"artifacts/latest_study.json").exists():
            self.root=ROOT/read_json(ROOT/"artifacts/latest_study.json")["path"]
            old=read_json(self.root/"status.json")
            self.id=old["study_id"];self.stages=old["stages"];self.runs=old["runs"]
        else:self.root.mkdir(parents=True)

    def stage(self,name,status,**more):
        self.stages[name]={"status":status,"at":now(),**more}
        atomic_json(self.root/"status.json",{"study_id":self.id,"stages":self.stages,"runs":self.runs,
                    "experiment_process_seconds":self.used})
        print(name,status,more,flush=True)

    def command(self,args,log:Path,experiment=False,timeout=None):
        log.parent.mkdir(parents=True,exist_ok=True)
        start=time.perf_counter()
        remaining=self.args.max_experiment_hours*3600-self.used
        import psutil
        if psutil.virtual_memory().available < 1024**3 or shutil.disk_usage(ROOT).free < 2*1024**3:
            raise RuntimeError("Insufficient RAM/disk safety margin before child launch")
        storage=sum(p.stat().st_size for p in ROOT.rglob("*") if p.is_file())
        if storage > self.args.max_storage_gib*1024**3:
            raise RuntimeError("Project/environment/cache storage budget exceeded")
        seconds=min(self.args.max_subprocess_minutes*60,remaining) if experiment else 5400
        if timeout is not None: seconds=min(seconds,timeout)
        if seconds<=0: raise TimeoutError("Total experiment subprocess budget exhausted")
        with log.open("w",encoding="utf8") as out:
            try:
                process=subprocess.run([sys.executable,*args],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,
                                       timeout=seconds,check=False)
                code=process.returncode
            except subprocess.TimeoutExpired:
                code=124
                out.write("\nSTOPPED BY BUDGET: subprocess terminated and waited.\n")
        elapsed=time.perf_counter()-start
        if experiment: self.used+=elapsed
        record={"command":[sys.executable,*args],"exit_status":code,
                "wall_seconds":elapsed,"experiment_budget":experiment,"ended_at":now(),"project_storage_bytes_before":storage}
        atomic_json(log.with_suffix(".command.json"),record)
        self.ledger.append(record)
        atomic_json(ROOT/"artifacts/process_ledger.json",self.ledger)
        if code:
            raise RuntimeError(f"Command exit {code}; see {log.relative_to(ROOT)}")
        return elapsed

    def worker(self,spec,label):
        run=ROOT/"artifacts/runs"/(self.id+"-"+label)
        if self.args.resume:
            candidates=[run,*sorted(run.parent.glob(run.name+"-retry-*"))]
            for candidate in candidates:
                if (candidate/"metrics.json").exists():
                    existing=read_json(candidate/"spec.json")
                    provenance=read_json(candidate/"provenance.json")
                    if existing!=spec or provenance["source"]["sha256"]!=source_manifest()["sha256"]:
                        raise ValueError("Refusing incompatible completed run reuse")
                    if provenance["status"]=="completed":
                        return candidate,read_json(candidate/"metrics.json")
            if run.exists():run=run.with_name(run.name+"-retry-"+uuid.uuid4().hex[:6])
        run.mkdir(parents=True)
        atomic_json(run/"spec.json",spec)
        try:
            elapsed=self.command(["scripts/worker.py","--spec",str(run/"spec.json"),"--run-dir",str(run)],
                                 run/"stdout.log",experiment=True)
        except Exception as error:
            p=run/"provenance.json"
            provenance=read_json(p) if p.exists() else {"parent_plan_hash":spec.get("plan_hash")}
            command=read_json(run/"stdout.command.json") if (run/"stdout.command.json").exists() else {}
            provenance.update(status="stopped by budget" if command.get("exit_status")==124 else "failed",
                              ended_at=now(),error=repr(error),exit_status=command.get("exit_status"))
            atomic_json(p,provenance)
            self.stage("run:"+label,provenance["status"],reason=repr(error))
            raise
        m=read_json(run/"metrics.json")
        m["whole_process_seconds"]=elapsed
        atomic_json(run/"metrics.json",m)
        self.runs.append(run.relative_to(ROOT).as_posix())
        self.stage("run:"+label,"completed",seconds=round(elapsed,3))
        return run,m

    def prepare(self,name):
        prepared=ROOT/"artifacts/prepared"/name/"config.json"
        if not prepared.exists():
            if name!="toy" and not (ROOT/"artifacts"/f"{name}-resolved.json").exists():
                if self.args.offline:
                    raise RuntimeError(f"Offline: no pinned {name} model/data artifacts")
                self.command(["scripts/fetch_public.py","--profile",name,"--max-download-gib",str(self.args.max_download_gib)],
                             self.root/(name+"-download.log"))
            self.command(["scripts/prepare_data.py","--profile",name],self.root/(name+"-prepare.log"))
        cfg=read_json(prepared)
        data=read_json(prepared.parent/"data.json")
        manifest=read_json(prepared.parent/"manifest.json")
        for split,rows in data.items():
            if digest(rows)!=manifest["splits"][split]["split_hash"]:
                raise ValueError("Prepared tokenized data differs from its manifest")
        if name!="toy":
            ledger=read_json(ROOT/"artifacts/download_ledger.json")
            model_files=[r for r in ledger if r["path"].startswith(f".cache/models/{name}/")]
            for record in model_files:
                if cfg["revision"] not in record["url"] or file_hash(ROOT/record["path"])!=record["sha256"]:
                    raise ValueError("Pinned model file/revision integrity mismatch")
        import torch
        device=self.args.device
        if device=="auto":device="cuda" if torch.cuda.is_available() else "cpu"
        smoke=torch.ones(2,device=device,requires_grad=True);smoke.square().sum().backward()
        cfg["device"]=device
        cfg.update(threads=self.args.threads,max_experiment_hours=self.args.max_experiment_hours,
                   max_subprocess_minutes=self.args.max_subprocess_minutes,max_download_gib=self.args.max_download_gib,
                   max_storage_gib=self.args.max_storage_gib)
        return cfg

    def report(self):
        self.command(["scripts/analyze.py","--study",str(self.root)],self.root/"analysis.log")
        self.command(["scripts/build_paper.py"],self.root/"paper.log")
        self.command(["scripts/verify_artifacts.py"],self.root/"verification.log")


def resume_study(args,plan):
    exe=Execution(args)
    cfg=plan["config"]
    base={"config":cfg,"plan_hash":plan["sha256"],"data_path":f'artifacts/prepared/{cfg["profile"]}/data.json'}
    for trial in range(plan["timing_trials"]):
        arms=["uncached_reference","cached_reference"] if trial%2==0 else ["cached_reference","uncached_reference"]
        for arm in arms:
            run,_=exe.worker({**base,"kind":"benchmark","arm":arm,"seed":42,"trial":trial,
                              "warmup_updates":plan["warmup_updates"],"measured_updates":plan["measured_updates"]},f"timing-{trial}-{arm}")
            if arm=="cached_reference":cold=run
        exe.worker({**base,"kind":"benchmark","arm":"cached_reference","seed":42,"trial":trial,
                    "warmup_updates":plan["warmup_updates"],"measured_updates":plan["measured_updates"],
                    "warm_cache":(cold/"reference-cache.json").relative_to(ROOT).as_posix()},f"timing-{trial}-warm")
    exe.stage("D_benchmark","completed")
    for seed in plan["seeds"]:
        for arm in ["uncached_reference","cached_reference"]:
            exe.worker({**base,"kind":"learning","arm":arm,"seed":seed},f"learn-{seed}-{arm}")
    exe.stage("E_learning","completed")
    exe.report();exe.stage("F_report","completed")
    atomic_json(ROOT/"artifacts/status.json",read_json(exe.root/"status.json"))


def main():
    args=parser().parse_args()
    if any(x<=0 for x in [args.max_experiment_hours,args.max_subprocess_minutes,args.max_download_gib,args.max_storage_gib,args.threads]):
        raise ValueError("Resource limits and CPU thread count must be positive")
    (ROOT/"tmp").mkdir(exist_ok=True)
    if args.report_only:
        final_latest=ROOT/'artifacts/finalization/latest.json'
        if final_latest.exists():
            final_folder=ROOT/read_json(final_latest)['path']
            evidence=final_folder/'evidence/historical.json'
            if evidence.exists():
                historical=read_json(evidence)
                latest_path=ROOT/'artifacts/latest_study.json'
                latest_id=Path(read_json(latest_path)['path']).name if latest_path.exists() else None
                if latest_id is None or all(r['run_id'].startswith(latest_id) for r in historical['runs']):
                    from dpo_mini.finalization.report import build
                    build(final_folder)
                    return
        latest=read_json(ROOT/"artifacts/latest_study.json")
        for command in [["scripts/analyze.py","--study",str(ROOT/latest["path"])],["scripts/build_paper.py"],["scripts/verify_artifacts.py"]]:
            subprocess.run([sys.executable,*command],cwd=ROOT,check=True)
        return
    if args.resume and (ROOT/"artifacts/latest_study.json").exists():
        latest=read_json(ROOT/"artifacts/latest_study.json")
        plan=read_json(ROOT/latest["path"]/"experiment_plan.json")
        if args.profile not in ["auto",plan["config"]["profile"]]:
            raise ValueError("Resume profile differs from frozen plan")
        if plan["code_hash"]!=source_manifest()["sha256"]:
            raise ValueError("Source changed since frozen plan; use a new study, not --resume")
        status=read_json(ROOT/latest["path"]/"status.json")
        if status["stages"].get("E_learning",{}).get("status")=="completed":
            subprocess.run([sys.executable,"scripts/reproduce.py","--report-only"],cwd=ROOT,check=True)
            return
        resume_study(args,plan)
        return
    exe=Execution(args)
    atomic_json(ROOT/"artifacts/environment.json",environment())
    try:
        exe.stage("A_core","running")
        exe.command(["-m","pytest","-q","-m","not integration and not network","--basetemp",str(ROOT/"tmp"/("pytest-"+exe.id)),
                     "--junitxml",str(exe.root/"core.xml")],exe.root/"core.log",experiment=True)
        exe.stage("A_core","completed")
        if not (ROOT/"artifacts/source_registry.json").exists() and not args.offline:
            exe.command(["scripts/fetch_public.py","--profile","sources"],exe.root/"sources.log")
        exe.stage("B_upstream","running")
        try:
            exe.command(["scripts/verify_upstream.py"],exe.root/"upstream.log",experiment=True)
            exe.stage("B_upstream","completed")
        except RuntimeError as e:
            exe.stage("B_upstream","failed",reason=str(e))
        # Required toy pilot precedes profile selection, independently of model availability.
        names=["toy"]+(["standard","compact"] if args.profile=="auto" else ([] if args.profile=="toy" else [args.profile]))
        pilots={}
        selected=None
        selected_seeds=args.seeds
        selected_warmup,selected_measured=5,20
        for name in names:
            try:
                cfg=exe.prepare(name)
                arms={}
                for arm in ["uncached_reference","cached_reference"]:
                    spec={"config":cfg,"kind":"pilot","arm":arm,"data_path":f"artifacts/prepared/{name}/data.json","seed":42}
                    _,arms[arm]=exe.worker(spec,"pilot-"+name+"-"+arm)
                pilots[name]=arms
                u=arms["uncached_reference"]["training_seconds"]/arms["uncached_reference"]["pairs"]
                c=arms["cached_reference"]["training_seconds"]/arms["cached_reference"]["pairs"]
                setup=sum(m["setup_including_cache_seconds"] for m in arms.values())
                # Conservative projection includes three arms/trial, cache setup, heldout evaluation,
                # generation, checkpoints and 50% headroom. One main epoch only.
                benchmark_pairs=25*cfg["effective_batch_pairs"]
                epoch=cfg["train_pairs"]*(u+2*c)
                estimate=1.5*(3*benchmark_pairs*(u+2*c)+len(args.seeds)*(epoch+3*setup+120))
                ram_margin=min(m["whole_workflow_memory"]["minimum_available_ram_bytes"] for m in arms.values())
                fits=estimate < max(0,args.max_experiment_hours*3600-exe.used)*.8 and ram_margin>2*1024**3
                seeds_for_profile=args.seeds
                warmup,measured=5,20
                if not fits and len(args.seeds)>1:
                    one_seed=1.5*(3*benchmark_pairs*(u+2*c)+(epoch+3*setup+120))
                    if one_seed < max(0,args.max_experiment_hours*3600-exe.used)*.8 and ram_margin>2*1024**3:
                        estimate=one_seed;fits=True;seeds_for_profile=args.seeds[:1]
                if not fits:
                    reduced=1.5*(3*10*cfg["effective_batch_pairs"]*(u+2*c)+(epoch+3*setup+120))
                    if reduced < max(0,args.max_experiment_hours*3600-exe.used)*.8 and ram_margin>2*1024**3:
                        estimate=reduced;fits=True;seeds_for_profile=args.seeds[:1];warmup,measured=2,8
                pilots[name]["projection"]={"seconds":estimate,"headroom_factor":1.5,"fits":fits,"ram_margin_bytes":ram_margin,
                                             "seeds":seeds_for_profile,"requested_seeds":args.seeds,"warmup_updates":warmup,"measured_updates":measured}
                atomic_json(exe.root/"pilots.json",pilots)
                if name!="toy" and fits:
                    selected=cfg
                    selected_seeds=seeds_for_profile
                    selected_warmup,selected_measured=warmup,measured
                    break
                if args.profile=="toy": selected=cfg
            except Exception as error:
                exe.stage("pilot:"+name,"failed",reason=repr(error))
                if name=="toy": raise
        if selected is None:
            selected=exe.prepare("toy")
        if args.pilot_only:
            exe.stage("C_pilot","completed",selected_profile=selected["profile"])
            return
        plan={"created_at":now(),"config":selected,"seeds":selected_seeds,"requested_seeds":args.seeds,
              "seed_reduction_reason":"CPU pilot budget/headroom; retain one complete paired seed" if selected_seeds!=args.seeds else None,
              "epochs":1,"timing_trials":3,
              "warmup_updates":selected_warmup,"measured_updates":selected_measured,
              "timing_reduction_reason":"CPU budget/headroom: reduced warmup and measured work, same in every arm" if selected_measured<20 else None,
              "code_hash":source_manifest()["sha256"],
              "data_file_sha256":file_hash(ROOT/f'artifacts/prepared/{selected["profile"]}/data.json'),
              "data_manifest_sha256":digest(read_json(ROOT/f'artifacts/prepared/{selected["profile"]}/manifest.json')),
              "pilot_artifact":str((exe.root/"pilots.json").relative_to(ROOT)),
              "selection_reason":"Largest measured profile fitting conservative wall-time and RAM projections; toy fallback otherwise",
              "metric_definitions":{"pair":"one chosen and one rejected sequence","strict_positive":"beta margin >1e-6",
               "ties":"absolute beta margin <=1e-6","response_tokens":"both completions including assistant end; no prompt/padding"},
              "timing_scope":"Fresh process; matched declared warmup/measured updates; evaluation/checkpoints absent from timing-only runs; train rows cycle for fixed timing workload",
              "stopping_rules":"6h cumulative experiment subprocesses or configured override; <=90min/subprocess; no OOM retry beyond two",
              "precision":"FP32 weights/compute/reductions; no quantization, compilation, packing or distributed training",
              "optimizer":"AdamW lr from config; betas .9/.999; eps1e-8; wd0; clip1; linear 3% warmup/decay"}
        plan["sha256"]=digest(plan)
        atomic_json(exe.root/"experiment_plan.json",plan)
        atomic_json(ROOT/"artifacts/experiment_plan.json",plan)
        atomic_json(ROOT/"artifacts/data_manifest.json",read_json(ROOT/f'artifacts/prepared/{selected["profile"]}/manifest.json'))
        atomic_json(ROOT/"artifacts/latest_study.json",{"path":exe.root.relative_to(ROOT).as_posix()})
        exe.stage("C_pilot_and_freeze","completed",plan_hash=plan["sha256"],profile=selected["profile"])
        try:
            exe.command(["scripts/profile_diagnostic.py","--profile",selected["profile"]],exe.root/"profiler.log",experiment=True)
            exe.stage("diagnostic_profile","completed")
        except RuntimeError as error:
            exe.stage("diagnostic_profile","failed",reason=str(error))
        base={"config":selected,"plan_hash":plan["sha256"],"data_path":f'artifacts/prepared/{selected["profile"]}/data.json'}
        exe.stage("D_benchmark","running")
        for trial in range(3):
            arms=["uncached_reference","cached_reference"] if trial%2==0 else ["cached_reference","uncached_reference"]
            cold_run=None
            for arm in arms:
                run,_=exe.worker({**base,"kind":"benchmark","arm":arm,"seed":42,"trial":trial,
                                  "warmup_updates":selected_warmup,"measured_updates":selected_measured},f"timing-{trial}-{arm}")
                if arm=="cached_reference":cold_run=run
            exe.worker({**base,"kind":"benchmark","arm":"cached_reference","seed":42,"trial":trial,
                        "warmup_updates":selected_warmup,"measured_updates":selected_measured,
                        "warm_cache":(cold_run/"reference-cache.json").relative_to(ROOT).as_posix()},f"timing-{trial}-warm")
        exe.stage("D_benchmark","completed")
        exe.stage("E_learning","running")
        for seed in selected_seeds:
            for arm in ["uncached_reference","cached_reference"]:
                exe.worker({**base,"kind":"learning","arm":arm,"seed":seed},f"learn-{seed}-{arm}")
        exe.stage("E_learning","completed")
        exe.stage("F_report","running")
        exe.report()
        exe.stage("F_report","completed")
        atomic_json(ROOT/"artifacts/status.json",read_json(exe.root/"status.json"))
    except Exception as e:
        exe.stage("execution","failed",error=repr(e))
        atomic_json(ROOT/"artifacts/status.json",read_json(exe.root/"status.json"))
        # Reporting an incomplete execution must not turn it into a successful study.
        if (exe.root/"experiment_plan.json").exists() and (ROOT/"scripts/analyze.py").exists():
            try: exe.report()
            except Exception as report_error: exe.stage("F_report","failed",reason=repr(report_error))
        raise


if __name__=="__main__": main()
