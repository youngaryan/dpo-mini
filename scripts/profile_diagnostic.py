"""A single diagnostic update; never included in reported timing trials."""
import _common
import argparse
import torch
from dpo_mini.config import Config
from dpo_mini.models import make_model,get_tokenizer
from dpo_mini.provenance import ROOT,read_json,atomic_json
from dpo_mini.train import Trainer

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--profile",default="toy");args=p.parse_args()
    cfg=Config(**read_json(ROOT/"artifacts/prepared"/args.profile/"config.json"))
    data=read_json(ROOT/"artifacts/prepared"/args.profile/"data.json")["train"][:1]
    model,ref=make_model(cfg),make_model(cfg,True)
    trainer=Trainer(model,ref,None,cfg,data,get_tokenizer(cfg).pad_token_id)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],record_shapes=False) as prof:
        trainer.run()
    target=ROOT/"artifacts/profiler";target.mkdir(parents=True,exist_ok=True)
    (target/"diagnostic.txt").write_text(prof.key_averages().table(sort_by="self_cpu_time_total",row_limit=25),encoding="utf8")
    atomic_json(target/"metadata.json",{"profile":args.profile,"pairs":1,"scope":"diagnostic only; excluded from timing trials"})
