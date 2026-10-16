import _common
import argparse
from dpo_mini.config import Config,profile
from dpo_mini.data import synthetic,prepare_pretrained
from dpo_mini.models import get_tokenizer
from dpo_mini.provenance import ROOT,atomic_json,read_json


def prepare(name):
    cfg=profile("toy") if name=="toy" else Config(**read_json(ROOT/"artifacts"/f"{name}-resolved.json"))
    if name=="toy":
        splits,manifest=synthetic(cfg)
    else:
        import pyarrow.parquet as pq
        raw={}
        for split in ["train_prefs","test_prefs"]:
            paths=sorted((ROOT/".cache/data").rglob(f"*{split}*.parquet"))
            if not paths:
                raise FileNotFoundError(f"No pinned local {split} files")
            raw[split]=[r for p in paths for r in pq.read_table(p).to_pylist()]
        splits,manifest=prepare_pretrained(raw,get_tokenizer(cfg),cfg)
        manifest["schema"]={s:{k:type(v).__name__ for k,v in rows[0].items()} for s,rows in raw.items()}
    cfg.train_pairs=len(splits["train"])
    cfg.validation_pairs=len(splits["validation"])
    cfg.test_pairs=len(splits["test"])
    target=ROOT/"artifacts/prepared"/name
    atomic_json(target/"data.json",splits)
    atomic_json(target/"manifest.json",manifest)
    atomic_json(target/"config.json",cfg.to_dict())
    tokenizer=get_tokenizer(cfg)
    atomic_json(target/"token-boundaries.json",[{"id":r["id"],"prompt_ids":r["prompt_ids"],
        "rendered_prompt":tokenizer.decode(r["prompt_ids"]), "chosen_ids":r["chosen_ids"],
        "chosen_mask":r["chosen_mask"],"rendered_chosen":tokenizer.decode(r["chosen_ids"]),
        "ending_id":tokenizer.eos_token_id} for r in splits["train"][:3]])
    print(name,{k:len(v) for k,v in splits.items()},manifest["statistics"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--profile",default="toy")
    prepare(p.parse_args().profile)
