"""Download public artifacts only, with OS-verified TLS and no authentication."""
import _common
import argparse
import json
import os
import time
import urllib.request
from pathlib import Path
from dpo_mini.config import profile
from dpo_mini.provenance import ROOT, atomic_json, file_hash, now, read_json


def get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "DPO-Mini-public-reproducibility/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def download(url: str, target: Path, ledger: list, budget: int):
    if target.exists():
        records=[r for r in ledger if r["path"]==target.relative_to(ROOT).as_posix() and r["url"]==url]
        if not records or file_hash(target)!=records[-1]["sha256"]:
            raise ValueError("Existing artifact is corrupt or belongs to a different revision; preserve it and use a new snapshot directory")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "DPO-Mini-public-reproducibility/0.1"})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=60) as response:
        size = int(response.headers.get("Content-Length", "0"))
        # Conservatively reserve 3 GiB for installed package downloads and metadata.
        used = sum(x["bytes"] for x in ledger) + 3 * 1024**3
        if used + size > budget:
            raise RuntimeError("Download budget would be exceeded")
        total = 0
        temp = target.with_suffix(target.suffix + ".partial")
        with temp.open("wb") as out:
            while chunk := response.read(1 << 20):
                total += len(chunk)
                if used + total > budget:
                    raise RuntimeError("Download byte budget exhausted")
                out.write(chunk)
        if size and total != size:
            raise IOError("Truncated public artifact download")
        os.replace(temp, target)
    ledger.append({"url": url, "path": target.relative_to(ROOT).as_posix(), "bytes": total,
                   "sha256": file_hash(target), "seconds": time.perf_counter() - started, "retrieved_at": now()})
    atomic_json(ROOT / "artifacts/download_ledger.json", ledger)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["standard", "compact", "sources"], default="sources")
    parser.add_argument("--max-download-gib", type=float, default=10)
    args = parser.parse_args()
    ledger_path = ROOT / "artifacts/download_ledger.json"
    ledger = read_json(ledger_path) if ledger_path.exists() else []
    if args.profile == "sources":
        sources = [
            ("R1","Direct Preference Optimization: Your Language Model is Secretly a Reward Model","Rafailov et al.","https://arxiv.org/abs/2305.18290"),
            ("R2","Original DPO implementation","Eric Mitchell and contributors","https://api.github.com/repos/eric-mitchell/direct-preference-optimization/commits/main"),
            ("R3","TRL DPO Trainer documentation","Hugging Face","https://huggingface.co/docs/trl/v0.24.0/en/dpo_trainer"),
            ("R4","TRL DPO implementation v0.24.0","Hugging Face","https://raw.githubusercontent.com/huggingface/trl/v0.24.0/trl/trainer/dpo_trainer.py"),
            ("R5","Qwen3-0.6B model card","Qwen Team","https://huggingface.co/Qwen/Qwen3-0.6B/raw/main/README.md"),
            ("R6","SmolLM2-135M-Instruct model card","Hugging Face TB","https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/raw/main/README.md"),
            ("R7","UltraFeedback Binarized dataset card","Hugging Face H4","https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized/raw/main/README.md"),
            ("R8","Chat templates","Hugging Face","https://huggingface.co/docs/transformers/v4.57.1/en/chat_templating"),
            ("R9","PyTorch benchmarking","PyTorch","https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html"),
            ("R10","PyTorch profiler","PyTorch","https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html"),
            ("R11","PyTorch reproducibility","PyTorch","https://docs.pytorch.org/docs/2.9/notes/randomness.html"),
            ("R12","PEFT LoRA documentation","Hugging Face","https://huggingface.co/docs/peft/v0.17.0/en/developer_guides/lora"),
            ("R13","LoRA: Low-Rank Adaptation of Large Language Models","Hu et al.","https://arxiv.org/abs/2106.09685"),
            ("R14","UltraFeedback: Boosting Language Models with Scaled AI Feedback","Cui et al.","https://arxiv.org/abs/2310.01377")]
        registry = []
        for key, title, author, url in sources:
            record = {"id": key, "title": title, "author_organization": author, "url": url,
                      "retrieved_at": now(), "status": "supplied_not_retrieved"}
            try:
                data = get(url)
                target = ROOT / "artifacts/sources" / (key + ".txt")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                record.update(status="retrieved", artifact=target.relative_to(ROOT).as_posix(), sha256=file_hash(target))
                if key == "R2":
                    sha = json.loads(data)["sha"]
                    record["revision"] = sha
                    rawurl = f"https://raw.githubusercontent.com/eric-mitchell/direct-preference-optimization/{sha}/trainers.py"
                    target = ROOT / "artifacts/sources/authors-trainers.py"
                    target.write_bytes(get(rawurl))
                    record.update(source_url=rawurl, source_sha256=file_hash(target))
                elif key == "R4":
                    record["revision"] = "v0.24.0; content hash pinned"
            except Exception as e:
                record.update(status="retrieval_failed", error=repr(e))
            registry.append(record)
            print(key, record["status"], flush=True)
        atomic_json(ROOT / "artifacts/source_registry.json", registry)
        return
    cfg = profile(args.profile)
    model = json.loads(get("https://huggingface.co/api/models/" + cfg.model_id))
    revision = model["sha"]
    allow = {"config.json","generation_config.json","tokenizer.json","tokenizer_config.json",
             "special_tokens_map.json","vocab.json","merges.txt","chat_template.jinja","README.md","LICENSE","LICENSE.txt"}
    for sibling in model["siblings"]:
        name = sibling["rfilename"]
        if name in allow or ("/" not in name and (name.endswith(".safetensors") or name.endswith(".safetensors.index.json"))):
            print("Downloading", args.profile, name, flush=True)
            download(f"https://huggingface.co/{cfg.model_id}/resolve/{revision}/{name}",
                     ROOT / ".cache/models" / args.profile / name, ledger, int(args.max_download_gib * 1024**3))
    dataset_id = "HuggingFaceH4/ultrafeedback_binarized"
    dataset = json.loads(get("https://huggingface.co/api/datasets/" + dataset_id))
    drev = dataset["sha"]
    for sibling in dataset["siblings"]:
        name = sibling["rfilename"]
        if name.endswith(".parquet") and any(k in name for k in ["train_prefs", "test_prefs"]):
            print("Downloading dataset", name, flush=True)
            download(f"https://huggingface.co/datasets/{dataset_id}/resolve/{drev}/{name}",
                     ROOT / ".cache/data" / name, ledger, int(args.max_download_gib * 1024**3))
    cfg.revision, cfg.dataset_revision = revision, drev
    atomic_json(ROOT / "artifacts" / f"{args.profile}-resolved.json", cfg.to_dict())
    atomic_json(ROOT / "artifacts" / f"{args.profile}-public-metadata.json", {"model": model, "dataset": dataset})
    print("Resolved", revision, drev, flush=True)


if __name__ == "__main__":
    main()
