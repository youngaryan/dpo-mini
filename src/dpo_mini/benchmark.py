"""Fresh-process worker with explicit cache/setup/training/evaluation boundaries."""
from __future__ import annotations
import copy
import gc
import os
import random
import threading
import time
from pathlib import Path
import psutil
import torch
from .config import Config
from .evaluate import evaluate, generate
from .loss import dpo_loss
from .models import make_model, get_tokenizer
from .provenance import ROOT, atomic_json, digest, now, read_json, source_manifest, state_hash
from .reference import ReferenceCache, build_cache, cache_context
from .scoring import model_scores
from .tokenization import collate
from .train import Trainer, synchronize


class MemorySampler:
    """RSS samples every 10ms; Windows process peak is also reported separately."""
    def __init__(self):
        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        self.minimum_available = psutil.virtual_memory().available
        self.done = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()

    def sample(self):
        while not self.done.wait(.01):
            self.peak = max(self.peak, self.process.memory_info().rss)
            self.minimum_available = min(self.minimum_available, psutil.virtual_memory().available)

    def finish(self):
        self.done.set()
        self.thread.join()
        info = self.process.memory_info()
        return {"sampled_peak_rss_bytes": max(self.peak, info.rss),
                "os_process_peak_rss_bytes": getattr(info, "peak_wset", None),
                "minimum_available_ram_bytes": self.minimum_available, "sampling_interval_seconds": .01}


def gpu_peak(device: str) -> dict:
    if not device.startswith("cuda"):
        return {"allocated_bytes": None, "reserved_bytes": None, "reason": "CUDA unavailable/not used"}
    return {"allocated_bytes": torch.cuda.max_memory_allocated(), "reserved_bytes": torch.cuda.max_memory_reserved()}


def run_worker(spec: dict, run_dir: Path) -> dict:
    start = time.perf_counter()
    cfg = Config(**spec["config"])
    cfg.validate()
    cfg.seed = spec.get("seed", cfg.seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(run_dir / "resolved_config.json", spec)
    provenance = {"started_at": now(), "status": "running", "source": source_manifest(),
                  "pid":os.getpid(),"process_create_time":psutil.Process().create_time(),
                  "parent_plan_hash": spec.get("plan_hash"), "seed": cfg.seed,
                  "model_revision": cfg.revision, "dataset_revision": cfg.dataset_revision}
    atomic_json(run_dir / "provenance.json", provenance)
    whole_memory = MemorySampler()
    if cfg.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    try:
        data = read_json(ROOT / spec["data_path"])
        data_manifest = read_json((ROOT / spec["data_path"]).parent / "manifest.json")
        for split,rows in data.items():
            if digest(rows)!=data_manifest["splits"][split]["split_hash"]:
                raise ValueError("Tokenized split hash differs from prepared manifest")
        provenance["data_sha256"] = digest(data)
        kind, arm = spec["kind"], spec["arm"]
        train = list(data["train"])
        random.Random(cfg.seed).shuffle(train)
        if kind == "pilot":
            ordered = sorted(train, key=lambda p: max(len(p[k+"_ids"]) for k in ["chosen","rejected"]))
            train = [ordered[round(i*(len(ordered)-1)/7)] for i in range(8)] if len(train) >= 8 else ordered
        all_pairs = train + data["validation"] + data["test"] if kind == "learning" else train
        # Bench repetitions reuse IDs intentionally; the cache itself contains unique IDs.
        tokenizer = get_tokenizer(cfg)
        policy = make_model(cfg)
        initial_hash = state_hash(policy)
        context = cache_context(cfg, initial_hash, tokenizer)
        parameters = {"total": sum(p.numel() for p in policy.parameters()),
                      "trainable": sum(p.numel() for p in policy.parameters() if p.requires_grad)}
        reference = cache = None
        cache_seconds = 0.0
        precompute_seconds = 0.0
        reference_load_seconds = 0.0
        reference_hash = initial_hash
        cache_path = run_dir / "reference-cache.json"
        cache_loaded_from = None
        if arm == "uncached_reference":
            t = time.perf_counter()
            reference = make_model(cfg, reference=True)
            if state_hash(reference) != initial_hash:
                raise AssertionError("Initial policy and frozen reference states differ")
            reference_load_seconds = time.perf_counter() - t
        else:
            t = time.perf_counter()
            if spec.get("warm_cache"):
                cache_path = ROOT / spec["warm_cache"]
                cache = ReferenceCache.load(cache_path, all_pairs, context)
                cache_loaded_from = str(cache_path.relative_to(ROOT))
            else:
                reference = make_model(cfg, reference=True)
                if state_hash(reference) != initial_hash:
                    raise AssertionError("Reference initialization differs")
                reference_load_seconds = time.perf_counter() - t
                precompute_start = time.perf_counter()
                cache = build_cache(cache_path, reference, all_pairs, context, tokenizer.pad_token_id,
                                    cfg.device, cfg.microbatch_pairs)
                synchronize(cfg.device)
                precompute_seconds = time.perf_counter() - precompute_start
                reference = None
                gc.collect()
                if cfg.device.startswith("cuda"):
                    torch.cuda.empty_cache()
            synchronize(cfg.device)
            cache_seconds = time.perf_counter() - t
        setup_seconds = time.perf_counter() - start
        eval_initial = samples_initial = None
        evaluation_seconds = 0.0
        if kind == "learning":
            t = time.perf_counter()
            eval_initial = {s: evaluate(policy, reference, cache, data[s], cfg, tokenizer.pad_token_id)
                            for s in ["validation","test"]}
            samples_initial = generate(policy, tokenizer, data["test"], cfg)
            synchronize(cfg.device)
            evaluation_seconds += time.perf_counter() - t
        correctness = {}
        if kind == "pilot":
            batch = collate(train[:1], tokenizer.pad_token_id, cfg.device)
            p = model_scores(policy, batch)
            if reference is not None:
                with torch.no_grad():
                    r = model_scores(reference, batch)
            else:
                r = cache.scores(train[:1], cfg.device)
            correctness["initial_score_max_abs_error"] = float((p.detach()-r).abs().max())
            assert torch.allclose(p.detach(), r, atol=cfg.score_atol, rtol=0)
            # Reference identity: initial LoRA output must equal disabled-adapter base.
            if cfg.profile != "toy":
                with policy.disable_adapter(), torch.no_grad():
                    base = model_scores(policy,batch)
                correctness["initial_lora_vs_base_max_abs_error"] = float((p.detach()-base).abs().max())
                assert torch.allclose(p.detach(),base,atol=cfg.score_atol,rtol=0)
            del p, r
        if kind == "benchmark":
            total = spec["warmup_updates"] + spec["measured_updates"]
            repeats = (total * cfg.effective_batch_pairs + len(train)-1)//len(train)
            work = (train * repeats)[:total * cfg.effective_batch_pairs]
        elif kind == "pilot":
            work = train
            total = 1
        else:
            work = train
            total = (len(work)+cfg.effective_batch_pairs-1)//cfg.effective_batch_pairs
        trainer = Trainer(policy, reference, cache, cfg, work, tokenizer.pad_token_id, total)
        def resource_guard():
            if psutil.virtual_memory().available < 1024**3:
                raise MemoryError("Stopping own experiment to preserve 1 GiB available RAM")
        trainer.guard = resource_guard
        trainer.progress_callback = lambda t: atomic_json(run_dir/"progress.json",{
            "status":"running","completed_updates":t.update_index,"completed_pairs":t.position,
            "trace":t.trace,"note":"Only completed optimizer updates; interrupted accumulation windows are not completed training"})
        identity = {"config": cfg.to_dict(),"data":digest(work),"reference":reference_hash,"plan":spec.get("plan_hash")}
        checkpoint = run_dir / "checkpoints/final.pt"
        if spec.get("resume_checkpoint"):
            trainer.restore(ROOT / spec["resume_checkpoint"],identity)
        if kind == "benchmark":
            trainer.run(max_updates=spec["warmup_updates"])
        synchronize(cfg.device)
        whole_gpu_before_training = gpu_peak(cfg.device)
        if cfg.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        training_memory = MemorySampler()
        t = time.perf_counter()
        trace_start = len(trainer.trace)
        trainer.run(deadline=time.monotonic()+cfg.max_subprocess_minutes*60)
        synchronize(cfg.device)
        training_seconds = time.perf_counter() - t
        memory = training_memory.finish()
        training_gpu = gpu_peak(cfg.device)
        trace = trainer.trace[trace_start:]
        measured_pairs = sum(r["pairs"] for r in trace)
        measured_tokens = sum(r["response_tokens"] for r in trace)
        final_eval = final_samples = None
        checkpoint_seconds = 0.0
        if kind == "learning":
            t = time.perf_counter()
            trainer.save(checkpoint, identity)
            checkpoint_seconds = time.perf_counter() - t
            t = time.perf_counter()
            final_eval = {s: evaluate(policy, reference, cache, data[s], cfg, tokenizer.pad_token_id)
                          for s in ["validation","test"]}
            final_samples = generate(policy, tokenizer, data["test"], cfg)
            evaluation_seconds += time.perf_counter() - t
        if reference is not None and state_hash(reference) != reference_hash:
            raise AssertionError("Frozen reference changed after optimization")
        if cache is not None and trainer.reference_forward_calls != 0:
            raise AssertionError("Cached training performed a reference forward")
        final_hash = state_hash(policy)
        synchronize(cfg.device)
        whole_stats = whole_memory.finish()
        gpu_after = gpu_peak(cfg.device)
        whole_gpu = {key:max(whole_gpu_before_training[key],gpu_after[key]) if gpu_after.get(key) is not None else None
                     for key in ["allocated_bytes","reserved_bytes"]}
        metrics = {"status":"completed","kind":kind,"arm":arm,"seed":cfg.seed,"profile":cfg.profile,
                   "workflow_seconds_excluding_imports":time.perf_counter()-start,"setup_including_cache_seconds":setup_seconds,
                   "cache_build_or_load_seconds":cache_seconds,"cache_is_warm":bool(spec.get("warm_cache")),
                   "reference_load_and_identity_seconds":reference_load_seconds,"reference_precompute_seconds":precompute_seconds,
                   "cache_path":cache_path.relative_to(ROOT).as_posix() if cache else None,
                   "cache_loaded_from":cache_loaded_from,"training_seconds":training_seconds,
                   "evaluation_and_generation_seconds":evaluation_seconds,"checkpoint_seconds":checkpoint_seconds,
                   "pairs":measured_pairs,"response_tokens":measured_tokens,"pairs_per_second":measured_pairs/training_seconds,
                   "response_tokens_per_second":measured_tokens/training_seconds,"training_memory":memory,
                   "whole_workflow_memory":whole_stats,"training_gpu":training_gpu,"whole_workflow_gpu":whole_gpu,
                   "whole_workflow_gpu_pretraining":whole_gpu_before_training,"parameters":parameters,
                   "reference_training_forward_calls":trainer.reference_forward_calls,
                   "policy_training_forward_calls":trainer.policy_forward_calls,"initial_state_sha256":initial_hash,
                   "final_state_sha256":final_hash,"updates":trainer.update_index,"trace":trainer.trace,
                   "initial_evaluation":eval_initial,"final_evaluation":final_eval,"initial_generations":samples_initial,
                   "final_generations":final_samples,"correctness":correctness,
                   "checkpoint":checkpoint.relative_to(ROOT).as_posix() if kind=="learning" else None,
                   "throughput_definition":"preference pairs and chosen+rejected scored response tokens per measured training block",
                   "training_scope":"collation, transfers, cache lookup/reference forward, policy forward/backward, clipping, AdamW and one local progress JSON per update; excludes warmup"}
        atomic_json(run_dir/"metrics.json",metrics)
        atomic_json(run_dir/"progress.json",{"status":"completed","completed_updates":trainer.update_index,"completed_pairs":trainer.position,"trace":trainer.trace})
        provenance.update(status="completed",ended_at=now(),exit_status=0)
        atomic_json(run_dir/"provenance.json",provenance)
        return metrics
    except BaseException as e:
        whole_memory.finish()
        provenance.update(status="stopped by budget" if isinstance(e,TimeoutError) else "failed",ended_at=now(),error=repr(e),exit_status=1)
        atomic_json(run_dir/"provenance.json",provenance)
        raise
