from __future__ import annotations
from dataclasses import asdict, dataclass


@dataclass
class Config:
    profile: str = "toy"
    model_id: str = "local/tiny-byte-transformer-v1"
    revision: str = "local-v1"
    dataset_revision: str = "synthetic-v1"
    train_pairs: int = 64
    validation_pairs: int = 16
    test_pairs: int = 16
    max_length: int = 128
    microbatch_pairs: int = 1
    effective_batch_pairs: int = 16
    beta: float = 0.1
    learning_rate: float = 0.001
    lora_rank: int = 16
    lora_alpha: int = 32
    seed: int = 42
    split_seed: int = 314159
    threads: int = 4
    device: str = "cpu"
    weight_dtype: str = "float32"
    compute_dtype: str = "float32"
    reduction_dtype: str = "float32"
    attention_backend: str = "eager"
    max_experiment_hours: float = 6.0
    max_subprocess_minutes: float = 90.0
    max_download_gib: float = 10.0
    max_storage_gib: float = 20.0
    tie_tolerance: float = 1e-6
    score_atol: float = 2e-5
    gradient_atol: float = 2e-6

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> None:
        import math
        if not math.isfinite(self.beta) or self.beta <= 0:
            raise ValueError("beta must be finite and positive")
        if self.effective_batch_pairs < 1 or self.microbatch_pairs < 1:
            raise ValueError("batch sizes are preference-pair counts and must be positive")
        if self.effective_batch_pairs % self.microbatch_pairs:
            raise ValueError("effective batch must be divisible by microbatch")


def profile(name: str, **overrides) -> Config:
    values = {}
    if name == "compact":
        values = dict(model_id="HuggingFaceTB/SmolLM2-135M-Instruct", train_pairs=256,
                      validation_pairs=64, test_pairs=64, max_length=256, learning_rate=5e-5)
    elif name == "standard":
        values = dict(model_id="Qwen/Qwen3-0.6B", train_pairs=2000,
                      validation_pairs=300, test_pairs=300, max_length=512, learning_rate=5e-5)
    elif name != "toy":
        raise ValueError(f"Unknown profile {name}")
    values.update(overrides)
    cfg = Config(profile=name, **values)
    cfg.validate()
    return cfg
