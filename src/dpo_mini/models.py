from __future__ import annotations
import torch
from torch import nn
from .provenance import ROOT, seed_all


class TinyCausalTransformer(nn.Module):
    """Two layers, width 64, four heads; ordinary eager attention, zero dropout."""
    def __init__(self):
        super().__init__()
        self.token = nn.Embedding(259, 64)
        self.position = nn.Embedding(128, 64)
        layer = nn.TransformerEncoderLayer(64, 4, dim_feedforward=128, dropout=0.0,
                                           batch_first=True, norm_first=True)
        self.layers = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(64)
        self.head = nn.Linear(64, 259, bias=False)
        self.forward_calls = 0

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        n = input_ids.shape[1]
        if n > 128:
            raise ValueError("Toy model context is 128 tokens")
        positions = attention_mask.cumsum(-1).sub(1).clamp_min(0)
        x = self.token(input_ids) + self.position(positions)
        causal = torch.ones(n, n, dtype=torch.bool, device=x.device).triu(1)
        x = self.layers(x, mask=causal, src_key_padding_mask=~attention_mask.bool())
        return self.head(self.norm(x))


def make_model(cfg, reference: bool = False):
    seed_all(cfg.seed, cfg.threads)
    if cfg.profile == "toy":
        model = TinyCausalTransformer()
    else:
        from transformers import AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model
        model = AutoModelForCausalLM.from_pretrained(
            str(ROOT / ".cache" / "models" / cfg.profile), revision=cfg.revision,
            local_files_only=True, trust_remote_code=False, use_safetensors=True,
            dtype=getattr(torch, cfg.weight_dtype), attn_implementation=cfg.attention_backend)
        targets = [name for name, module in model.named_modules()
                   if name.endswith(("q_proj", "v_proj")) and isinstance(module, nn.Linear)]
        if not targets:
            raise ValueError("No verified q_proj/v_proj LoRA targets")
        model = get_peft_model(model, LoraConfig(r=cfg.lora_rank, lora_alpha=cfg.lora_alpha,
                      lora_dropout=0.0, bias="none", target_modules=["q_proj", "v_proj"],
                      task_type="CAUSAL_LM", init_lora_weights=True))
        model.config.use_cache = False
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
    model.to(cfg.device)
    if reference:
        model.requires_grad_(False)
    model.eval()  # autograd stays enabled for the policy; all dropout is disabled
    return model


def get_tokenizer(cfg):
    from .tokenization import ByteTokenizer
    if cfg.profile == "toy":
        return ByteTokenizer()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / ".cache" / "models" / cfg.profile), revision=cfg.revision,
                local_files_only=True, trust_remote_code=False)
    if tokenizer.eos_token != "<|im_end|>":
        raise ValueError(f"Inspect unrecognized assistant ending {tokenizer.eos_token!r}")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer
