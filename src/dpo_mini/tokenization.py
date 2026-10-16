from __future__ import annotations
import torch


class ByteTokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    vocab_size = 259

    def encode(self, text: str) -> list[int]:
        return [x + 3 for x in text.encode("utf-8")]

    def decode(self, ids: list[int]) -> str:
        return bytes([x - 3 for x in ids if 3 <= x <= 258]).decode("utf-8", errors="replace")


def tokenize_pair(row: dict, tokenizer, max_length: int, qwen: bool = False) -> dict:
    if isinstance(tokenizer, ByteTokenizer):
        prompt = [1] + tokenizer.encode(row["prompt"] + "\nAnswer:")
        encode = tokenizer.encode
    else:
        kwargs = {"enable_thinking": False} if qwen else {}
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": row["prompt"]}],
                                              tokenize=True, add_generation_prompt=True, **kwargs)
        encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    # Explicit boundary by construction. Completion tokens are suffix-encoded;
    # a whole-chat render is NOT assumed to have the same token prefix.
    result = {k: row[k] for k in ["id", "prompt_hash", "prompt", "chosen", "rejected"]}
    result["prompt_ids"] = prompt
    for label in ["chosen", "rejected"]:
        content = encode(row[label])
        if not content:
            raise ValueError("empty response")
        if tokenizer.eos_token_id in content:
            raise ValueError("embedded response-ending token")
        ids = prompt + content + [tokenizer.eos_token_id]
        if len(ids) > max_length:
            raise ValueError("length cap")
        result[label + "_ids"] = ids
        result[label + "_mask"] = [0] * len(prompt) + [1] * (len(content) + 1)
    return result


def collate(pairs: list[dict], pad_id: int = 0, device: str = "cpu") -> dict:
    if not pairs:
        raise ValueError("Cannot collate zero preference pairs")
    sequences = [(p[k + "_ids"], p[k + "_mask"]) for k in ["chosen", "rejected"] for p in pairs]
    length = max(len(ids) for ids, _ in sequences)
    ids = torch.full((len(sequences), length), pad_id, dtype=torch.long, device=device)
    att = torch.zeros_like(ids)
    mask = torch.zeros_like(ids)
    for i, (tokens, scored) in enumerate(sequences):
        if len(tokens) != len(scored):
            raise ValueError("Token and scoring-mask lengths differ")
        ids[i, :len(tokens)] = torch.tensor(tokens, device=device)
        att[i, :len(tokens)] = 1
        mask[i, :len(tokens)] = torch.tensor(scored, device=device)
    return {"input_ids": ids, "attention_mask": att, "completion_mask": mask,
            "pair_ids": [p["id"] for p in pairs], "pair_count": len(pairs)}
