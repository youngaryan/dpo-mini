import copy
import math
import json
import torch
import pytest
from dpo_mini.config import profile
from dpo_mini.data import synthetic, extract_single_turn
from dpo_mini.loss import dpo_loss
from dpo_mini.models import make_model
from dpo_mini.provenance import seed_all, state_hash, digest
from dpo_mini.reference import build_cache, cache_context, ReferenceCache
from dpo_mini.scoring import completion_scores, model_scores
from dpo_mini.tokenization import ByteTokenizer, collate
from dpo_mini.train import Trainer


@pytest.fixture(autouse=True)
def deterministic():
    seed_all(7, 2)


def fixture():
    cfg = profile("toy", train_pairs=19, validation_pairs=2, test_pairs=2, effective_batch_pairs=8, threads=2)
    data, _ = synthetic(cfg)
    return cfg, data["train"], make_model(cfg), make_model(cfg, True)


def test_equal_and_analytic_gradient():
    pc = torch.tensor([-3.], dtype=torch.double, requires_grad=True)
    pr = torch.tensor([-3.], dtype=torch.double, requires_grad=True)
    ref = torch.tensor([-3.], dtype=torch.double)
    loss = dpo_loss(pc, pr, ref, ref)
    assert loss.item() == pytest.approx(math.log(2), abs=1e-15)
    g = torch.autograd.grad(loss, (pc, pr))
    assert g[0].item() == pytest.approx(-0.05)
    assert g[1].item() == pytest.approx(0.05)
    actual = dpo_loss(torch.tensor([-2.], dtype=torch.double), torch.tensor([-4.], dtype=torch.double), ref, ref)
    assert actual.item() == pytest.approx(math.log1p(math.exp(-0.2)), abs=1e-15)


def test_gradcheck():
    pc = torch.randn(4, dtype=torch.double, requires_grad=True)
    pr = torch.randn(4, dtype=torch.double, requires_grad=True)
    r = torch.randn(4, dtype=torch.double)
    assert torch.autograd.gradcheck(lambda a, b: dpo_loss(a, b, r, r, .1), (pc, pr), eps=1e-6, atol=1e-7, rtol=1e-5)


@pytest.mark.parametrize("beta", [0, -1, float("nan"), float("inf")])
def test_invalid_beta(beta):
    with pytest.raises(ValueError, match="beta"):
        dpo_loss(*[torch.zeros(1)] * 4, beta=beta)


def test_hand_calculated_shift_and_ending():
    # Prompt IDs [0,1], response [2,0]; first response predicted from position 1.
    probs = torch.tensor([[[.2,.3,.5],[.1,.2,.7],[.6,.3,.1],[.1,.8,.1]]], dtype=torch.double)
    ids = torch.tensor([[0,1,2,0]])
    mask = torch.tensor([[0,0,1,1]])
    score = completion_scores(probs.log(), ids, torch.ones_like(ids), mask)
    assert score.item() == pytest.approx(math.log(.7) + math.log(.6), abs=1e-14)


def test_padding_alias_invalid_masked_gather_and_lengths():
    logits = torch.zeros(2, 5, 4, dtype=torch.double, requires_grad=True)
    ids = torch.tensor([[1,2,3,0,0],[1,2,0,0,0]])  # EOS=pad=0, one real end per row
    attention = torch.tensor([[1,1,1,1,0],[1,1,1,0,0]])
    mask = torch.tensor([[0,0,1,1,0],[0,0,1,0,0]])
    expected = torch.tensor([-2 * math.log(4), -math.log(4)], dtype=torch.double)
    scores = completion_scores(logits, ids, attention, mask)
    assert torch.allclose(scores, expected, atol=1e-14)
    padded_ids = torch.cat([ids, torch.full((2,2), -100)], 1)
    padded_logits = torch.cat([logits, torch.full((2,2,4), float("nan"))], 1)
    zeros = torch.zeros((2,2), dtype=torch.long)
    assert torch.equal(scores, completion_scores(padded_logits, padded_ids, torch.cat([attention,zeros],1), torch.cat([mask,zeros],1)))
    with pytest.raises(ValueError, match="nonempty"):
        completion_scores(logits, ids, attention, torch.zeros_like(ids))


def test_causal_mask_padding_and_prompt_gradients():
    cfg, pairs, model, _ = fixture()
    batch = collate(pairs[:1])
    original = model(batch["input_ids"], batch["attention_mask"])
    changed = batch["input_ids"].clone()
    changed[:, -1] = 88
    altered = model(changed, batch["attention_mask"])
    assert torch.allclose(original[:, :-1], altered[:, :-1], atol=1e-7, rtol=0)
    score = model_scores(model, batch)
    pad = lambda x, n: torch.cat([x, torch.full((len(x),3), n, dtype=x.dtype)],1)
    padded = {"input_ids": pad(batch["input_ids"], 0), "attention_mask": pad(batch["attention_mask"],0),
              "completion_mask": pad(batch["completion_mask"],0)}
    assert torch.allclose(score, model_scores(model, padded), atol=2e-5, rtol=0)
    score.sum().backward()
    assert model.token.weight.grad[1].norm() > 0  # BOS only occurs in the conditioning prompt


def test_frozen_cache_loss_gradient_equivalence(tmp_path):
    cfg, pairs, policy, reference = fixture()
    h = state_hash(reference)
    context = cache_context(cfg, h, ByteTokenizer())
    cache = build_cache(tmp_path / "cache.json", reference, pairs, context)
    batch = collate(pairs[:1])
    a = model_scores(reference, batch)
    b = cache.scores(pairs[:1], "cpu")
    assert torch.equal(a, b)
    p = model_scores(policy, batch)
    l1 = dpo_loss(p[:1], p[1:], a[:1], a[1:])
    l2 = dpo_loss(p[:1], p[1:], b[:1], b[1:])
    g1 = torch.autograd.grad(l1, tuple(policy.parameters()), retain_graph=True)
    g2 = torch.autograd.grad(l2, tuple(policy.parameters()))
    assert all(torch.equal(x,y) for x,y in zip(g1,g2))
    before = state_hash(policy)
    trainer = Trainer(policy, reference, None, cfg, pairs, 0)
    trainer.run(max_updates=1)
    assert state_hash(policy) != before and state_hash(reference) == h
    assert torch.equal(a, model_scores(reference, batch))
    assert all(p.grad is None and not p.requires_grad for p in reference.parameters())
    cached_trainer = Trainer(policy, None, cache, cfg, pairs, 0)
    cached_trainer.run(max_updates=1)
    assert cached_trainer.reference_forward_calls == 0


def test_cache_integrity_and_reorder(tmp_path):
    cfg, pairs, _, ref = fixture()
    context = cache_context(cfg, state_hash(ref), ByteTokenizer())
    path = tmp_path / "cache.json"
    cache = build_cache(path, ref, pairs, context)
    reversed_cache = ReferenceCache.load(path, pairs[::-1], context)
    assert torch.equal(cache.scores([pairs[0]], "cpu"), reversed_cache.scores([pairs[0]], "cpu"))
    with pytest.raises(ValueError):
        ReferenceCache.load(path, pairs, {**context, "ending_id": 3})
    altered = copy.deepcopy(pairs)
    altered[0]["chosen_ids"][-1] = 3
    with pytest.raises(ValueError):
        ReferenceCache.load(path, altered, context)
    with pytest.raises(ValueError):
        ReferenceCache.load(path, pairs[:-1], context)
    with pytest.raises(ValueError):
        ReferenceCache.load(path, pairs + [pairs[0]], context)
    payload = json.loads(path.read_text())
    duplicate = copy.deepcopy(payload)
    duplicate['rows'][1] = copy.deepcopy(duplicate['rows'][0])
    duplicate['sha256'] = digest({'context': context, 'rows': duplicate['rows']})
    with pytest.raises(ValueError, match='duplicate'):
        ReferenceCache(duplicate, pairs, context)
    missing = copy.deepcopy(payload)
    missing['rows'].pop()
    missing['sha256'] = digest({'context': context, 'rows': missing['rows']})
    with pytest.raises(ValueError, match='Missing'):
        ReferenceCache(missing, pairs, context)
    payload["rows"][0]["chosen"] += 1
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="checksum"):
        ReferenceCache.load(path, pairs, context)
    path.write_text('{"version":')
    with pytest.raises(ValueError):
        ReferenceCache.load(path, pairs, context)
    with pytest.raises(ValueError, match="interrupted"):
        ReferenceCache.load(tmp_path / "pending.tmp", pairs, context)


def test_accumulation_partial_window_matches_whole_batch():
    cfg, pairs, model, ref = fixture()
    # Double precision isolates accumulation arithmetic from FP32 batched GEMM
    # noise, which Adam can amplify for near-zero attention-bias gradients.
    model.double()
    ref.double()
    direct = copy.deepcopy(model)
    c2 = copy.deepcopy(cfg)
    c2.microbatch_pairs = 8
    a = Trainer(model, ref, None, cfg, pairs, 0)
    b = Trainer(direct, ref, None, c2, pairs, 0)
    a.run(); b.run()
    assert [x["pairs"] for x in a.trace] == [8,8,3]
    assert a.position == 19
    assert max((p-q).abs().max().item() for p,q in zip(model.parameters(),direct.parameters())) < 1e-10


def test_resume_matches_uninterrupted_and_rejects_mismatch(tmp_path):
    cfg, pairs, model, ref = fixture()
    full = Trainer(model, ref, None, cfg, pairs, 0)
    full.run()
    stopped = Trainer(make_model(cfg), ref, None, cfg, pairs, 0)
    stopped.run(max_updates=1)
    path = tmp_path / "resume.pt"
    identity = {"config": cfg.to_dict(), "data": digest(pairs)}
    stopped.save(path, identity)
    resumed = Trainer(make_model(cfg), ref, None, cfg, pairs, 0)
    with pytest.raises(ValueError, match="mismatch"):
        resumed.restore(path, {})
    resumed.restore(path, identity)
    resumed.run()
    assert all(torch.equal(p,q) for p,q in zip(full.model.parameters(),resumed.model.parameters()))
    assert full.trace == resumed.trace


def test_data_splits_and_score_filters():
    cfg = profile("toy")
    _, manifest = synthetic(cfg)
    assert [manifest["splits"][k]["count_pairs"] for k in ["train","validation","test"]] == [64,16,16]
    row = {"chosen": [{"role":"user","content":"hi"},{"role":"assistant","content":"yes"}],
           "rejected": [{"role":"user","content":"hi"},{"role":"assistant","content":"no"}],
           "score_chosen": 1, "score_rejected": 1}
    with pytest.raises(ValueError, match="non-strict"):
        extract_single_turn(row)
    row["score_chosen"] = 2
    assert extract_single_turn(row)["chosen"] == "yes"
