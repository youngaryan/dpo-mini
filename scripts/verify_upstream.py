"""Version-specific TRL adapter. Execute installed methods without Trainer setup."""
import _common
import copy
import inspect
from types import SimpleNamespace
import torch
from dpo_mini.provenance import ROOT, atomic_json, file_hash, now, seed_all
from dpo_mini.loss import dpo_loss
from dpo_mini.scoring import model_scores
from dpo_mini.tokenization import collate


def difference(a, b):
    a,b=a.detach(),b.detach()
    return {"max_absolute": float((a-b).abs().max()),
            "max_relative_floor_1e-12": float(((a-b).abs()/b.abs().clamp_min(1e-12)).max())}


def main():
    import trl
    from trl import DPOTrainer
    from trl.trainer.dpo_config import FDivergenceType
    from transformers import GPT2Config, GPT2LMHeadModel
    seed_all(42, 2)
    if trl.__version__ != "0.24.0":
        raise RuntimeError("This adapter is pinned to TRL 0.24.0; inspect a new release before adapting")
    trainer = object.__new__(DPOTrainer)
    attributes = dict(accelerator=SimpleNamespace(device=torch.device("cpu")), reference_free=False,
        f_divergence_type=FDivergenceType.REVERSE_KL, f_divergence_params={}, beta=.1, label_smoothing=0.,
        pad_token_id=0, label_pad_token_id=-100, is_encoder_decoder=False, aux_loss_enabled=False,
        max_length=None, truncation_mode="keep_start", use_logits_to_keep=False, padding_free=False,
        use_weighting=False, loss_type=["sigmoid"], args=SimpleNamespace(rpo_alpha=None, ld_alpha=None))
    for k,v in attributes.items():
        setattr(trainer,k,v)
    a=torch.tensor([-2.,-5.,-1.],dtype=torch.double,requires_grad=True)
    b=torch.tensor([-4.,-4.,-2.],dtype=torch.double,requires_grad=True)
    r1=torch.tensor([-3.,-4.1,-1.3],dtype=torch.double)
    r2=torch.tensor([-3.,-4.9,-2.1],dtype=torch.double)
    ours=dpo_loss(a,b,r1,r2)
    upstream=trainer.dpo_loss(a,b,r1,r2,loss_type="sigmoid")[0].mean()
    ga=torch.cat(torch.autograd.grad(ours,(a,b),retain_graph=True))
    gb=torch.cat(torch.autograd.grad(upstream,(a,b)))
    report={"status":"completed", "scope":"installed TRL sigmoid loss, completion scores and tiny-model parameter gradients; not full Trainer parity",
            "version":trl.__version__,"recorded_at":now(),"source_file_sha256":file_hash(__import__('pathlib').Path(inspect.getfile(DPOTrainer))),
            "score_loss_error":difference(ours,upstream),"score_gradient_error":difference(ga,gb),
            "tolerances":{"float64_loss":1e-12,"float64_gradient":1e-11,"fp32_scores":2e-5,"fp32_parameter_gradients":2e-6}}
    # Locally initialized GPT-2 fixture, same weights/tokens/masks/positions, no downloads.
    config=GPT2Config(vocab_size=32,n_positions=32,n_embd=32,n_layer=2,n_head=4,
                      resid_pdrop=0.,embd_pdrop=0.,attn_pdrop=0.,use_cache=False)
    config._attn_implementation="eager"
    model=GPT2LMHeadModel(config).eval()
    other=copy.deepcopy(model)
    pairs=[{"id":"one","chosen_ids":[1,4,6,7,2],"rejected_ids":[1,4,9,2],
            "chosen_mask":[0,0,1,1,1],"rejected_mask":[0,0,1,1]},
           {"id":"two","chosen_ids":[1,3,5,2],"rejected_ids":[1,3,8,9,2],
            "chosen_mask":[0,0,1,1],"rejected_mask":[0,0,1,1,1]}]
    trl_batch={"prompt_input_ids":torch.tensor([[1,4],[1,3]]), "prompt_attention_mask":torch.ones(2,2,dtype=torch.long),
               "chosen_input_ids":torch.tensor([[6,7,2],[5,2,0]]),"chosen_attention_mask":torch.tensor([[1,1,1],[1,1,0]]),
               "rejected_input_ids":torch.tensor([[9,2,0],[8,9,2]]),"rejected_attention_mask":torch.tensor([[1,1,0],[1,1,1]])}
    scores=model_scores(model,collate(pairs))
    result=trainer.concatenated_forward(other,trl_batch)
    scores2=torch.cat([result["chosen_logps"],result["rejected_logps"]])
    fixed=torch.tensor([-6.,-5.,-5.,-6.])
    l1=dpo_loss(scores[:2],scores[2:],fixed[:2],fixed[2:])
    l2=trainer.dpo_loss(scores2[:2],scores2[2:],fixed[:2],fixed[2:],loss_type="sigmoid")[0].mean()
    g1=torch.cat([g.reshape(-1) for g in torch.autograd.grad(l1,tuple(model.parameters()))])
    g2=torch.cat([g.reshape(-1) for g in torch.autograd.grad(l2,tuple(other.parameters()))])
    report.update(completion_score_error=difference(scores,scores2), model_loss_error=difference(l1,l2),
                  model_parameter_gradient_error=difference(g1,g2), tiny_model="local GPT2 2 layers / width32 / heads4; FP32 eager")
    assert report["score_loss_error"]["max_absolute"] < 1e-12
    assert report["score_gradient_error"]["max_absolute"] < 1e-11
    assert report["completion_score_error"]["max_absolute"] < 2e-5
    assert report["model_parameter_gradient_error"]["max_absolute"] < 2e-6
    atomic_json(ROOT/"artifacts/upstream.json",report)
    print(report)


if __name__=="__main__":
    try:
        main()
    except Exception as error:
        atomic_json(ROOT/"artifacts/upstream.json",{"status":"failed","error":repr(error),"recorded_at":now()})
        raise
