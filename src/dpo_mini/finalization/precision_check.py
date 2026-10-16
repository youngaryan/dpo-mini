"""Predeclared diagnosis: FP32 model, FP64 score reducers, actual installed TRL."""
import torch
from ..config import Config
from ..models import make_model,get_tokenizer
from ..provenance import ROOT,read_json,atomic_json,file_hash
from .parity import upstream,compare
from .strategy import restore_adapter


def run(output):
    output.mkdir(parents=True,exist_ok=True)
    source=output.parent/'parity'
    cfg=Config(**read_json(ROOT/'artifacts/prepared/compact/config.json'))
    rows=read_json(ROOT/'artifacts/prepared/compact/data.json')['train'][:4]
    atomic_json(output/'declared-checks.json',{'purpose':'Isolate FP32 score-reduction differences; production code unchanged',
        'score_tolerance':1e-10,'loss_tolerance':1e-12,'fp32_weight_gradient_tolerance':2e-6,
        'same_training_ids':[r['id'] for r in rows],'model_weights_and_logits_compute':'FP32',
        'only_intervention':'Cast identical returned logits to FP64 before each unmodified scoring routine'})
    tok=get_tokenizer(cfg);model=make_model(cfg);ref=make_model(cfg,reference=True);trainer,_=upstream(tok.pad_token_id)
    results={}
    for name,fixture in [('identity','initial-adapter.pt'),('nontrivial','nontrivial-adapter.pt')]:
        restore_adapter(model,torch.load(source/fixture,map_location='cpu',weights_only=True))
        r=compare(model,ref,rows,tok,trainer,logit_dtype=torch.float64)
        assert r['max_score_error']<=1e-10 and r['max_loss_error']<=1e-12 and r['max_gradient_error']<=2e-6
        results[name]=r
    atomic_json(output/'result.json',{'status':'passed','scope':'Diagnostic FP64 reductions on FP32 pretrained logits and weights; not full FP64 training',
        'results':results,'original_parity_sha256':file_hash(source/'result.json')})
    return {'status':'passed'}
