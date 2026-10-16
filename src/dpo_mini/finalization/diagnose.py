"""Diagnose a recorded tolerance failure without changing either production scorer."""
import inspect
import torch
from ..config import Config
from ..models import make_model,get_tokenizer
from ..provenance import ROOT,read_json,atomic_json,file_hash
from ..scoring import completion_scores
from ..tokenization import collate
from .parity import upstream,trl_batch
from .strategy import restore_adapter


def run(output):
    from trl.trainer.utils import selective_log_softmax
    output.mkdir(parents=True,exist_ok=True)
    source=output.parent/'parity'
    cfg=Config(**read_json(ROOT/'artifacts/prepared/compact/config.json'))
    rows=read_json(ROOT/'artifacts/prepared/compact/data.json')['train'][:4]
    tok=get_tokenizer(cfg);model=make_model(cfg);trainer,_=upstream(tok.pad_token_id)
    records=[]
    for name,fixture in [('identity','initial-adapter.pt'),('nontrivial','nontrivial-adapter.pt')]:
        restore_adapter(model,torch.load(source/fixture,map_location='cpu',weights_only=True))
        for pair in rows:
            batch=collate([pair],tok.pad_token_id)
            captured={}
            def hook(module,args,kwargs,result):
                captured['input_ids']=(kwargs['input_ids'] if 'input_ids' in kwargs else args[0]).detach().clone()
                captured['attention_mask']=kwargs['attention_mask'].detach().clone()
                captured['logits']=result.logits.detach()
            with torch.no_grad():
                logits=model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask']).logits
                handle=model.register_forward_hook(hook,with_kwargs=True)
                try:result=trainer.concatenated_forward(model,trl_batch(pair,tok.pad_token_id))
                finally:handle.remove()
                assert torch.equal(captured['input_ids'],batch['input_ids'])
                assert torch.equal(captured['attention_mask'],batch['attention_mask'])
                logit_error=float((logits-captured['logits']).abs().max())
                assert logit_error==0
                mask=batch['completion_mask'][:,1:].bool()
                selected=logits[:,:-1][mask]
                labels=batch['input_ids'][:,1:][mask]
                ours_tokens=selected.log_softmax(-1).gather(-1,labels[:,None]).squeeze(-1)
                upstream_tokens=selective_log_softmax(selected,labels)
                precise=selected.double().log_softmax(-1).gather(-1,labels[:,None]).squeeze(-1)
                precise_trl=selective_log_softmax(selected.double(),labels)
                ours=completion_scores(logits,batch['input_ids'],batch['attention_mask'],batch['completion_mask'])
                trl=torch.cat([result['chosen_logps'],result['rejected_logps']])
                double=completion_scores(logits.double(),batch['input_ids'],batch['attention_mask'],batch['completion_mask'])
                counts=mask.sum(1).tolist()
                # Bound separates token arithmetic from reductions; no fitted constants.
                # Sum of observed per-token differences plus each FP32 reduction's
                # error versus a double sum of its own FP32 token values.
                bounds=[];offset=0;details=[]
                for i,count in enumerate(counts):
                    a=ours_tokens[offset:offset+count];b=upstream_tokens[offset:offset+count]
                    bound=float((a.double()-b.double()).abs().sum()+abs(ours[i].double()-a.double().sum())+abs(trl[i].double()-b.double().sum()))
                    bounds.append(bound)
                    details.append({'ours_fp32_vs_double':float(abs(ours[i].double()-double[i])),
                                    'trl_fp32_vs_double':float(abs(trl[i].double()-double[i]))})
                    offset+=count
                records.append({'state':name,'id':pair['id'],'identical_inputs_masks_logits':True,
                    'logit_max_abs':logit_error,'completion_tokens':counts,'ours_scores':ours.tolist(),'trl_scores':trl.tolist(),
                    'double_scores':double.tolist(),'per_token_double_implementation_max_abs':float((precise-precise_trl).abs().max()),
                    'fp32_token_difference_max_abs':float((ours_tokens-upstream_tokens).abs().max()),
                    'fp32_ours_token_vs_double_max_abs':float((ours_tokens.double()-precise).abs().max()),
                    'fp32_trl_token_vs_double_max_abs':float((upstream_tokens.double()-precise).abs().max()),
                    'reduction_diagnostic':details,'observed_numerical_decomposition_bounds':bounds,
                    'observed_score_errors':(ours-trl).abs().tolist()})
                del selected,logits,captured,precise,precise_trl,ours_tokens,upstream_tokens
    result={'status':'completed','records':records,'scope':'Same actual installed TRL forward; hooks only observe inputs/logits; FP64 reductions diagnostic only',
            'selective_log_softmax_source_sha256':file_hash(__import__('pathlib').Path(inspect.getfile(selective_log_softmax))),
            'selective_log_softmax_source':inspect.getsource(selective_log_softmax),
            'initial_thresholds_not_modified':True}
    atomic_json(output/'result.json',result)
    return result
