"""Actual installed TRL 0.24 sigmoid/forward checks on preserved training pairs."""
import copy
import inspect
from types import SimpleNamespace
import torch
from ..config import Config
from ..models import make_model, get_tokenizer
from ..provenance import ROOT, atomic_json, read_json, file_hash, seed_all
from ..scoring import model_scores, completion_scores
from ..loss import dpo_loss
from ..tokenization import collate
from ..reference import build_cache, cache_context
from .strategy import (SharedReference, StrategyTrainer, disabled_reference, adapter_state,
                       base_state, tensor_digest, restore_adapter)

THRESHOLDS = {'fp64_loss': 1e-12, 'fp64_gradient': 1e-11, 'fp32_scores': 2e-5,
              'fp32_loss': 2e-6, 'fp32_gradients': 2e-6, 'trajectory_parameters': 2e-6}


def upstream(pad):
    import trl
    from trl import DPOTrainer
    from trl.trainer.dpo_config import FDivergenceType
    if trl.__version__ != '0.24.0':
        raise RuntimeError('Only inspected TRL 0.24.0 integration is supported')
    trainer = object.__new__(DPOTrainer)
    values = dict(accelerator=SimpleNamespace(device=torch.device('cpu')), reference_free=False,
        f_divergence_type=FDivergenceType.REVERSE_KL, f_divergence_params={}, beta=.1, label_smoothing=0.,
        pad_token_id=pad, label_pad_token_id=-100, is_encoder_decoder=False, aux_loss_enabled=False,
        max_length=None, truncation_mode='keep_start', use_logits_to_keep=False, padding_free=False,
        use_weighting=False, loss_type=['sigmoid'], args=SimpleNamespace(rpo_alpha=None, ld_alpha=None))
    for k, v in values.items():
        setattr(trainer, k, v)
    return trainer, file_hash(__import__('pathlib').Path(inspect.getfile(DPOTrainer)))


def trl_batch(pair, pad):
    # One preference pair per microbatch: same right padding as production.
    prompt = pair['prompt_ids']
    assert pair['chosen_ids'][:len(prompt)] == pair['rejected_ids'][:len(prompt)] == prompt
    result = {'prompt_input_ids': torch.tensor([prompt]),
              'prompt_attention_mask': torch.ones(1, len(prompt), dtype=torch.long)}
    for kind in ['chosen', 'rejected']:
        ids = pair[kind+'_ids'][len(prompt):]
        assert pair[kind+'_mask'] == [0]*len(prompt)+[1]*len(ids)
        result[kind+'_input_ids'] = torch.tensor([ids])
        result[kind+'_attention_mask'] = torch.ones(1, len(ids), dtype=torch.long)
    return result


def error(a, b):
    a, b = a.detach().double(), b.detach().double()
    d = (a-b).abs()
    return {'max_absolute': float(d.max()), 'max_relative_floor_1e-8': float((d/b.abs().clamp_min(1e-8)).max()),
            'rms_absolute': float(d.square().mean().sqrt()), 'left_norm': float(a.norm()), 'right_norm': float(b.norm())}


def compare(model, reference, rows, tokenizer, trainer, logit_dtype=None):
    other = copy.deepcopy(model)
    handles=[]
    if logit_dtype is not None:
        def cast(module,args,output):
            output.logits=output.logits.to(logit_dtype)
            return output
        handles=[m.register_forward_hook(cast) for m in [model,other]]
    model.zero_grad(set_to_none=True)
    other.zero_grad(set_to_none=True)
    results = []
    for pair in rows:
        batch = collate([pair], tokenizer.pad_token_id)
        with torch.no_grad():
            ref = model_scores(reference, batch)
        score = model_scores(model, batch)
        result = trainer.concatenated_forward(other, trl_batch(pair, tokenizer.pad_token_id))
        score2 = torch.cat([result['chosen_logps'], result['rejected_logps']])
        loss = dpo_loss(score[:1], score[1:], ref[:1], ref[1:])
        loss2 = trainer.dpo_loss(score2[:1], score2[1:], ref[:1], ref[1:], loss_type='sigmoid')[0].mean()
        (loss/len(rows)).backward()
        (loss2/len(rows)).backward()
        results.append({'id':pair['id'], 'ours_scores': score.detach().tolist(), 'trl_scores':score2.detach().tolist(),
                        'reference_scores':ref.tolist(), 'score_error': error(score,score2),
                        'ours_loss':float(loss.detach()), 'trl_loss':float(loss2.detach()), 'loss_error':error(loss,loss2)})
    grads = []
    for (n, p), (n2, q) in zip(model.named_parameters(), other.named_parameters()):
        assert n == n2
        if not p.requires_grad:
            assert p.grad is None and q.grad is None
            continue
        row = {'name':n,'shape':list(p.shape),'dtype':str(p.dtype), 'ours_present':p.grad is not None, 'trl_present':q.grad is not None}
        assert row['ours_present'] == row['trl_present']
        if p.grad is not None:
            row.update(error(p.grad,q.grad))
            row['max_absolute_values']=float(torch.maximum(p.grad.abs(),q.grad.abs()).max())
            row['max_error_over_atol2e-6_rtol1e-4']=float(((p.grad-q.grad).abs()/(2e-6+1e-4*q.grad.abs())).max())
        grads.append(row)
    report = {'examples':results, 'gradients':grads, 'gradient_tensor_count':len(grads),
              'gradient_element_count':sum(p.numel() for p in model.parameters() if p.requires_grad),
              'max_score_error':max(r['score_error']['max_absolute'] for r in results),
              'max_loss_error':max(r['loss_error']['max_absolute'] for r in results),
              'mean_ours_loss':sum(r['ours_loss'] for r in results)/len(rows),
              'mean_trl_loss':sum(r['trl_loss'] for r in results)/len(rows),
              'max_gradient_error':max(r.get('max_absolute',0.) for r in grads),
              'gradient_presence_agrees':True}
    report['initial_thresholds_pass'] = (report['max_score_error'] <= THRESHOLDS['fp32_scores'] and
        report['max_loss_error'] <= THRESHOLDS['fp32_loss'] and report['max_gradient_error'] <= THRESHOLDS['fp32_gradients'])
    for handle in handles:handle.remove()
    del other
    return report


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    cfg = Config(**read_json(ROOT/'artifacts/prepared/compact/config.json'))
    data = read_json(ROOT/'artifacts/prepared/compact/data.json')
    rows = data['train'][:4]  # fixed original order; never selected for agreement
    atomic_json(output/'declared-checks.json', {'thresholds':THRESHOLDS,'ids':[p['id'] for p in rows],
        'selection':'First four preserved training rows, before scoring; unequal lengths retained',
        'scope':'Installed concatenated_forward and sigmoid loss, not full TRL Trainer execution'})
    tok = get_tokenizer(cfg)
    model, ref = make_model(cfg), make_model(cfg, reference=True)
    trl, source = upstream(tok.pad_token_id)
    before = adapter_state(model)
    base_before = tensor_digest(base_state(model))
    fixture = {k: [{n:p[n] for n in ['id','prompt_ids','chosen_ids','rejected_ids','chosen_mask','rejected_mask']} for p in rows]
               for k in ['pairs']}
    atomic_json(output/'inputs.json',fixture)
    torch.save(before, output/'initial-adapter.pt')
    report = {'trl_version':'0.24.0','peft_version':'0.17.1','trl_source_sha256':source,
        'peft_source_sha256':file_hash(__import__('pathlib').Path(inspect.getfile(type(model)))),
        'peft_api':'PeftModel.disable_adapter; documentation 0.17.0, installed implementation 0.17.1',
        'scope':'Pretrained SmolLM2 LoRA; identical weights, one-pair right-padded microbatches, FP32 eager CPU',
        'thresholds':THRESHOLDS,'identity':compare(model,ref,rows,tok,trl)}
    atomic_json(output/'parity-progress.json',report)
    small = copy.deepcopy(cfg)
    small.effective_batch_pairs = 2
    learner = StrategyTrainer(model,ref,None,small,rows,tok.pad_token_id,total_updates=2)
    learner.run()
    changed = adapter_state(model)
    torch.save(changed,output/'nontrivial-adapter.pt')
    report['adapter_changed'] = tensor_digest(before) != tensor_digest(changed)
    assert report['adapter_changed']
    report['training_updates'] = learner.trace
    report['nontrivial'] = compare(model,ref,rows,tok,trl)
    report['base_before_sha256'] = base_before
    report['base_after_sha256'] = tensor_digest(base_state(model))
    assert base_before == report['base_after_sha256']
    report['adapter_change_max'] = max(float((changed[n]-before[n]).abs().max()) for n in before)
    with torch.no_grad():
        original = model_scores(model,collate([rows[0]],tok.pad_token_id))
        altered = copy.deepcopy(rows[0])
        altered['chosen_ids'][len(altered['prompt_ids'])] = (altered['chosen_ids'][len(altered['prompt_ids'])]+1)%model.config.vocab_size
        wrong_result = trl.concatenated_forward(model,trl_batch(altered,tok.pad_token_id))
        wrong = torch.cat([wrong_result['chosen_logps'],wrong_result['rejected_logps']])
    report['negative_control'] = {'deliberately_changed_first_completion_token':True,'score_error':error(original,wrong),
                                  'scope':'Local original input against actual installed TRL forward with changed completion input',
                                  'detected':not torch.allclose(original,wrong,atol=THRESHOLDS['fp32_scores'],rtol=0)}
    assert report['negative_control']['detected']
    report['shared'] = shared_check(model,ref,rows,tok,cfg,before,output)
    report['status'] = 'passed' if report['identity']['initial_thresholds_pass'] and report['nontrivial']['initial_thresholds_pass'] else 'threshold_failure_requires_diagnosis'
    atomic_json(output/'result.json',report)
    return report


def shared_check(model, ref, rows, tok, cfg, initial, output):
    base_hash = tensor_digest(base_state(model))
    separate_base=base_state(ref)
    names=[]
    for name,value in base_state(model).items():
        assert name in separate_base and torch.equal(value,separate_base[name])
        names.append({'name':name,'shape':list(value.shape),'dtype':str(value.dtype),'numel':value.numel()})
    reference_errors = []
    for state in [initial, adapter_state(model)]:
        restore_adapter(model,state)
        for pair in rows:
            batch=collate([pair],tok.pad_token_id)
            a=model_scores(SharedReference(model),batch)
            with torch.no_grad(): b=model_scores(ref,batch)
            assert not a.requires_grad
            reference_errors.append(float((a-b).abs().max()))
    parameters=list(model.parameters())
    flags=[p.requires_grad for p in parameters]
    modes=[m.training for m in model.modules()]
    grads=[p.grad for p in parameters]
    enablement=[m.disable_adapters for m in model.modules() if hasattr(m,'disable_adapters') and isinstance(getattr(m,'disable_adapters'),bool)]
    try:
        with disabled_reference(model):
            raise RuntimeError('deliberate exception')
    except RuntimeError as e:
        assert str(e)=='deliberate exception'
    assert flags==[p.requires_grad for p in parameters]
    assert modes==[m.training for m in model.modules()]
    assert all(a is b for a,b in zip(grads,[p.grad for p in parameters]))
    assert enablement==[m.disable_adapters for m in model.modules() if hasattr(m,'disable_adapters') and isinstance(getattr(m,'disable_adapters'),bool)]
    cfg=copy.deepcopy(cfg);cfg.effective_batch_pairs=2
    trajectories={};states={}; times={};gradient_states={}
    import time
    for arm in ['separate_reference','shared_base_reference','cached_shared_base']:
        start=time.perf_counter()
        restore_adapter(model,initial)
        model.zero_grad(set_to_none=True)
        seed_all(cfg.seed,cfg.threads)
        reference=ref if arm=='separate_reference' else SharedReference(model)
        cache=None
        if arm=='cached_shared_base':
            cache=build_cache(output/'smoke-cache.json',reference,rows,cache_context(cfg,base_hash,tok),tok.pad_token_id)
            reference=None
        learner=StrategyTrainer(model,reference,cache,cfg,rows+rows[:2],tok.pad_token_id,total_updates=3)
        opt_ids={id(p) for g in learner.optimizer.param_groups for p in g['params']}
        assert opt_ids=={id(p) for n,p in model.named_parameters() if 'lora_' in n}
        learner.run()
        assert tensor_digest(base_state(model))==base_hash
        assert all(torch.equal(value,separate_base[name]) for name,value in base_state(model).items())
        gradient_states[arm]={name:p.grad.detach().clone() for name,p in model.named_parameters() if p.requires_grad and p.grad is not None}
        trajectories[arm]={'trace':learner.trace,'reference_forward_calls':learner.reference_forward_calls,
                           'adapter_sha256':tensor_digest(adapter_state(model)), 'base_sha256':base_hash,
                           'last_clipped_gradient_sha256':tensor_digest(gradient_states[arm])}
        states[arm]=adapter_state(model)
        times[arm]=time.perf_counter()-start
    maximum=max(float((states['separate_reference'][n]-s[n]).abs().max()) for s in states.values() for n in s)
    trace_diff=max(abs(a['loss']-b['loss']) for t in trajectories.values()
                   for a,b in zip(trajectories['separate_reference']['trace'],t['trace']))
    assert maximum<=THRESHOLDS['trajectory_parameters'] and max(reference_errors)<=THRESHOLDS['fp32_scores']
    assert trajectories['cached_shared_base']['reference_forward_calls']==0
    gradient_difference=max(float((gradient_states['separate_reference'][n]-g[n]).abs().max()) for g in gradient_states.values() for n in g)
    expected_gradients={name for name,p in model.named_parameters() if p.requires_grad}
    assert all(set(g)==expected_gradients for g in gradient_states.values())
    assert gradient_difference<=THRESHOLDS['fp32_gradients']
    assert trace_diff<=THRESHOLDS['fp32_loss']
    return {'status':'passed','before_after_reference_max_abs':max(reference_errors),'exception_restore_passed':True,
        'base_parameters_and_all_buffers_unchanged':True,'only_adapter_optimized':True,'reference_has_no_grad':True,
        'trajectories':trajectories,'max_adapter_difference':maximum,'max_trace_loss_difference':trace_diff,
        'pilot_seconds':times,'pilot_pairs_per_arm':6,'pilot_updates_per_arm':3,
        'actual_named_base_tensors_compared_to_separate_reference':names,
        'last_clipped_gradient_max_abs_difference':gradient_difference,
        'hash_scope':'All named base parameters including tied aliases and all buffers including nonpersistent; names, shapes, dtypes, contiguous bytes'}
