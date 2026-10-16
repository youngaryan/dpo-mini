"""Mutually exclusive monotonic stages for original replay and new A/B/C trials."""
import copy
import gc
import random
import time
import os
import numpy as np
import psutil
import torch
from ..config import Config
from ..models import make_model,get_tokenizer
from ..provenance import ROOT,atomic_json,read_json,digest,file_hash,now,state_hash
from ..reference import ReferenceCache,cache_context,token_identity
from ..tokenization import collate
from ..scoring import model_scores
from ..train import Trainer
from ..evaluate import evaluate,generate
from ..benchmark import MemorySampler
from .strategy import SharedReference,StrategyTrainer,tensor_digest,base_state,adapter_state,restore_adapter
from .common import Stages,execution_manifest


def run(spec,output):
    start=time.perf_counter();stages=Stages();output.mkdir(parents=True,exist_ok=True)
    atomic_json(output/'spec.json',spec)
    source=execution_manifest()
    if spec.get('execution_sha256') and spec['execution_sha256']!=source['sha256']:
        raise ValueError('Execution source differs from frozen follow-up plan')
    atomic_json(output/'provenance.json',{'started_at':now(),'pid':os.getpid(),
        'process_create_time':psutil.Process().create_time(),'source':source,'status':'running'})
    whole_memory=MemorySampler()
    cfg=Config(**spec['config']);cfg.validate()
    with stages.span('load_data_tokenizer_policy_validate'):
        data=read_json(ROOT/'artifacts/prepared/compact/data.json')
        manifest=read_json(ROOT/'artifacts/prepared/compact/manifest.json')
        assert all(digest(rows)==manifest['splits'][split]['split_hash'] for split,rows in data.items())
        assert file_hash(ROOT/'artifacts/prepared/compact/data.json')==spec['data_sha256']
        train=list(data['train']);random.Random(cfg.seed).shuffle(train)
        if spec.get('train_limit'):train=train[:spec['train_limit']]
        tok=get_tokenizer(cfg);policy=make_model(cfg)
        base_hash=tensor_digest(base_state(policy))
        params={'total':sum(p.numel() for p in policy.parameters()),'trainable':sum(p.numel() for p in policy.parameters() if p.requires_grad)}
    kind=spec['kind'];arm=spec.get('arm')
    if kind=='fixture':
        with stages.span('save_initial_fixture'):
            learner=StrategyTrainer(policy,SharedReference(policy),None,cfg,train,tok.pad_token_id,total_updates=25)
            initial={'adapter':adapter_state(policy),'optimizer':learner.optimizer.state_dict(),
                'torch_rng':torch.get_rng_state(),'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),
                'base_sha256':base_hash,'full_model_sha256':tensor_digest(policy.state_dict())}
            torch.save(initial,output/'initial.pt')
        result={'status':'completed','kind':kind,'initial_sha256':file_hash(output/'initial.pt'),
            'base_sha256':base_hash,'full_model_sha256':initial['full_model_sha256'],'stages_seconds':stages.values}
        whole_memory.finish();atomic_json(output/'metrics.json',result);return result
    initial=torch.load(ROOT/spec['initial_fixture'],map_location='cpu',weights_only=False)
    with stages.span('initial_fixture_identity'):
        assert base_hash==initial['base_sha256']
        restore_adapter(policy,initial['adapter'])
        initial_hash=tensor_digest(policy.state_dict())
        assert initial_hash==initial['full_model_sha256']
    reference=cache=None
    all_pairs=train+data['validation']+data['test'] if kind=='replay' else train
    with stages.span('reference_construction_and_identity'):
        if arm in ['separate_reference','historical_cached_reference']:
            reference=make_model(cfg,reference=True)
            assert tensor_digest(reference.state_dict())==initial_hash
        else: reference=SharedReference(policy)
        context=cache_context(cfg,base_hash,tok)
        context.update(strategy='shared_base' if arm=='cached_shared_base' else 'separate_reference',
                       threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),
                       software=spec['versions'],reference_model_revision=cfg.revision)
    if arm in ['cached_shared_base','historical_cached_reference']:
        rows=[]
        splits={'train':train,'validation':data['validation'],'test':data['test']} if kind=='replay' else {'train':train}
        for split,pairs in splits.items():
            with stages.span('reference_precompute_'+split):
                with torch.no_grad():
                    for pair in pairs:
                        values=model_scores(reference,collate([pair],tok.pad_token_id)).tolist()
                        rows.append({'id':pair['id'],'token_sha256':token_identity(pair),'chosen':values[0],'rejected':values[1]})
        with stages.span('cache_persist_verify_read'):
            payload={'version':1,'context':context,'rows':rows,'sha256':digest({'context':context,'rows':rows})}
            atomic_json(output/'reference-cache.json',payload)
            cache=ReferenceCache.load(output/'reference-cache.json',all_pairs,context)
            reference=None;gc.collect()
    initial_eval=final_eval=initial_gen=final_gen=None
    if kind=='replay':
        with stages.span('initial_evaluation'):
            initial_eval={s:evaluate(policy,reference,cache,data[s],cfg,tok.pad_token_id) for s in ['validation','test']}
        with stages.span('initial_generation'):
            initial_gen=generate(policy,tok,data['test'],cfg)
    warmup=0 if kind=='replay' else spec['warmup_updates']
    total=16 if kind=='replay' else warmup+spec['measured_updates']
    work=(train*((total*cfg.effective_batch_pairs+len(train)-1)//len(train)))[:total*cfg.effective_batch_pairs]
    with stages.span('optimizer_rng_restore'):
        klass=Trainer if kind=='replay' else StrategyTrainer
        learner=klass(policy,reference,cache,cfg,work,tok.pad_token_id,total_updates=total)
        learner.optimizer.load_state_dict(initial['optimizer'])
        torch.set_rng_state(initial['torch_rng']);random.setstate(initial['python_rng']);np.random.set_state(initial['numpy_rng'])
        rng_hash=digest({'torch':torch.get_rng_state().tolist(),'python':repr(random.getstate()),'numpy':repr(np.random.get_state())})
        def guard():
            if psutil.virtual_memory().available<2**30:raise MemoryError('Available RAM below 1 GiB')
        learner.guard=guard
        learner.progress_callback=lambda t:atomic_json(output/'progress.json',{
            'status':'running','completed_updates':t.update_index,'completed_pairs':t.position,'trace':t.trace})
    if warmup:
        with stages.span('warmup_training'):
            learner.run(max_updates=warmup,deadline=time.monotonic()+5400)
    sampler=MemorySampler()
    with stages.span('measured_training'):
        learner.run(deadline=time.monotonic()+5400)
    train_memory=sampler.finish()
    if kind=='replay':
        with stages.span('checkpoint_write'):
            learner.save(output/'checkpoints/final.pt',{'config':cfg.to_dict(),'data':digest(work),'plan':spec['plan_sha256'],'base':base_hash})
        with stages.span('final_evaluation'):
            final_eval={s:evaluate(policy,reference,cache,data[s],cfg,tok.pad_token_id) for s in ['validation','test']}
        with stages.span('final_generation'):
            final_gen=generate(policy,tok,data['test'],cfg)
    with stages.span('final_identity_validation'):
        assert tensor_digest(base_state(policy))==base_hash
        if arm=='separate_reference':assert tensor_digest(reference.state_dict())==initial_hash
        final_hash=tensor_digest(policy.state_dict())
        adapters=adapter_state(policy)
        torch.save(adapters,output/'final-adapter.pt')
        assert tensor_digest(adapters)!=tensor_digest(initial['adapter'])
        if cache is not None:assert learner.reference_forward_calls==0
    trace=learner.trace[warmup:];pairs=sum(x['pairs'] for x in trace)
    result={'status':'completed','kind':kind,'arm':arm,'trial':spec.get('trial'),'source_sha256':source['sha256'],
        'plan_sha256':spec['plan_sha256'],'config':cfg.to_dict(),'parameters':params,
        'warmup_updates':warmup,'measured_updates':len(trace),'updates':learner.update_index,'pairs':pairs,
        'response_tokens':sum(x['response_tokens'] for x in trace),'training_seconds':stages.values['measured_training'],
        'pairs_per_second':pairs/stages.values['measured_training'],'training_memory':train_memory,
        'reference_training_forward_calls':learner.reference_forward_calls,'policy_training_forward_calls':learner.policy_forward_calls,
        'initial_model_sha256':initial_hash,'final_model_sha256':final_hash,'base_sha256':base_hash,
        'adapter_sha256':tensor_digest(adapters),'initial_rng_sha256':rng_hash,'initial_fixture_sha256':file_hash(ROOT/spec['initial_fixture']),
        'data_order_sha256':digest([p['id'] for p in work]),'data_order_ids':[p['id'] for p in work],
        'trace':learner.trace,'initial_evaluation':initial_eval,'final_evaluation':final_eval,
        'initial_generations':initial_gen,'final_generations':final_gen,
        'torch_threads':torch.get_num_threads(),'torch_interop_threads':torch.get_num_interop_threads(),
        'gpu_memory':None,'gpu_note':'CPU only; unavailable, not zero','hardware_exclusive':False,
        'timing_scope':'Training includes collation, reference/lookup, policy forward/backward, clipping, AdamW, progress JSON; reference first only in new ABC',
        'cache_scope':'Train+validation+test' if kind=='replay' and cache else 'Train only' if cache else None}
    with stages.span('shutdown_release'):
        del learner,policy,reference,cache,adapters,initial;gc.collect()
    result['whole_workflow_memory']=whole_memory.finish()
    result['stages_seconds']=stages.values
    result['internal_observed_seconds']=time.perf_counter()-start
    result['internal_unallocated_seconds']=result['internal_observed_seconds']-sum(stages.values.values())
    assert result['internal_unallocated_seconds']>=-1e-6
    atomic_json(output/'metrics.json',result)
    atomic_json(output/'provenance.json',{'status':'completed','source':source,'plan_sha256':spec['plan_sha256'],'ended_at':now()})
    atomic_json(output/'progress.json',{'status':'completed','completed_updates':total,'completed_pairs':len(work)})
    return result
