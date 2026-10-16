"""Offline local regression cases; pretrained checks are a separate executable."""
import copy
import torch
import pytest
from dpo_mini.config import profile
from dpo_mini.models import make_model
from dpo_mini.data import synthetic
from dpo_mini.train import Trainer
from dpo_mini.finalization.strategy import tensor_digest, disabled_reference, SharedReference


def test_checkpoint_atomic_failure_preserves_previous(tmp_path,monkeypatch):
    cfg=profile('toy',train_pairs=4,effective_batch_pairs=2,threads=2)
    data,_=synthetic(cfg)
    trainer=Trainer(make_model(cfg),make_model(cfg,True),None,cfg,data['train'],0)
    trainer.run(max_updates=1)
    path=tmp_path/'checkpoint.pt';trainer.save(path,{'fixture':1})
    previous=path.read_bytes()
    def fail(*args,**kwargs):
        args[1].write(b'incomplete')
        raise OSError('simulated interrupted writer')
    monkeypatch.setattr(torch,'save',fail)
    with pytest.raises(OSError,match='interrupted'): trainer.save(path,{'fixture':1})
    assert path.read_bytes()==previous
    assert list(tmp_path.iterdir())==[path]


def test_canonical_hash_covers_shape_dtype_and_name():
    a=torch.tensor([1,2],dtype=torch.int32)
    assert tensor_digest({'a':a})!=tensor_digest({'b':a})
    assert tensor_digest({'a':a})!=tensor_digest({'a':a.reshape(1,2)})
    assert tensor_digest({'a':a})!=tensor_digest({'a':a.view(torch.float32)})
    assert tensor_digest({'b':a,'a':a})==tensor_digest({'a':a,'b':a})


def test_shared_context_restores_flags_modes_existing_gradients_exception():
    from transformers import LlamaConfig,LlamaForCausalLM
    from peft import LoraConfig,get_peft_model
    from peft.tuners.tuners_utils import BaseTunerLayer
    cfg=LlamaConfig(vocab_size=24,hidden_size=16,intermediate_size=32,num_hidden_layers=1,
                    num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=16)
    model=get_peft_model(LlamaForCausalLM(cfg),LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'))
    model.train()
    layers=[m for m in model.modules() if isinstance(m,BaseTunerLayer)]
    layers[0].eval()
    params=list(model.parameters())
    for p in params:
        if p.requires_grad: p.grad=torch.ones_like(p)
    flags=[p.requires_grad for p in params];grads=[p.grad for p in params]
    modes=[m.training for m in model.modules()]
    for raises in [False,True]:
        try:
            with disabled_reference(model):
                out=model(input_ids=torch.tensor([[1,2,3]])).logits
                assert not out.requires_grad
                assert all(layer.disable_adapters for layer in layers)
                if raises: raise RuntimeError('controlled')
        except RuntimeError as e: assert str(e)=='controlled'
        assert flags==[p.requires_grad for p in params]
        assert modes==[m.training for m in model.modules()]
        assert all(a is b for a,b in zip(grads,[p.grad for p in params]))
        assert all(not layer.disable_adapters for layer in layers)
    assert model(input_ids=torch.tensor([[1,2,3]])).logits.requires_grad


def test_resume_restores_optimizer_and_all_cpu_rngs(tmp_path):
    import random
    import numpy as np
    cfg=profile('toy',train_pairs=4,effective_batch_pairs=2,threads=2)
    data,_=synthetic(cfg)
    trainer=Trainer(make_model(cfg),make_model(cfg,True),None,cfg,data['train'],0)
    trainer.run(max_updates=1)
    torch.rand(7);np.random.random(4);random.random()
    path=tmp_path/'resume.pt';trainer.save(path,{'test':'rng'})
    expected=(torch.rand(3),np.random.random(3),random.random())
    target=Trainer(make_model(cfg),make_model(cfg,True),None,cfg,data['train'],0)
    target.restore(path,{'test':'rng'})
    assert torch.equal(torch.rand(3),expected[0])
    assert np.array_equal(np.random.random(3),expected[1])
    assert random.random()==expected[2]
    a,b=trainer.optimizer.state_dict(),target.optimizer.state_dict()
    assert a['param_groups']==b['param_groups']
    assert a['state'].keys()==b['state'].keys()
    for key,state in a['state'].items():
        for name,value in state.items():
            assert torch.equal(value,b['state'][key][name]) if isinstance(value,torch.Tensor) else value==b['state'][key][name]


def test_completed_stage_hash_and_frozen_source_gate(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from dpo_mini.finalization import pipeline
    from dpo_mini.provenance import atomic_json,file_hash,digest
    monkeypatch.setattr(pipeline,'ROOT',tmp_path)
    path=tmp_path/'result.json';atomic_json(path,{'status':'completed'})
    atomic_json(tmp_path/'ledger.json',[{'status':'completed','result':{'path':'result.json','sha256':file_hash(path)}}])
    assert pipeline.checked_record(tmp_path,path)
    atomic_json(path,{'status':'changed'})
    with pytest.raises(ValueError,match='changed'):pipeline.checked_record(tmp_path,path)
    plan={'execution_source':{'sha256':'old'}};plan['sha256']=digest(plan)
    atomic_json(tmp_path/'plan.json',plan)
    monkeypatch.setattr(pipeline,'execution_manifest',lambda:{'sha256':'new'})
    with pytest.raises(AssertionError,match='source changed'):
        pipeline.freeze(tmp_path,None,SimpleNamespace())
