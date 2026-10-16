"""Independent arithmetic and independent raw-data selection audit."""
import collections
import math
import random
import statistics
import unicodedata
import zipfile
import numpy as np
import torch
from ..provenance import ROOT, read_json, atomic_json, file_hash, digest
from ..config import Config
from ..models import get_tokenizer
from .strategy import tensor_digest


def recompute(examples, beta=.1, threshold=1e-6):
    # Python double / math.log1p, independent of torch implementation and stored loss.
    margins=[beta*((r['policy_chosen']-r['reference_chosen'])-(r['policy_rejected']-r['reference_rejected'])) for r in examples]
    losses=[max(-x,0)+math.log1p(math.exp(-abs(x))) for x in margins]
    wins=sum(x>threshold for x in margins);ties=sum(abs(x)<=threshold for x in margins)
    return {'pairs':len(examples),'mean_loss_double':sum(losses)/len(losses),'wins':wins,'ties':ties,
            'strict_positive_accuracy':wins/len(margins),'margin_mean_double':sum(margins)/len(margins),
            'max_saved_loss_roundoff':max(abs(a-r['loss']) for a,r in zip(losses,examples)),
            'max_saved_margin_roundoff':max(abs(a-r['reward_margin']) for a,r in zip(margins,examples)),
            'id_digest':digest([r['id'] for r in examples])}


def data_audit(output):
    import pyarrow.parquet as pq
    cfg=Config(**read_json(ROOT/'artifacts/prepared/compact/config.json'))
    tok=get_tokenizer(cfg)
    data=read_json(ROOT/'artifacts/prepared/compact/data.json')
    manifest=read_json(ROOT/'artifacts/prepared/compact/manifest.json')
    normal=lambda s:' '.join(unicodedata.normalize('NFKC',s).split()).casefold()
    stats=collections.Counter();accepted={};test_groups=set();sources={}
    for split in ['test_prefs','train_prefs']:
        files=list((ROOT/'.cache/data').rglob(split+'-*.parquet'))
        assert len(files)==1
        sources[split]={'file':files[0].relative_to(ROOT).as_posix(),'sha256':file_hash(files[0])}
        accepted[split]=[];seen=set()
        for raw in pq.read_table(files[0]).to_pylist():
            stats[split+':original']+=1
            reason=None
            c,r=raw.get('chosen'),raw.get('rejected')
            if not isinstance(c,list) or not isinstance(r,list) or len(c)!=2 or len(r)!=2:
                reason='unsupported conversation structure'
            elif c[0]!=r[0] or c[0].get('role')!='user' or c[-1].get('role')!='assistant' or r[-1].get('role')!='assistant':
                reason='inconsistent context or roles'
            else:
                prompt,chosen,rejected=c[0].get('content'),c[1].get('content'),r[1].get('content')
                if any(not isinstance(x,str) or not x.strip() for x in [prompt,chosen,rejected]): reason='empty text'
                elif chosen.strip()==rejected.strip(): reason='identical completions'
                elif raw.get('prompt') is not None and normal(raw['prompt'])!=normal(prompt): reason='prompt field mismatch'
                else:
                    for ck,rk in [('score_chosen','score_rejected'),('chosen_score','rejected_score')]:
                        if ck in raw and rk in raw:
                            if not all(isinstance(raw[k],(int,float)) and math.isfinite(raw[k]) for k in [ck,rk]): reason='invalid preference scores'
                            elif raw[ck]<=raw[rk]: reason='non-strict or reversed scores'
            if reason is None:
                identity=digest([prompt,chosen,rejected]);group=digest(normal(prompt))
                if split=='test_prefs':test_groups.add(group)
                if identity in seen: reason='duplicate example'
                else:
                    seen.add(identity)
                    prompt_ids=tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=True,add_generation_prompt=True)
                    sequences={}
                    for kind,text in [('chosen',chosen),('rejected',rejected)]:
                        suffix=tok.encode(text,add_special_tokens=False)
                        if not suffix:reason='empty response';break
                        if tok.eos_token_id in suffix:reason='embedded response-ending token';break
                        if len(prompt_ids)+len(suffix)+1>cfg.max_length:reason='length cap';break
                        sequences[kind+'_ids']=prompt_ids+suffix+[tok.eos_token_id]
                        sequences[kind+'_mask']=[0]*len(prompt_ids)+[1]*(len(suffix)+1)
                    if reason is None:
                        accepted[split].append({'id':identity,'prompt_hash':group,'prompt_ids':prompt_ids,**sequences})
            if reason:stats[split+':excluded:'+reason]+=1
        stats[split+':eligible']=len(accepted[split])
    pool=[]
    for row in accepted['train_prefs']:
        if row['prompt_hash'] in test_groups:stats['train_prefs:excluded:test prompt conflict']+=1
        else:pool.append(row)
    def unique(rows):
        result=[];groups=set()
        for row in sorted(rows,key=lambda x:x['id']):
            if row['prompt_hash'] in groups:stats['excluded:duplicate prompt group']+=1
            else:groups.add(row['prompt_hash']);result.append(row)
        return result
    pool=unique(pool);test=unique(accepted['test_prefs']);rng=random.Random(cfg.split_seed)
    rng.shuffle(pool);rng.shuffle(test)
    selected={'validation':pool[:64],'train':pool[64:320],'test':test[:64]}
    for split,rows in selected.items():
        assert [r['id'] for r in rows]==[r['id'] for r in data[split]]
        for reconstructed,saved in zip(rows,data[split]):
            assert all(saved[k]==v for k,v in reconstructed.items())
    assert dict(stats)==manifest['statistics'],(dict(stats),manifest['statistics'])
    groups={k:{r['prompt_hash'] for r in rows} for k,rows in selected.items()}
    assert not (groups['train']&groups['test'] or groups['validation']&groups['test'] or groups['train']&groups['validation'])
    return {'status':'passed','statistics':dict(stats),'sources':sources,'all_selected_ids_tokens_masks_match':True,
            'no_normalized_prompt_overlap':True,'split_ids':{k:[r['id'] for r in rows] for k,rows in selected.items()},
            'implementation':'Independent raw parquet filter, tokenization reconstruction, group selection and seed shuffle; does not call prepare_pretrained/extract_single_turn'}


def equal_nested(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal_nested(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal_nested(x,y) for x,y in zip(a,b))
    return a==b


def run(output):
    output.mkdir(parents=True,exist_ok=True)
    folder=output.parent
    inventory=read_json(folder/'inventory.json')
    study=read_json(folder/'original/latest_study.json')['path'].split('/')[-1]
    plan=read_json(folder/'original/experiment_plan.json')
    archived=zipfile.ZipFile(folder/'original/source-and-paper.zip')
    import hashlib
    archived_files={p:hashlib.sha256(archived.read(p)).hexdigest() for p in archived.namelist()
                    if p.startswith(('src/','scripts/','tests/','configs/'))}
    assert digest(archived_files)==plan['code_hash']
    runs=[];incomplete=[];evidence=[]
    for path in sorted((ROOT/'artifacts/runs').glob(study+'*')):
        metrics=path/'metrics.json'
        if not metrics.exists():
            incomplete.append({'run_id':path.name,'status':'incomplete','progress':read_json(path/'progress.json') if (path/'progress.json').exists() else None})
            continue
        m=read_json(metrics)
        if m['kind'] not in ['benchmark','learning']: continue
        assert m['status']=='completed'
        spec=read_json(path/'resolved_config.json')
        assert spec['plan_hash']==plan['sha256']
        assert m['updates']==(25 if m['kind']=='benchmark' else 16)
        assert m['pairs']==(320 if m['kind']=='benchmark' else 256)
        expected_lr=[5e-5 if i<1 else 5e-5*(m['updates']-i)/(m['updates']-1) for i in range(m['updates'])]
        assert all(abs(t['learning_rate']-lr)<1e-18 for t,lr in zip(m['trace'],expected_lr))
        records={k:v for k,v in m.items() if k not in ['initial_generations','final_generations']}
        records.update(run_id=path.name,artifact=metrics.relative_to(ROOT).as_posix(),artifact_sha256=file_hash(metrics))
        runs.append(records)
        if m['kind']=='learning':
            for phase in ['initial_evaluation','final_evaluation']:
                for split,evaluation in m[phase].items():
                    result=recompute(evaluation['examples'])
                    assert result['max_saved_loss_roundoff']<1e-7
                    assert result['max_saved_margin_roundoff']<1e-7
                    assert abs(result['mean_loss_double']-evaluation['dpo_loss'])<1e-7
                    assert result['strict_positive_accuracy']==evaluation['strict_positive_accuracy']
                    evidence.append({'run_id':path.name,'phase':phase,'split':split,**result})
    timing={}
    for arm in ['uncached_reference','cached_reference','warm_cached_reference']:
        group=[r for r in runs if r['kind']=='benchmark' and (('warm_cached_reference' if r['cache_is_warm'] else r['arm'])==arm)]
        assert len(group)==3
        timing[arm]={key:{'median':statistics.median(values),'min':min(values),'max':max(values)} for key,values in {
            'training_seconds':[r['training_seconds'] for r in group], 'whole_process_seconds':[r['whole_process_seconds'] for r in group],
            'training_rss_mib':[r['training_memory']['sampled_peak_rss_bytes']/2**20 for r in group],
            'precompute_seconds':[r['reference_precompute_seconds'] for r in group]}.items()}
    learning=[r for r in runs if r['kind']=='learning']
    assert len(learning)==2
    a,b=learning
    assert a['trace']==b['trace'] and a['final_evaluation']==b['final_evaluation']
    paths=[ROOT/r['checkpoint'] for r in learning]
    hashes=[file_hash(p) for p in paths]
    checkpoints=[torch.load(p,map_location='cpu',weights_only=False) for p in paths] # only our trusted local files
    assert equal_nested(*checkpoints)
    scopes={k:equal_nested(checkpoints[0][k],checkpoints[1][k]) for k in checkpoints[0]}
    canonical_hashes=[tensor_digest(c['model']) for c in checkpoints]
    assert hashes[0]==hashes[1] and canonical_hashes[0]==canonical_hashes[1]
    checkpoint={'file_sha256':hashes,'file_bytes_identical':True,'all_saved_fields_equal':scopes,
                'canonical_model_sha256':canonical_hashes,'named_tensor_count':len(checkpoints[0]['model']),
                'hash_definition':'Sorted names including tied aliases, shapes, dtypes, length-delimited contiguous tensor bytes',
                'old_hash_scope':'Sorted names and raw tensor bytes; no explicit shape/dtype; state_dict only',
                'paths':[p.relative_to(ROOT).as_posix() for p in paths]}
    del checkpoints
    u=timing['uncached_reference']['training_seconds']['median']*.8
    c=timing['cached_reference']['training_seconds']['median']*.8
    pre=timing['cached_reference']['precompute_seconds']['median']
    audit={'status':'passed','original_source_sha256':plan['code_hash'],'original_plan_sha256':plan['sha256'],
        'original_source_archive_verified':True,'runs':runs,'incomplete_attempts':incomplete,'independent_metrics':evidence,
        'timing':timing,'checkpoint_equality':checkpoint,
        'projection':{'U':u,'C':c,'P':pre,'equal_setup_one_pass_cached_minus_uncached_seconds':pre+c-u,
                       'equal_setup_break_even_passes':pre/(u-c),'scope':'Stationary 256-pair training-pass projection from 320-pair benchmark; not observed learning'},
        'historical_attribution':'Unresolved: old eval/generation lumped and split cache precompute absent; new replay cannot retroactively identify old stage causes'}
    atomic_json(output/'numeric-audit.json',audit)
    audit['data']=data_audit(output)
    atomic_json(output/'result.json',audit)
    return audit
