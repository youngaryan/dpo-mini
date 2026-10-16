"""Regenerate evidence-linked numbers and all manuscript formats, without training."""
import csv
import math
import os
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
from ..provenance import ROOT,atomic_json,read_json,digest,file_hash,now

ARM_LABELS={'separate_reference':'A: separate reference','shared_base_reference':'B: shared base','cached_shared_base':'C: cold shared cache'}


def sanitized(value):
    if isinstance(value,dict):
        return {k:sanitized(v) for k,v in value.items() if k not in ['torch_build','sys_path','initial_generations','final_generations']}
    if isinstance(value,list):return [sanitized(v) for v in value]
    if isinstance(value,str):
        for prefix in [str(ROOT),str(ROOT).replace('\\','/')]:value=value.replace(prefix,'.')
        return value.replace('\\','/')
    return value


def tests(path):
    if not path.exists():return {'status':'not_run'}
    root=ET.parse(path).getroot()
    suites=[root] if root.tag=='testsuite' else list(root.findall('testsuite'))
    out={k:sum(int(s.get(k,'0')) for s in suites) for k in ['tests','failures','errors','skipped']}
    out['passed']=out['tests']-out['failures']-out['errors']-out['skipped']
    out['status']='passed' if not out['failures'] and not out['errors'] else 'failed'
    return out


def prepare_evidence(folder):
    evidence=folder/'evidence';evidence.mkdir(parents=True,exist_ok=True)
    # In a release/clean copy raw heavy artifacts are deliberately absent. Keep
    # existing evidence immutable and use it directly for report-only operations.
    if not (folder/'audit/result.json').exists():
        assert (evidence/'manifest.json').exists(),'Numeric evidence missing'
        return evidence
    historical=read_json(folder/'audit/result.json')
    atomic_json(evidence/'historical.json',sanitized(historical))
    index=read_json(folder/'run-index.json') if (folder/'run-index.json').exists() else []
    followup=[]
    for entry in index:
        path=ROOT/entry['path']
        assert file_hash(path/'metrics.json')==entry['metrics_sha256']
        followup.append({'index':entry,'metrics':read_json(path/'metrics.json'),
                         'parent':read_json(path/'parent-timing.json')})
    atomic_json(evidence/'followup.json',sanitized(followup))
    checks={'core':tests(folder/'core-tests.xml'),'clean_core':tests(folder/'clean-core-tests.xml')}
    for key,name in [('parity','parity/result.json'),('diagnosis','diagnose/result.json'),('precision','precision/result.json'),
                     ('adjudication','parity-adjudication.json'),('clean','clean-reproduction.json'),
                     ('clean_isolation','clean-isolation.json'),('clean_pretrained','clean-pretrained-result.json'),
                     ('tiny_upstream','original/upstream.json'),('clean_tiny_upstream','clean-tiny-upstream.json')]:
        if (folder/name).exists():checks[key]=read_json(folder/name)
    if (folder/'installed-source-observations.json').exists():
        import json
        checks['installed_source_observations']=json.loads((folder/'installed-source-observations.json').read_text(encoding='utf-8-sig'))
    atomic_json(evidence/'checks.json',sanitized(checks))
    inv=read_json(folder/'inventory.json');ledger=read_json(folder/'ledger.json')
    measured=sum(r.get('charged_seconds',0) for r in ledger)
    resources={'new_measured_process_seconds':measured,'historical_measured_process_seconds':inv['previous_measured_experiment_seconds'],
        'historical_accounted_seconds':inv['previous_budget_accounted_experiment_seconds'],
        'cumulative_measured_seconds':measured+inv['previous_measured_experiment_seconds'],
        'cumulative_accounted_seconds':measured+inv['previous_budget_accounted_experiment_seconds'],
        'new_budget_seconds':inv['new_experiment_budget_seconds'],'new_download_bytes':0,
        'new_download_scope':'Offline finalization; existing model/data and uv package cache reused. Public documentation viewed separately; no new payload downloads.',
        'pass_started_at':inv['created_at'],'updated_at':now(),
        'elapsed_wall_seconds':(datetime.fromisoformat(now())-datetime.fromisoformat(inv['created_at'])).total_seconds(),
        'storage_at_start_bytes':inv['storage_at_start_bytes'],'storage_budget_additional_bytes':inv['additional_storage_limit_bytes'],
        'accounting':'Leaf subprocess elapsed including failed attempts, tests, checks and installation. Never add parent orchestration duration. Wall time includes editing and waits.',
        'ledger':[{k:r[k] for k in ['id','stage','status','started_at','ended_at','charged_seconds','exit_code','source','log'] if k in r} for r in ledger]}
    if (folder/'storage-audit.json').exists():resources['storage']=read_json(folder/'storage-audit.json')
    atomic_json(evidence/'resources.json',sanitized(resources))
    sources=read_json(folder/'original/source_registry.json')
    for source in sources:
        if source.get('id') in ['R1','R13','R14']:
            import re
            import html
            cached=ROOT/source['artifact']
            assert file_hash(cached)==source['sha256']
            content=cached.read_text(encoding='utf8')
            source['authors']=[html.unescape(x) for x in re.findall(r'name="citation_author" content="([^"]+)"',content)]
            source['year']=re.search(r'name="citation_date" content="(\d{4})',content).group(1)
            source['metadata_verification']='Citation metadata parsed from hash-verified historical primary-source HTML; no new retrieval date invented'
        if source.get('id')=='R2':
            source['historical_retrieval_url']=source.get('url')
            source['url']='https://github.com/eric-mitchell/direct-preference-optimization/tree/f8b8c0f49dc92a430bae41585f9d467d3618fe2f'
            source['citation_note']='Pinned commit actually inspected historically; moving API was retrieval metadata only'
    sources.append({'id':'R15','title':'PEFT model adapter API','author':'Hugging Face',
        'url':'https://huggingface.co/docs/peft/v0.17.0/en/package_reference/peft_model',
        'retrieved_at':'2026-10-15','status':'retrieved via browser in this finalization pass',
        'version_note':'Documentation 0.17.0; installed PEFT 0.17.1 disable_adapter implementation inspected locally'})
    atomic_json(evidence/'sources.json',sanitized(sources))
    atomic_json(evidence/'environment.json',sanitized(read_json(folder/'environment.json')))
    if (folder/'plan.json').exists():atomic_json(evidence/'plan.json',sanitized(read_json(folder/'plan.json')))
    manifest={p.name:file_hash(p) for p in sorted(evidence.glob('*.json')) if p.name!='manifest.json'}
    atomic_json(evidence/'manifest.json',{'files':manifest,'scope':'Portable numeric derivatives of preserved raw artifacts; raw origin hashes retained; excludes weights and dataset text'})
    return evidence


def stats(values):
    return {'n':len(values),'median':statistics.median(values),'min':min(values),'max':max(values)} if values else None


def calculate(folder):
    e=prepare_evidence(folder)
    h=read_json(e/'historical.json');follow=read_json(e/'followup.json');checks=read_json(e/'checks.json')
    plan=read_json(e/'plan.json') if (e/'plan.json').exists() else None
    timing=[r for r in follow if r['metrics']['kind']=='benchmark']
    replay=[r for r in follow if r['metrics']['kind']=='replay']
    summary={'historical':{'timing':h['timing'],'projection':h['projection'],'learning':[],'paired_reductions':[]},
             'followup':{'timing':{},'paired_reductions':[],'replay':replay},'checks':checks,
             'resources':read_json(e/'resources.json'),'plan':plan,'cohort_complete':len(timing)==9 and len(replay)==2}
    for run in h['runs']:
        if run['kind']=='learning':
            summary['historical']['learning'].append({k:run[k] for k in ['run_id','arm','training_seconds','whole_process_seconds','final_evaluation','initial_evaluation','updates','pairs','trace']})
    workflow_components={}
    for run in h['runs']:
        if run['kind']!='learning':continue
        components={k:run[k] for k in ['setup_including_cache_seconds','training_seconds','evaluation_and_generation_seconds','checkpoint_seconds']}
        components['unallocated_parent_residual_seconds']=run['whole_process_seconds']-sum(components.values())
        assert components['unallocated_parent_residual_seconds']>=0
        components['whole_process_seconds']=run['whole_process_seconds']
        workflow_components[run['arm']]=components
    summary['historical']['workflow_components']=workflow_components
    summary['historical']['workflow_cached_minus_uncached']={key:workflow_components['cached_reference'][key]-workflow_components['uncached_reference'][key]
        for key in workflow_components['cached_reference']}
    for arm in ARM_LABELS:
        rows=[r for r in timing if r['metrics']['arm']==arm]
        summary['followup']['timing'][arm]={
            'training_seconds':stats([r['metrics']['training_seconds'] for r in rows]),
            'whole_process_seconds':stats([r['parent']['whole_process_seconds'] for r in rows]),
            'training_rss_mib':stats([r['metrics']['training_memory']['sampled_peak_rss_bytes']/2**20 for r in rows]),
            'whole_os_peak_mib':stats([r['metrics']['whole_workflow_memory']['os_process_peak_rss_bytes']/2**20 for r in rows]),
            'precompute_seconds':stats([r['metrics']['stages_seconds'].get('reference_precompute_train',0) for r in rows]),
            'trials':[{'trial':r['metrics']['trial'],'path':r['index']['path'],'training_seconds':r['metrics']['training_seconds'],
                      'whole_process_seconds':r['parent']['whole_process_seconds']} for r in rows]}
    for trial in range(3):
        selected={r['metrics']['arm']:r for r in timing if r['metrics']['trial']==trial}
        if len(selected)==3:
            for a,b in [('separate_reference','shared_base_reference'),('shared_base_reference','cached_shared_base'),('separate_reference','cached_shared_base')]:
                summary['followup']['paired_reductions'].append({'trial':trial,'baseline':a,'comparison':b,
                    'training_percent':100*(selected[a]['metrics']['training_seconds']-selected[b]['metrics']['training_seconds'])/selected[a]['metrics']['training_seconds'],
                    'whole_percent':100*(selected[a]['parent']['whole_process_seconds']-selected[b]['parent']['whole_process_seconds'])/selected[a]['parent']['whole_process_seconds']})
        old={('warm_cached_reference' if r['cache_is_warm'] else r['arm']):r for r in h['runs'] if r['kind']=='benchmark' and f'-timing-{trial}-' in r['run_id']}
        if len(old)==3:
            a,b=old['uncached_reference'],old['cached_reference']
            summary['historical']['paired_reductions'].append({'trial':trial,
                'training_percent':100*(a['training_seconds']-b['training_seconds'])/a['training_seconds'],
                'whole_percent':100*(a['whole_process_seconds']-b['whole_process_seconds'])/a['whole_process_seconds']})
    if summary['cohort_complete']:
        a=summary['followup']['timing']['shared_base_reference'];b=summary['followup']['timing']['cached_shared_base']
        u=a['training_seconds']['median']*256/(plan['measured_updates']*16)
        c=b['training_seconds']['median']*256/(plan['measured_updates']*16)
        pre=b['precompute_seconds']['median']
        overhead={}
        for arm in ['shared_base_reference','cached_shared_base']:
            vals=[]
            for r in timing:
                if r['metrics']['arm']!=arm:continue
                s=r['metrics']['stages_seconds']
                vals.append(r['parent']['whole_process_seconds']-s['measured_training']-s.get('warmup_training',0)-s.get('reference_precompute_train',0))
            overhead[arm]=statistics.median(vals)
        su,sc=overhead['shared_base_reference'],overhead['cached_shared_base']
        summary['followup']['projection']={'U':u,'C':c,'P':pre,'S_u':su,'S_c_excluding_P':sc,
            'break_even_passes':(sc-su+pre)/(u-c) if u>c else None,
            'one_pass_cached_minus_shared_seconds':sc+pre+c-su-u,
            'formula':'S_c + P + E*C < S_u + E*U; E > (S_c-S_u+P)/(U-C) when U>C',
            'scope':'Training-only 256-pair pass projection; median measured per-pair work. Setup excludes P and warmup; includes measured nontraining checks and unallocated launch/exit residual. No held-out evaluation, generation or learning checkpoint.'}
        summary['followup']['all_benchmark_final_model_hashes_equal']=len({r['metrics']['final_model_sha256'] for r in timing})==1
        summary['followup']['all_benchmark_traces_equal']=all(r['metrics']['trace']==timing[0]['metrics']['trace'] for r in timing)
    atomic_json(folder/'summary.json',summary)
    return summary,h,follow


def verify(folder):
    evidence=folder/'evidence'
    if not (evidence/'manifest.json').exists():prepare_evidence(folder)
    manifest=read_json(evidence/'manifest.json')
    for name,expected in manifest['files'].items():
        assert Path(name).name==name and file_hash(evidence/name)==expected,(name,'evidence hash mismatch')
    h=read_json(evidence/'historical.json');follow=read_json(evidence/'followup.json')
    plan=read_json(evidence/'plan.json')
    plan_body={key:value for key,value in plan.items() if key!='sha256'}
    assert digest(plan_body)==plan['sha256'],'Frozen plan digest mismatch'
    execution_files_checked=0
    for name,sha in plan['execution_source']['files'].items():
        path=ROOT/name
        if path.exists():
            assert file_hash(path)==sha,('Frozen execution source differs',name)
            execution_files_checked+=1
    benchmark=[row for row in follow if row['metrics']['kind']=='benchmark']
    for row in benchmark:
        r=row['metrics']
        assert r['plan_sha256']==plan['sha256']
        assert r['source_sha256']==plan['execution_source']['sha256']
        assert r['config']==plan['config']
        assert r['warmup_updates']==plan['warmup_updates']
        assert r['measured_updates']==plan['measured_updates']
        assert r['pairs']==plan['measured_updates']*plan['config']['effective_batch_pairs']
        assert r['initial_model_sha256']==plan['initial_full_model_sha256']
        assert r['initial_fixture_sha256']==plan['initial_fixture_sha256']
    if benchmark:
        for field in ['data_order_sha256','initial_rng_sha256','base_sha256','response_tokens']:
            assert len({row['metrics'][field] for row in benchmark})==1,('Unmatched benchmark field',field)
    if (folder/'summary.json').exists():
        summary=read_json(folder/'summary.json')
        for arm,values in summary['followup']['timing'].items():
            selected=[r for r in benchmark if r['metrics']['arm']==arm]
            expected={
                'training_seconds':stats([r['metrics']['training_seconds'] for r in selected]),
                'whole_process_seconds':stats([r['parent']['whole_process_seconds'] for r in selected]),
                'training_rss_mib':stats([r['metrics']['training_memory']['sampled_peak_rss_bytes']/2**20 for r in selected]),
                'whole_os_peak_mib':stats([r['metrics']['whole_workflow_memory']['os_process_peak_rss_bytes']/2**20 for r in selected])}
            for key,value in expected.items():assert values[key]==value,('Derived statistic differs',arm,key)
    paper_consistency='No structured manuscript available yet'
    if (ROOT/'paper/report_blocks.json').exists():
        structured=read_json(ROOT/'paper/report_blocks.json')
        if isinstance(structured,list):
            paper_consistency='Legacy manuscript format; finalization manuscript consistency not checked before regeneration'
            blocks=[]
        else:
            assert isinstance(structured,dict) and 'blocks' in structured,'Unknown structured manuscript format'
            blocks=structured['blocks']
            paper_consistency='Regenerated finalization manuscript tables match CSV files'
        for block in blocks:
            if block['kind']=='table':
                with (ROOT/'paper/tables'/(block['name']+'.csv')).open(encoding='utf8',newline='') as stream:
                    actual=list(csv.reader(stream))
                expected=[[str(v) for v in row] for row in [block['columns'],*block['rows']]]
                assert actual==expected,('Manuscript CSV differs',block['name'])
    archived_raw_files_checked=0
    if (folder/'inventory.json').exists():
        inventory=read_json(folder/'inventory.json')
        for name,record in inventory.get('historical_artifact_references',{}).items():
            path=ROOT/name
            if path.exists():
                assert file_hash(path)==record['sha256'],('Historical raw artifact changed',name)
                archived_raw_files_checked+=1
    from .audit import recompute
    counts={'historical_learning_examples':0,'followup_learning_examples':0}
    for r in h['runs']:
        if r['kind']=='learning':
            for phase in ['initial_evaluation','final_evaluation']:
                for split,ev in r[phase].items():
                    assert [row['id'] for row in ev['examples']]==h['data']['split_ids'][split]
                    calc=recompute(ev['examples'])
                    assert abs(calc['mean_loss_double']-ev['dpo_loss'])<1e-7
                    assert calc['strict_positive_accuracy']==ev['strict_positive_accuracy']
                    assert calc['ties']/calc['pairs']==ev['tie_rate']
                    counts['historical_learning_examples']+=len(ev['examples'])
    for arm,record in h['timing'].items():
        rows=[r for r in h['runs'] if r['kind']=='benchmark' and ('warm_cached_reference' if r['cache_is_warm'] else r['arm'])==arm]
        for key in ['training_seconds','whole_process_seconds']:
            assert statistics.median(r[key] for r in rows)==record[key]['median']
    for record in follow:
        r=record['metrics'];parent=record['parent']
        assert abs(sum(r['stages_seconds'].values())+parent['unallocated_residual_seconds']-parent['whole_process_seconds'])<1e-8
        assert parent['unallocated_residual_seconds']>=0
        if r['kind'] not in ['replay','benchmark','smoke']:continue
        assert r['updates']==len(r['trace'])
        assert r['pairs']==sum(x['pairs'] for x in r['trace'][r['warmup_updates']:])
        assert r['data_order_sha256']==digest(r['data_order_ids'])
        if 'cached' in r['arm']:assert r['reference_training_forward_calls']==0
        if r['kind']=='replay':
            original=next(x for x in h['runs'] if x['kind']=='learning' and x['arm']==('uncached_reference' if r['arm']=='separate_reference' else 'cached_reference'))
            assert r['trace']==original['trace'],'Replay trace differs from original protocol'
            assert r['initial_evaluation']==original['initial_evaluation'] and r['final_evaluation']==original['final_evaluation']
            assert r['final_model_sha256'] in h['checkpoint_equality']['canonical_model_sha256']
            for phase in ['initial_evaluation','final_evaluation']:
                for split,ev in r[phase].items():
                    assert [row['id'] for row in ev['examples']]==h['data']['split_ids'][split]
                    calc=recompute(ev['examples']);assert abs(calc['mean_loss_double']-ev['dpo_loss'])<1e-7
                    assert calc['strict_positive_accuracy']==ev['strict_positive_accuracy']
                    assert calc['ties']/calc['pairs']==ev['tie_rate']
                    counts['followup_learning_examples']+=len(ev['examples'])
    checkpoints=h['checkpoint_equality'];local=[]
    for name,expected in zip(checkpoints['paths'],checkpoints['file_sha256']):
        path=ROOT/name
        if path.exists():assert file_hash(path)==expected;local.append(name)
    local_adapters=[]
    import torch
    from .strategy import tensor_digest
    benchmark_states=[]
    for record in follow:
        r=record['metrics'];path=ROOT/record['index']['path']/'final-adapter.pt'
        if path.exists() and 'adapter_sha256' in r:
            state=torch.load(path,map_location='cpu',weights_only=True)
            assert tensor_digest(state)==r['adapter_sha256']
            local_adapters.append(path.relative_to(ROOT).as_posix())
            if r['kind']=='benchmark':benchmark_states.append(state)
        if r['kind']=='benchmark':
            expected=sum(row['pairs'] for row in r['trace'])/r['config']['microbatch_pairs']
            assert r['policy_training_forward_calls']==expected
            assert r['reference_training_forward_calls']==(0 if r['arm']=='cached_shared_base' else expected)
    if benchmark_states:
        baseline=benchmark_states[0]
        assert all(state.keys()==baseline.keys() and all(torch.equal(value,baseline[name]) for name,value in state.items()) for state in benchmark_states)
    replay_checkpoint_checks=[]
    if local:
        from .audit import equal_nested
        # Only these project-generated, locally retained checkpoints are loaded.
        original=torch.load(ROOT/local[0],map_location='cpu',weights_only=False)
        for record in follow:
            if record['metrics']['kind']!='replay':continue
            path=ROOT/record['index']['path']/'checkpoints/final.pt'
            if not path.exists():continue
            actual=torch.load(path,map_location='cpu',weights_only=False)
            assert actual['identity']['plan']==record['metrics']['plan_sha256']
            assert actual['model'].keys()==original['model'].keys()
            assert all(torch.equal(value,original['model'][name]) for name,value in actual['model'].items())
            comparable=[name for name in original if name!='identity']
            field_equality={name:equal_nested(actual[name],original[name]) for name in comparable}
            assert all(field_equality[name] for name in ['model','optimizer','scheduler','scaler','update_index','data_position','trace'])
            replay_checkpoint_checks.append({'path':path.relative_to(ROOT).as_posix(),'sha256':file_hash(path),
                'named_model_tensors_compared':len(actual['model']),'saved_fields_compared_to_historical_except_identity':field_equality,
                'identity_scope':'New plan/reference identity intentionally differs; no old-to-new file-byte equality claim'})
            del actual
        del original
    availability={'historical_checkpoint_files':len(local),
        'pretrained_weight_cache':any((ROOT/'.cache/models/compact').glob('*.safetensors')),
        'prepared_tokenized_data':(ROOT/'artifacts/prepared/compact/data.json').exists(),
        'raw_dataset':(ROOT/'.cache/data').exists(),'local_adapter_files':len(local_adapters)}
    if availability['prepared_tokenized_data'] and (evidence/'plan.json').exists():
        assert file_hash(ROOT/'artifacts/prepared/compact/data.json')==read_json(evidence/'plan.json')['data_sha256']
    scope=('numeric evidence plus historical raw/checkpoint file hashes and named local adapter tensors' if local
           else 'numeric evidence only; full historical checkpoint tensors unavailable for direct verification')
    if not availability['pretrained_weight_cache']:scope+='; pretrained weight cache unavailable'
    if not availability['raw_dataset']:scope+='; raw dataset unavailable'
    report={'status':'passed','scope':scope,'counts':counts,'evidence_files_verified':len(manifest['files']),
        'frozen_execution_source_files_verified':execution_files_checked,'plan_digest_verified':True,
        'manuscript_consistency_scope':paper_consistency,
        'historical_checkpoint_files_rehashed':len(local),'historical_inventory_files_rehashed':archived_raw_files_checked,
        'local_adapter_files_rehashed_and_named_tensors_checked':len(local_adapters),
        'availability':availability,
        'replay_checkpoint_checks':replay_checkpoint_checks,
        'human_review':'not recorded','at':now()}
    atomic_json(folder/'verification.json',report)
    atomic_json(ROOT/'artifacts/verification.json',{'finalization':folder.relative_to(ROOT).as_posix(),**report})
    print('Verification:',scope,flush=True)
    return report


def build(folder):
    summary,historical,follow=calculate(folder)
    from .manuscript import write
    write(folder,summary,historical,follow)
    atomic_json(ROOT/'artifacts/analysis/summary.json',{'finalization':folder.relative_to(ROOT).as_posix(),**summary})
    atomic_json(ROOT/'artifacts/claims.json',read_json(folder/'claims.json'))
    atomic_json(ROOT/'artifacts/source_registry.json',read_json(folder/'evidence/sources.json'))
    verify(folder)
