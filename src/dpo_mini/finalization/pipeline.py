"""Plan, sequential fresh-process experiments and idempotent evidence gates."""
import sys
import uuid
import zipfile
import xml.etree.ElementTree as ET
from ..provenance import ROOT,read_json,atomic_json,digest,file_hash,now,source_manifest
from .common import execution_manifest

ARMS=['separate_reference','shared_base_reference','cached_shared_base']


def checked_record(folder,path):
    if not path.exists():return False
    relative=path.relative_to(ROOT).as_posix()
    records=[r for r in read_json(folder/'ledger.json') if r.get('status')=='completed' and r.get('result',{}).get('path')==relative]
    if not records:return False
    if records[-1]['result']['sha256']!=file_hash(path):raise ValueError('Completed artifact changed: '+relative)
    return True


def benchmark(folder,label,spec,execute,budget,resume):
    index_path=folder/'run-index.json'
    if resume and index_path.exists():
        for entry in reversed(read_json(index_path)):
            if entry['label']!=label:continue
            previous=ROOT/entry['path']
            if checked_record(folder,previous/'metrics.json'):
                assert read_json(previous/'spec.json')==spec,'Completed run protocol changed'
                assert file_hash(previous/'spec.json')==entry['spec_sha256'],'Completed spec artifact changed'
                assert file_hash(previous/'parent-timing.json')==entry['parent_timing_sha256']
                return read_json(previous/'metrics.json')
    out=folder/'runs'/label
    result=out/'metrics.json'
    if result.exists():
        if not resume:raise ValueError('Existing run requires --resume; refusing overwrite')
        if checked_record(folder,result):
            saved=read_json(out/'spec.json')
            if saved!=spec:raise ValueError('Saved run differs from requested protocol')
            if (out/'parent-timing.json').exists():
                metrics=read_json(result);parent=read_json(out/'parent-timing.json')
                records=[r for r in read_json(folder/'ledger.json') if r.get('result',{}).get('path')==result.relative_to(ROOT).as_posix() and r['status']=='completed']
                assert parent['whole_process_seconds']==records[-1]['charged_seconds']
                assert abs(sum(metrics['stages_seconds'].values())+parent['unallocated_residual_seconds']-parent['whole_process_seconds'])<1e-8
                index=read_json(index_path) if index_path.exists() else []
                index.append({'label':label,'path':out.relative_to(ROOT).as_posix(),'metrics_sha256':file_hash(result),
                    'parent_timing_sha256':file_hash(out/'parent-timing.json'),'spec_sha256':file_hash(out/'spec.json'),
                    'index_recovered_from_completed_ledger_at':now()})
                atomic_json(index_path,index)
                return metrics
            raise ValueError('Completed child needs parent-timer reconciliation before resume')
    if out.exists():out=folder/'runs'/(label+'-retry-'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/'launch-spec.json',spec)
    record=execute(folder,label,[sys.executable,'scripts/finalization_worker.py','benchmark',str(out),
                               '--spec',str(out/'launch-spec.json')],budget,result if out.name==label else out/'metrics.json')
    metrics=read_json(out/'metrics.json')
    # Parent elapsed stays in separate raw record; child result is immutable.
    accounted=sum(metrics['stages_seconds'].values())
    residual=record['charged_seconds']-accounted
    assert residual>=-.01
    atomic_json(out/'parent-timing.json',{'whole_process_seconds':record['charged_seconds'],
        'allocated_stage_seconds':accounted,'unallocated_residual_seconds':residual,
        'definition':'Monotonic Popen-to-exit elapsed minus nonoverlapping child stages; residual includes imports, launch, final output and interpreter exit, not attributed retrospectively'})
    index_path=folder/'run-index.json';index=read_json(index_path) if index_path.exists() else []
    index.append({'label':label,'path':out.relative_to(ROOT).as_posix(),'metrics_sha256':file_hash(out/'metrics.json'),
                  'parent_timing_sha256':file_hash(out/'parent-timing.json'),'spec_sha256':file_hash(out/'spec.json')})
    atomic_json(index_path,index)
    return metrics


def freeze(folder,execute,args):
    path=folder/'plan.json'
    if path.exists():
        plan=read_json(path)
        assert plan['execution_source']['sha256']==execution_manifest()['sha256'],'Worker source changed after freeze'
        check=dict(plan);sha=check.pop('sha256');assert digest(check)==sha
        if 'numerical_adjudication_sha256' in plan:
            assert file_hash(folder/'parity-adjudication.json')==plan['numerical_adjudication_sha256'],'Numerical adjudication changed'
            decision=read_json(folder/'parity-adjudication.json')
            for key,name in [('original_result_sha256','parity/result.json'),('diagnostic_result_sha256','diagnose/result.json'),('precision_result_sha256','precision/result.json')]:
                assert file_hash(folder/name)==decision[key],('Numerical adjudication input changed',name)
            assert checked_record(folder,folder/'audit/result.json')
            assert checked_record(folder,folder/'parity/result.json')
            assert checked_record(folder,folder/'core-tests.xml')
            core=ET.parse(folder/'core-tests.xml').getroot().find('testsuite')
            assert int(core.get('failures'))==int(core.get('errors'))==0,'Core failure prevents reuse'
        if 'environment_sha256' in plan:
            assert file_hash(folder/'environment.json')==plan['environment_sha256'],'Environment record changed'
            import importlib.metadata
            for name,version in plan['versions'].items():
                assert importlib.metadata.version(name)==version,'Installed dependency differs from frozen plan'
            assert file_hash(ROOT/plan['initial_fixture'])==plan['initial_fixture_sha256'],'Initial fixture changed'
            assert file_hash(ROOT/'artifacts/prepared/compact/data.json')==plan['data_sha256'],'Prepared dataset changed'
        return plan
    audit=read_json(folder/'audit/result.json');assert audit['status']=='passed'
    assert checked_record(folder,folder/'audit/result.json')
    core=ET.parse(folder/'core-tests.xml').getroot().find('testsuite')
    assert int(core.get('failures'))==int(core.get('errors'))==0
    parity=read_json(folder/'parity/result.json');assert parity['shared']['status']=='passed'
    decision=read_json(folder/'parity-adjudication.json')
    assert decision['status']=='numerically_explained_with_revised_bounds'
    assert decision['original_result_sha256']==file_hash(folder/'parity/result.json')
    assert decision['diagnostic_result_sha256']==file_hash(folder/'diagnose/result.json')
    assert decision['precision_result_sha256']==file_hash(folder/'precision/result.json')
    env=read_json(folder/'environment.json')
    cfg=read_json(ROOT/'artifacts/prepared/compact/config.json')
    base={'config':cfg,'data_sha256':file_hash(ROOT/'artifacts/prepared/compact/data.json'),
          'versions':env['versions'],'execution_sha256':execution_manifest()['sha256']}
    fixture=benchmark(folder,'initial-fixture',{**base,'kind':'fixture'},execute,args.max_experiment_hours,True)
    fixture_entry=next(r for r in reversed(read_json(folder/'run-index.json')) if r['label']=='initial-fixture')
    fixture_path=ROOT/fixture_entry['path']/'initial.pt'
    # Before selecting final size, exercise this exact instrumented worker on all
    # arms. Separate from the reported timings and the training-only parity pilot.
    pilot_results=[]
    for arm in ARMS:
        cfg_small={**cfg,'effective_batch_pairs':2}
        pilot_results.append(benchmark(folder,'worker-smoke-'+arm,{**base,'config':cfg_small,'kind':'smoke','arm':arm,
            'plan_sha256':'pre-freeze-worker-smoke','initial_fixture':fixture_path.relative_to(ROOT).as_posix(),
            'train_limit':4,'warmup_updates':0,'measured_updates':3},execute,args.max_experiment_hours,True))
    assert len({r['final_model_sha256'] for r in pilot_results})==1
    assert len({r['data_order_sha256'] for r in pilot_results})==1
    historical=read_json(folder/'audit/result.json')
    estimate=9*max(r['whole_process_seconds'] for r in historical['runs'] if r['kind']=='benchmark')+2*max(
        r['whole_process_seconds'] for r in historical['runs'] if r['kind']=='learning')+900
    spent=sum(r.get('charged_seconds',0) for r in read_json(folder/'ledger.json'))
    remaining=args.max_experiment_hours*3600-spent
    if estimate+2700>remaining:
        measured=10
        reason='Full 5+20 projection plus 45-minute reserve exceeded remaining process allowance; all arms reduced equally to 5+10'
    else:measured=20;reason=None
    plan={'created_at':now(),'cohort':'Post-historical local follow-up; not public preregistration',
        'execution_source':execution_manifest(),'full_source_at_freeze':source_manifest(),
        'environment_sha256':file_hash(folder/'environment.json'),'config':cfg,
        'data_sha256':base['data_sha256'],'versions':base['versions'],
        'initial_fixture':fixture_path.relative_to(ROOT).as_posix(),'initial_fixture_sha256':file_hash(fixture_path),
        'initial_full_model_sha256':fixture['full_model_sha256'], 'seed':42,'trials':3,'warmup_updates':5,'measured_updates':measured,
        'order':[ARMS,ARMS[1:]+ARMS[:1],ARMS[2:]+ARMS[:2]],'replay_order':['separate_reference','historical_cached_reference'],
        'replay_updates':16,'replay_pairs':256,'numerical_adjudication_sha256':file_hash(folder/'parity-adjudication.json'),
        'original_tolerances':parity['thresholds'],'scope_selection':{'full_work_projection_seconds':estimate,'spent_seconds':spent,
            'remaining_seconds':remaining,'reserve_seconds':2700,'reduction_reason':reason},
        'stopping_rules':'Sequential leaf processes, total additional <=6h, each <=90min, >=1GiB available RAM; failures preserved; no favorable trial selection',
        'reference_before_policy':'New ABC and smoke only; historical replay retains policy-before-reference ordering',
        'cache_construction':'New C shared base, train only; replay C separate reference, train+validation+test',
        'parent_timer':'Popen to exit; child stages mutually exclusive, residual explicitly unallocated',
        'source_scope':'Worker closure hash gates numerical execution; full source snapshot and report revision separately tracked'}
    plan['sha256']=digest(plan);atomic_json(path,plan)
    with zipfile.ZipFile(folder/'frozen-source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in plan['full_source_at_freeze']['files']:z.write(ROOT/name,name)
    return plan


def run(args,folder,execute):
    if args.stage=='clean':
        from .clean import run as clean
        return clean(folder,execute,args)
    if args.stage=='package':
        from .package import build
        return build(folder,execute,args)
    if args.stage is None:
        # Explicitly consume manual, hash-verified stages; never infer success from a directory name.
        for name in ['audit','parity']:
            if not checked_record(folder,folder/name/'result.json'):
                execute(folder,name,[sys.executable,'scripts/finalization_worker.py',name,str(folder/name)],args.max_experiment_hours,folder/name/'result.json')
    plan=freeze(folder,execute,args)
    common={'config':plan['config'],'data_sha256':plan['data_sha256'],'versions':plan['versions'],
            'execution_sha256':plan['execution_source']['sha256'],'plan_sha256':plan['sha256'],
            'initial_fixture':plan['initial_fixture']}
    for arm in plan['replay_order']:
        benchmark(folder,'replay-'+arm,{**common,'kind':'replay','arm':arm},execute,args.max_experiment_hours,args.resume)
    for trial,order in enumerate(plan['order']):
        for arm in order:
            benchmark(folder,f'timing-{trial}-{arm}',{**common,'kind':'benchmark','arm':arm,'trial':trial,
                'warmup_updates':plan['warmup_updates'],'measured_updates':plan['measured_updates']},execute,args.max_experiment_hours,args.resume)
    atomic_json(folder/'experiments-complete.json',{'status':'completed','plan_sha256':plan['sha256'],'at':now()})
    if args.stage=='experiments':return
    from .clean import run as clean
    clean(folder,execute,args)
    from .report import build
    build(folder)
    from .package import build as package
    package(folder,execute,args)
