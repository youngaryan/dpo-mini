"""Read-only run observation plus explicit computational/review stage status."""
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET
from ..provenance import ROOT,read_json,atomic_json,file_hash,now


def refresh(folder):
    def result(name):return read_json(folder/name) if (folder/name).exists() else {}
    ledger=result('ledger.json') or []
    index=result('run-index.json') or []
    complete=[]
    for entry in index:
        path=ROOT/entry['path']/'metrics.json'
        if path.exists() and file_hash(path)==entry['metrics_sha256']:complete.append(read_json(path))
    core={'status':'pending'}
    if (folder/'core-tests.xml').exists():
        suite=ET.parse(folder/'core-tests.xml').getroot().find('testsuite')
        values={k:int(suite.get(k,0)) for k in ['tests','failures','errors','skipped']}
        core={**values,'passed':values['tests']-values['failures']-values['errors']-values['skipped'],
              'status':'passed' if not values['failures'] and not values['errors'] else 'failed'}
    parity=result('parity/result.json');adjudication=result('parity-adjudication.json')
    render=read_json(ROOT/'paper/render-record.json') if (ROOT/'paper/render-record.json').exists() else {}
    visual=result('pdf-visual-inspection.json')
    visual_current=bool(visual.get('status')=='passed' and (ROOT/'paper/paper.pdf').exists() and visual.get('pdf_sha256')==file_hash(ROOT/'paper/paper.pdf'))
    stages={'historical_audit':{'status':result('audit/result.json').get('status','pending')},'core_correctness':core,
        'pretrained_lora_upstream':{'status':'completed_with_qualification' if adjudication else 'pending',
            'original_tolerances':parity.get('status','not_run'),'adjudication':adjudication.get('status','not_run')},
        'shared_base_correctness':{'status':parity.get('shared',{}).get('status','pending')},
        'instrumented_accounting':{'status':'completed' if sum(r['kind']=='replay' for r in complete)==2 else 'pending',
                                    'completed_arms':sum(r['kind']=='replay' for r in complete)},
        'three_arm_benchmark':{'status':'completed' if sum(r['kind']=='benchmark' for r in complete)==9 else 'pending',
                               'completed_runs':sum(r['kind']=='benchmark' for r in complete)},
        'clean_reproduction':{'status':result('clean-reproduction.json').get('status','pending')},
        'manuscript_pdf':{'status':'passed' if visual_current else 'pending_visual_inspection','pages':render.get('pages')},
        'review_bundle':{'status':result('bundle-verification.json').get('status','pending')},
        'owner_approval':{'status':'not_recorded'}}
    active=[]
    for row in ledger:
        if row['status']!='running':continue
        progress=folder/'runs'/row['stage']/'progress.json'
        active.append({'stage':row['stage'],'started_at':row['started_at'],'pid':row.get('pid'),
                       'progress':read_json(progress) if progress.exists() else {'status':'setup/evaluation or uninstrumented progress'}})
    inventory=result('inventory.json')
    measured=sum(r.get('charged_seconds',0) for r in ledger)
    record={'updated_at':now(),'stages':stages,'active':active,'new_measured_leaf_process_seconds':measured,
        'historical_measured_seconds':inventory.get('previous_measured_experiment_seconds'),
        'historical_accounted_seconds':inventory.get('previous_budget_accounted_experiment_seconds'),
        'new_budget_seconds':21600,'owner_release_approval':'not recorded'}
    atomic_json(folder/'status.json',record)
    return record
