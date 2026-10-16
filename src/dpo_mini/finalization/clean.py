"""Fresh isolated environment and scoped source/evidence reproduction."""
import os
import shutil
import sys
import uuid
from pathlib import Path
from ..provenance import ROOT,read_json,atomic_json,file_hash,digest,source_manifest,now
from .common import offline_environment


def copy_tree(source,target):
    shutil.copytree(source,target,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))


def run(folder,execute,args):
    offline_environment()
    clean=ROOT/'clean-room'
    record_path=folder/'clean-reproduction.json'
    if record_path.exists() and args.resume:
        result=read_json(record_path)
        if (result.get('status')=='passed' and result['source_sha256']==source_manifest()['sha256']
                and result.get('lock_sha256')==file_hash(ROOT/'requirements.lock')):
            assert result.get('proof_artifacts'),'Clean reproduction proof manifest missing'
            for name,sha in result['proof_artifacts'].items():
                assert file_hash(folder/name)==sha,('Clean reproduction artifact changed',name)
            return result
    if clean.exists() and not args.resume:
        raise ValueError('Clean room already exists; use --resume to verify/reuse completed installation')
    clean.mkdir(exist_ok=True)
    for name in ['src','scripts','tests','configs','docs','paper']:
        copy_tree(ROOT/name,clean/name)
    for name in ['pyproject.toml','requirements.lock','README.md','THIRD_PARTY_NOTICES.md']:
        shutil.copy2(ROOT/name,clean/name)
    copied_source={p.relative_to(clean).as_posix():file_hash(p)
        for name in ['src','scripts','tests','configs'] for p in (clean/name).rglob('*')
        if p.is_file() and '__pycache__' not in p.parts}
    assert digest(copied_source)==source_manifest()['sha256'],'Clean source copy differs from finalized source'
    cache_files={}
    # Explicit verified public cached inputs only; no working environment copies.
    for name in ['.cache/models/compact','artifacts/prepared/compact']:
        copy_tree(ROOT/name,clean/name)
        for p in (ROOT/name).rglob('*'):
            if p.is_file():
                relative=p.relative_to(ROOT).as_posix();h=file_hash(p)
                assert file_hash(clean/relative)==h
                cache_files[relative]=h
    relative=folder.relative_to(ROOT)
    destination=clean/relative;destination.mkdir(parents=True,exist_ok=True)
    (clean/'artifacts/finalization').mkdir(parents=True,exist_ok=True)
    shutil.copy2(ROOT/'artifacts/finalization/latest.json',clean/'artifacts/finalization/latest.json')
    # Report uses a portable numeric evidence package, prepared before this stage.
    from .report import prepare_evidence
    prepare_evidence(folder)
    copy_tree(folder/'evidence',destination/'evidence')
    for name in ['plan.json','parity-adjudication.json','inventory.json','environment.json','run-index.json','core-tests.xml']:
        if (folder/name).exists():shutil.copy2(folder/name,destination/name)
    for name in ['parity','diagnose','precision']:
        copy_tree(folder/name,destination/name)
    python=clean/('.venv/Scripts/python.exe' if os.name=='nt' else '.venv/bin/python')
    if not python.exists():
        execute(folder,'clean-create-venv',[sys.executable,'-m','venv','--without-pip',str(clean/'.venv')],args.max_experiment_hours)
    uv=shutil.which('uv')
    if not uv:raise RuntimeError('Offline cached-wheel installer uv unavailable')
    install_marker=folder/'clean-install.json'
    if not install_marker.exists():
        install=execute(folder,'clean-install-offline',[uv,'--offline','--cache-dir',str(ROOT/'.cache/uv'),'pip','install',
            '--python',str(python),'-r',str(clean/'requirements.lock')],args.max_experiment_hours,cwd=clean)
        atomic_json(install_marker,{'status':'completed','ledger_id':install['id'],'lock_sha256':file_hash(clean/'requirements.lock'),
            'network':'disabled by --offline','cache':'Existing project uv package cache; not a redownload test'})
    execute(folder,'clean-dependency-check',[uv,'--offline','pip','check','--python',str(python)],args.max_experiment_hours,cwd=clean)
    probe=clean/'isolation_probe.py'
    probe.write_text('''import sys,site,json,importlib.metadata,pathlib,hashlib,base64
root=pathlib.Path(__file__).resolve().parent
assert pathlib.Path(sys.prefix).resolve()==(root/'.venv').resolve()
assert not site.ENABLE_USER_SITE
assert 'include-system-site-packages = false' in (root/'.venv/pyvenv.cfg').read_text().lower()
assert all(not ('site-packages' in p and not pathlib.Path(p).resolve().is_relative_to(root/'.venv')) for p in sys.path)
versions={}
for line in (root/'requirements.lock').read_text().splitlines():
 if '==' in line and not line.startswith('#'):
  name,version=line.split('==');versions[name]=importlib.metadata.version(name)
  assert versions[name]==version,(name,versions[name],version)
import torch,transformers,peft,trl,numpy,pytest,reportlab,fitz
origins={m.__name__:str(pathlib.Path(m.__file__).resolve().relative_to(root)) for m in [torch,transformers,peft,trl,numpy,pytest,reportlab,fitz]}
checked=0
for dist in importlib.metadata.distributions():
 for f in dist.files or []:
  if f.hash and f.hash.mode=='sha256':
   p=pathlib.Path(dist.locate_file(f))
   if p.is_file():
    actual=base64.urlsafe_b64encode(hashlib.sha256(p.read_bytes()).digest()).rstrip(b'=').decode()
    assert actual==f.hash.value,(dist.metadata['Name'],str(f))
    checked+=1
result={'status':'passed','isolated':True,'user_site_enabled':site.ENABLE_USER_SITE,'versions':versions,'module_origins':origins,'installed_RECORD_hashes_checked':checked,'sys_path':[str(pathlib.Path(p).resolve()) for p in sys.path if p]}
(root/'isolation.json').write_text(json.dumps(result,sort_keys=True))
print('Isolation and installed RECORD hashes verified:',checked)
''',encoding='utf8')
    execute(folder,'clean-isolation',[str(python),'-I',str(probe)],args.max_experiment_hours,cwd=clean)
    shutil.copy2(clean/'isolation.json',folder/'clean-isolation.json')
    (clean/'test-tmp').mkdir(exist_ok=True)
    execute(folder,'clean-core',[str(python),'-m','pytest','-q','tests','--basetemp',str(clean/'test-tmp'/uuid.uuid4().hex),
        '--junitxml',str(folder/'clean-core-tests.xml')],args.max_experiment_hours,folder/'clean-core-tests.xml',cwd=clean)
    execute(folder,'clean-tiny-upstream',[str(python),'scripts/verify_upstream.py'],args.max_experiment_hours,cwd=clean)
    shutil.copy2(clean/'artifacts/upstream.json',folder/'clean-tiny-upstream.json')
    output=destination/'clean-parity'
    execute(folder,'clean-pretrained-upstream-smoke',[str(python),'scripts/finalization_worker.py','parity',str(output)],
            args.max_experiment_hours,output/'result.json',cwd=clean)
    result=read_json(output/'result.json');original=read_json(folder/'parity/result.json')
    assert result['shared']['status']=='passed'
    for state in ['identity','nontrivial']:
        for key in ['max_score_error','max_loss_error','max_gradient_error']:
            assert result[state][key]==original[state][key]
    shutil.copy2(output/'result.json',folder/'clean-pretrained-result.json')
    execute(folder,'clean-verify-evidence',[str(python),'scripts/finalize.py','--offline','--verify-only'],args.max_experiment_hours,cwd=clean)
    execute(folder,'clean-regenerate-report',[str(python),'scripts/finalize.py','--offline','--report-only'],args.max_experiment_hours,cwd=clean)
    result={'status':'passed','at':now(),'source_sha256':source_manifest()['sha256'],
        'lock_sha256':file_hash(ROOT/'requirements.lock'),
        'proof_artifacts':{name:file_hash(folder/name) for name in ['clean-isolation.json','clean-core-tests.xml','clean-tiny-upstream.json','clean-pretrained-result.json']},
        'root_relative':clean.relative_to(ROOT).as_posix(),'fresh_venv':True,'working_venv_copied':False,
        'offline_cached_packages':True,'verified_cached_model_data_files':cache_files,
        'scope':'Full offline core including toy resume; pinned tiny TRL; actual pretrained identity/nontrivial parity and three-strategy 3-update smoke; numeric raw-evidence verification and report regeneration',
        'not_repeated':'Full historical and new timing/learning campaigns were not repeated in the fresh environment'}
    atomic_json(record_path,result)
    return result
