"""Portable, scoped review archive with extraction and report-only execution checks."""
import hashlib
import json
import re
import platform
import shutil
import sys
import uuid
import zipfile
from pathlib import Path,PurePosixPath
from ..provenance import ROOT,read_json,atomic_json,file_hash,now
from .report import sanitized


def build(folder,execute,args):
    dist=ROOT/'dist';dist.mkdir(exist_ok=True)
    stage=dist/('review-stage-'+uuid.uuid4().hex[:8]);stage.mkdir()
    sources=[]
    for group in ['src','scripts','tests','configs']:
        sources += [p for p in (ROOT/group).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc']
    sources += [ROOT/name for name in ['README.md','RESULTS.md','pyproject.toml','requirements.lock','THIRD_PARTY_NOTICES.md']]
    sources += list((ROOT/'licenses').glob('*.txt'))
    sources += [ROOT/'docs'/name for name in ['FINALIZATION_REPORT.md','PAPER_AUDIT.md','INTERVIEW_NOTES.md','OWNER_REVIEW.md']]
    sources += [ROOT/'paper'/name for name in ['paper.pdf','paper.md','main.tex','references.bib','report_blocks.json','render-record.json']]
    structured=read_json(ROOT/'paper/report_blocks.json')
    for block in structured['blocks']:
        if block['kind']=='figure':sources.append(ROOT/'paper'/block['path'])
        if block['kind']=='table':sources.append(ROOT/'paper/tables'/(block['name']+'.csv'))
    sources += list((ROOT/'paper/figures').glob('equation-*.png'))
    relative=folder.relative_to(ROOT)
    sources += list((folder/'evidence').glob('*.json'))
    sources += [folder/name for name in ['plan.json','summary.json','claims.json','verification.json','parity-adjudication.json','parity-revised-bound-declaration.json','pdf-visual-inspection.json'] if (folder/name).exists()]
    sources += [ROOT/'artifacts/finalization/latest.json']
    sources += [ROOT/'artifacts/prepared/compact'/name for name in ['manifest.json','config.json']]
    sources += list((folder/'logs').glob('*.log'))
    sources += [folder/name for name in ['core-tests.xml','clean-core-tests.xml','clean-install.json','clean-reproduction.json','clean-computation-source.json','clean-targeted-repair.json','compatibility-checks.json'] if (folder/name).exists()]
    files=[];redactions={}
    for source in sorted(set(sources)):
        target=stage/source.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        if source.suffix.lower() in ['.json','.md','.txt','.log','.xml','.py','.yaml','.toml','.lock','.bib','.tex']:
            text=source.read_bytes().decode('utf8')
            # Preserve source locally; redact only release-copy machine paths.
            for prefix in [str(ROOT),str(ROOT).replace('\\','/'),str(ROOT).replace('\\','\\\\')]:text=text.replace(prefix,'.')
            for prefix in [str(Path.home()),str(Path.home()).replace('\\','/'),str(Path.home()).replace('\\','\\\\')]:text=text.replace(prefix,'<USER_HOME>')
            if platform.node():text=re.sub(re.escape(platform.node()),'REDACTED_HOST',text,flags=re.I)
            target.write_bytes(text.encode('utf8'))
        else:shutil.copy2(source,target)
        if file_hash(source)!=file_hash(target):
            redactions[source.relative_to(ROOT).as_posix()]={'local_original_sha256':file_hash(source),
                'release_copy_sha256':file_hash(target),'policy':'Project/user paths and local machine hostname redacted only in the release copy'}
        files.append(target)
    # Evidence hashes must remain exact; no silent sanitizing of numeric records.
    manifest=read_json(stage/relative/'evidence/manifest.json')
    for name,sha in manifest['files'].items():
        assert file_hash(stage/relative/'evidence'/name)==sha,('sanitization changed evidence',name)
    atomic_json(stage/'RELEASE_REDACTIONS.json',{'files':redactions,'numeric_evidence_unchanged':True})
    files.append(stage/'RELEASE_REDACTIONS.json')
    exclusion={'scope':'Portable numeric-evidence integrity; full local model/checkpoint verification is unavailable here',
        'excluded':['virtual environments','dependency and model caches','all model weights','large checkpoints','full raw datasets','raw dataset text/token fixtures','credentials','.env files','Git internals','unrelated files'],
        'retraining':'Obtain pinned public model/data and install locked dependencies separately; this archive alone is not complete offline pretrained retraining',
        'local_historical_checkpoint_audit':'Numeric audit includes file and tensor identity results; excluded files cannot be directly reopened from this archive'}
    atomic_json(stage/'BUNDLE_SCOPE.json',exclusion);files.append(stage/'BUNDLE_SCOPE.json')
    secret_patterns=[re.compile(r'(?<![A-Za-z0-9])sk-[A-Za-z0-9]{24,}'),re.compile(r'AKIA[0-9A-Z]{16}'),
                     re.compile(r'ghp_[A-Za-z0-9]{30,}'),re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')]
    issues=[];scanned=0
    for path in files:
        if path.suffix.lower()=='.png':continue
        if path.suffix.lower()=='.pdf':
            import fitz
            with fitz.open(path) as document:
                content='\n'.join(page.get_text() for page in document)
        else:content=path.read_text(encoding='utf8')
        scanned+=1
        if any(p.search(content) for p in secret_patterns):issues.append({'file':path.relative_to(stage).as_posix(),'category':'secret-like pattern'})
        if re.search(r'[A-Za-z]:[\\/]+Users[\\/]',content,re.I):issues.append({'file':path.relative_to(stage).as_posix(),'category':'private absolute user path'})
        if platform.node() and platform.node().lower() in content.lower():issues.append({'file':path.relative_to(stage).as_posix(),'category':'local machine hostname'})
    scan={'status':'passed' if not issues else 'blocked','files_scanned':scanned,'included_files':len(files),'issues':issues,
          'scope':'Only intended release files; no credential directories searched; matching values never printed'}
    atomic_json(folder/'release-scan.json',scan)
    if issues:raise ValueError('Release scan requires review; file/category only recorded, no secret values printed')
    file_manifest={p.relative_to(stage).as_posix():{'sha256':file_hash(p),'bytes':p.stat().st_size} for p in sorted(files)}
    atomic_json(stage/'BUNDLE_MANIFEST.json',{'files':file_manifest,'excludes_self':True,'created_at':now()})
    archive=dist/'dpo-mini-review-bundle.zip'
    if archive.exists():
        previous=folder/'bundle-revisions';previous.mkdir(exist_ok=True)
        shutil.copy2(archive,previous/(file_hash(archive)+'.zip'))
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(stage.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(stage).as_posix())
    checksum=file_hash(archive)
    (dist/'dpo-mini-review-bundle.sha256').write_text(checksum+'  dpo-mini-review-bundle.zip\n',encoding='ascii')
    extracted=dist/('extracted-review-'+uuid.uuid4().hex[:8]);extracted.mkdir()
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            path=PurePosixPath(info.filename)
            assert not path.is_absolute() and '..' not in path.parts and ':' not in info.filename and '\\' not in info.filename
            assert (info.external_attr>>16)&0o170000 != 0o120000,'Symlink archive members forbidden'
            target=(extracted/info.filename).resolve()
            assert target.is_relative_to(extracted.resolve())
        assert z.testzip() is None
        z.extractall(extracted)
    for name,row in read_json(extracted/'BUNDLE_MANIFEST.json')['files'].items():assert file_hash(extracted/name)==row['sha256']
    before={p.name:file_hash(p) for p in (extracted/relative/'evidence').glob('*.json')}
    execute(folder,'bundle-verify',[sys.executable,'scripts/finalize.py','--offline','--verify-only'],args.max_experiment_hours,cwd=extracted)
    execute(folder,'bundle-regenerate',[sys.executable,'scripts/finalize.py','--offline','--report-only'],args.max_experiment_hours,cwd=extracted)
    after={p.name:file_hash(p) for p in (extracted/relative/'evidence').glob('*.json')}
    assert before==after,'Report-only changed raw numeric evidence'
    verification=read_json(extracted/relative/'verification.json')
    assert verification['historical_checkpoint_files_rehashed']==0
    summary=read_json(extracted/relative/'summary.json')
    original=read_json(folder/'summary.json')
    assert summary==original,'Extracted report calculations differ'
    result={'status':'passed','archive':'dist/dpo-mini-review-bundle.zip','sha256':checksum,'bytes':archive.stat().st_size,
        'members':len(file_manifest)+1,'relative_members_verified':True,'content_hashes_verified':True,
        'release_scan':scan,'extracted_report_only_executed':True,'extracted_verify_only_executed':True,
        'raw_numeric_evidence_unchanged':True,'summary_matches':True,'verification_scope':verification['scope'],
        'at':now(),'owner_release_approval':'not recorded'}
    atomic_json(dist/'dpo-mini-review-bundle.verification.json',result)
    atomic_json(folder/'bundle-verification.json',result)
    return result
