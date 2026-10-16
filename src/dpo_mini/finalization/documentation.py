"""Current documentation derived from the same evidence snapshot as the paper."""
from ..provenance import ROOT,read_json,file_hash,now


def write(folder,s,h):
    relative=folder.relative_to(ROOT).as_posix()
    resources=s['resources'];core=s['checks']['core'];clean=s['checks'].get('clean',{})
    headline=('The historical cold-cache median measured training block was 720.766 s versus 1007.559 s for a separate reference (28.46% reduction). '
              'Whole benchmark processes were 1184.306 versus 1324.106 s (10.56% reduction). These are three trials of five warm-up plus twenty measured updates, not the learning workflow. '
              'The historical single-seed sixteen-update workflows took 1409.365 versus 1518.458 s. Test loss ended at 0.690726; strict positives were 39/64, with zero ties.')
    new='New benchmark completion is pending.'
    if s['cohort_complete']:
        t=s['followup']['timing'];a,b,c=[t[k] for k in ['separate_reference','shared_base_reference','cached_shared_base']]
        new=(f'New A/B/C median training times: {a["training_seconds"]["median"]:.3f} / {b["training_seconds"]["median"]:.3f} / {c["training_seconds"]["median"]:.3f} s. '
             f'Median cold whole-process times: {a["whole_process_seconds"]["median"]:.3f} / {b["whole_process_seconds"]["median"]:.3f} / {c["whole_process_seconds"]["median"]:.3f} s. '
             'A = separate reference; B = shared frozen base; C = new shared-base cache. Each has three fresh-process trials, five warm-up and twenty measured updates. Do not pool with historical trials.')
    commands='''```powershell
python scripts/bootstrap.py --profile toy --offline
.\\.venv\\Scripts\\python.exe -m pytest -q
.\\.venv\\Scripts\\python.exe scripts/finalize.py --offline --max-experiment-hours 6 --resume
.\\.venv\\Scripts\\python.exe scripts/finalize.py --offline --report-only
.\\.venv\\Scripts\\python.exe scripts/finalize.py --offline --verify-only
```
'''
    readme=f'''# DPO-Mini

An independent original-sigmoid DPO implementation and measured study of fixed-reference costs. Technical report by Aryan Golbaghi.

Read [the paper](paper/paper.pdf), [results](RESULTS.md), [the finalization report](docs/FINALIZATION_REPORT.md), and [owner review](docs/OWNER_REVIEW.md).

{headline}

{new}

Pretrained TRL parity **does not pass the original FP32 absolute thresholds**. Identical logits exposed different CPU reduction arithmetic; diagnostic FP64 reducers recover matching adapter gradients. The original failures and a single explicit numerical adjudication remain in the report. Shared-base reference invariants and short matched trajectories pass. This is function-level evidence, not complete TRL Trainer equivalence.

## Installation and local commands

Python 3.12 and an isolated project-local environment are used. `requirements.lock` pins the actual stack. `bootstrap.py` creates `.venv` and installs through uv when available, or pip; it never changes global packages. A fresh compact study can be started with `python scripts/bootstrap.py --profile compact`. This may download public pinned inputs and run experiments. Do not use that command merely to inspect the included report.

{commands}
The toy bootstrap route requires the locked dependencies already cached in offline mode; it executes a small synthetic pipeline. The direct pytest command is the lightweight local check. Integration checks are separate and need the pinned local SmolLM2 cache. `--offline` never falls back to online acquisition. The finalizer consumes this finalized evidence and compatible completed stage manifests; it does not repeat expensive completed trials when resumed. `--report-only` and `--verify-only` do not train or download.

The finalization entry point operates on `artifacts/finalization/latest.json`. It is distinct from the original `reproduce.py` protocol. Historical plans and runs are immutable. Source changes affecting the measurement worker invalidate plan compatibility. Presentation-only changes receive a new report record without relabelling old trials.

## Evidence and reproducibility scope

The active pass is `{relative}`. Its `evidence/` directory supports report regeneration and independent arithmetic checks; `audit/` retains independent raw-parquet reconstruction and checkpoint comparisons. `runs/` contains fresh-process specs, stages, parent timers and numerical outcomes. `ledger.json` includes failed attempts and completed leaf subprocess costs. `claims.json` links tables to hashes, fields and formulas.

The clean reproduction uses a genuinely fresh `.venv` in a scoped source copy, verified offline cached packages and verified public model/data caches. Scope: all local tests including toy resume, pinned tiny/pretrained TRL checks, the three-strategy short trajectory, numeric artifact verification and report regeneration. It does not repeat all timing or learning experiments. Current clean status: `{clean.get('status','pending')}`.

The portable `dist/dpo-mini-review-bundle.zip` omits weights, giant checkpoints, full datasets, caches, virtual environments, credentials and unrelated files. After extraction and installation of the locked environment, run `python scripts/finalize.py --offline --report-only` and `--verify-only`. The verifier explicitly reports excluded checkpoint/weight limitations. The archive manifest and the external archive checksum/verification record describe integrity; a bundle without model/data payloads cannot perform complete offline pretrained retraining.

## Limits and attribution

One nonexclusive CPU host, one small filtered dataset, one learning seed and three timing repetitions. No human quality ratings, paid judge or general helpfulness/truthfulness claim. Recorded historical stage aggregates reconcile the workflow difference; finer evaluation/generation and per-split cache attribution remains unresolved. See [third-party notices](THIRD_PARTY_NOTICES.md), primary references in the paper, and the source registry. No blanket license grant for project code, datasets or weights is invented.
'''
    (ROOT/'README.md').write_text(readme,encoding='utf8')
    (ROOT/'RESULTS.md').write_text(f'''# Measured results

{headline}

{new}

All timings are seconds on AMD Ryzen 5 PRO 7540U, four computation threads, FP32 eager CPU. The benchmark measures 320 pairs after 80 warm-up pairs. The actual learning workflow processes 256 pairs, evaluates 64 validation and 64 test pairs before and after, generates eight short responses at each phase, and writes a final checkpoint.

Core: {core['passed']} passed, {core['failures']} failed, {core['errors']} errors, {core['skipped']} skipped in the final working suite. Earlier runner-setup failures remain recorded.

Pretrained upstream: original FP32 score/loss thresholds fail in both adapter states; the trained-adapter gradient maximum is 2.0102597773e-6, narrowly above 2e-6. Same-input/logit diagnostics isolate arithmetic; FP64 reductions give score errors below 2e-13 and identical FP32 adapter gradients. A single revised numerical allowance is documented. No end-to-end TRL Trainer claim.

Shared base: reference scores, base parameters/buffers, optimizer membership, adapter/mode/gradient restoration including exceptions, and three-update trajectories are checked. Reference scoring has no gradient path; policy adapters update; cached training performs zero reference forwards.

Historical checkpoint file bytes and all saved tensor/optimizer/scheduler/RNG fields match for the paired learning files. Adapter hashes alone are not used to make this claim. Historical loss/margin aggregates and filtering were independently reconstructed.

The old equal-setup one-pass projection makes caching 7.505 s slower (break-even 1.033 passes), while the observed old learning workflow was 109.093 s faster. They have different boundaries and workload costs. The new original-design replay provides its own exclusive stage accounting; it cannot retroactively identify unrecorded historical costs.

Unsupported conclusions to avoid: statistical significance from three trials; learning-seed variability from one seed; improved general truthfulness/helpfulness; human preference gains; exact cross-platform identity; universal FP32 upstream equivalence; complete offline retraining from the lightweight bundle. Owner approval is not recorded.
''',encoding='utf8')
    status=[('Historical audit','passed: independent raw arithmetic, full membership/filter reconstruction, archived source identity, checkpoint file/field equality'),
            ('Core implementation correctness',f'{core["passed"]} passed, {core["failures"]} failed, {core["errors"]} errors, {core["skipped"]} skipped'),
            ('Pretrained LoRA upstream','completed with qualification: original FP32 bounds failed; identical-logit and FP64 diagnostics explain numerical discrepancy; one revised bound record'),
            ('Shared-base correctness','passed: pretrained before/after reference, base/buffer, gradient/state restoration and matched short trajectories'),
            ('Instrumented accounting','completed' if len(s['followup']['replay'])==2 else 'running/pending'),
            ('Three-arm benchmark','completed: 3 trials per arm, 5 warm-up + 20 measured updates' if s['cohort_complete'] else 'running/pending'),
            ('Clean environment',clean.get('status','pending')),
            ('Manuscript/PDF','generated from shared structured source; explicit page-inspection record is separate'),
            ('Review bundle','manifest, checksum and extraction-test results recorded separately in dist'),
            ('Owner approval','not recorded; all human approval fields remain unchecked')]
    rows='\n'.join('|'+a+'|'+b+'|' for a,b in status)
    (ROOT/'docs/FINALIZATION_REPORT.md').write_text(f'''# Finalization report

Evidence snapshot generated {now()}. The authoritative later leaf-process total is in `{relative}/ledger.json`; resource figures below have this report-generation cutoff.

| Stage | Status and exact scope |
|---|---|
{rows}

## Actual changes

Preserved the original source, paper, analysis, plans, run IDs and interruptions before edits. Added the pinned shared-base reference context, canonical tensor hashing, reference-first follow-up loop, actual pretrained TRL fixtures, negative control, numerical diagnosis, independent parquet/raw-score audit, exclusive workflow timers, sequential fresh-process A/B/C protocol, offline finalizer, clean reproduction and portable review packaging. Original mathematical scoring and the historical training loop remain unchanged.

The old report's broad upstream wording is narrowed: tiny-model parity did not establish pretrained parity. Actual pretrained FP32 comparisons exceed the original tolerances. The report preserves that failure and explains the reduction difference using bit-identical logits and FP64 diagnostic results. No claim of full TRL Trainer equivalence is made. 'Across 1 seeds' is replaced by 'for the single paired seed'. Byte-identical historical checkpoint files are now supported by file hashes plus complete saved-field comparison; other scopes are named separately.

Three timing boundaries are reported separately. New C uses shared-base construction, whereas historical C and the replay use a separate reference. Original and follow-up cohorts are never pooled. The old 7.505 s projected disadvantage and 109.093 s observed learning-workflow advantage are both retained. Historical aggregate setup, training, combined evaluation/generation, checkpoint and residual differences reconcile that total. Finer historical phase and per-split cache attribution remains unresolved; the replay only measures its own detailed costs.

## Executed commands and failures

Working commands include `python scripts/finalize.py --offline --stage core`, `--stage audit`, `--stage parity`, `--stage diagnose`, `--stage precision`, and `--max-experiment-hours 6 --resume --stage experiments`. Windows execution uses the project `.venv/Scripts/python.exe`. The first core runner omitted the test temporary parent (13 passed, 4 setup errors); its immediate repair passed all 17 tests then present. The final suite, after additional RNG/resume and evidence-gate coverage, has {core['passed']} passes. The first diagnostic hook assumed keyword input IDs; installed TRL passed them positionally. That diagnostic failed before producing a result, then the bounded repair completed. Those initial failures and all later failed verification attempts remain in the ledger with their costs and logs. There was no automatic-approval rejection or administrator invocation.

The first clean artifact-verification attempt encountered the legacy paper's list-form structured blocks. The new format uses a dictionary. The verifier now identifies the legacy format explicitly, and manuscript/CSV checks apply to the regenerated finalization format. The affected report and verification stages were rerun in the clean environment. Completed numerical checks were retained only after a recorded source-difference and artifact-hash check; their computation source remains separately archived. The review copy also removes local machine hostnames from JUnit metadata while retaining the untouched local originals.

{commands}
The full default finalizer can consume compatible manual stages, execute missing planned measurements sequentially, perform clean checks, rebuild the report and package it. Resume validates immutable plans, measurement-source identity and completed result hashes; it does not equate process exit zero with original-tolerance parity success. The numerical adjudication is a separate required gate. Reports are generated only from recorded evidence.

Clean reproduction status: `{clean.get('status','pending')}`. It creates a new venv without system site packages and installs the exact lock offline from the existing uv cache; it does not copy/rename the working venv. Package dependency consistency, module origins and installed RECORD hashes are checked. Cached public model and prepared-data files are explicitly copied and rehashed. Full tests include toy resume. Pretrained checks include the same fixed initial/trained fixtures and the short three-strategy trajectory. Raw numeric verification and paper/table regeneration execute in the copy. The complete benchmark/learning campaigns are not repeated there. See clean-install, clean-isolation, clean-core XML, clean-pretrained and ledger logs.

## Resource accounting

Historical measured leaf subprocesses: {resources['historical_measured_process_seconds']:.6f} s; historical total including allowances: {resources['historical_accounted_seconds']:.6f} s. New measured leaf subprocesses at this snapshot: {resources['new_measured_process_seconds']:.6f} s. Available cumulative measured total: {resources['cumulative_measured_seconds']:.6f} s; cumulative including historical allowances: {resources['cumulative_accounted_seconds']:.6f} s. Parent wait/orchestration time is not added to children. Actual elapsed wall time at the snapshot: {resources['elapsed_wall_seconds']:.3f} s, including editing and waits. Failed/retried subprocesses count. No new dependency/model/data payload download occurred in offline execution. Final storage and subsequent verification costs are retained in the final resource records.

Additional limits are 21600 experiment-process seconds, 5400 seconds per child, 5 GiB downloads and 12 GiB storage. The frozen full-work estimate retained a 2700-second analysis/reproduction reserve. No Qwen training retry, larger data subset, hyperparameter search, GPU acquisition, cloud service, paid judge or publication occurred.

## Delivery and residual limits

Final artifacts: `paper/paper.pdf`, `paper/paper.md`, `paper/main.tex`, `paper/references.bib`, generated tables/figures, and `dist/dpo-mini-review-bundle.zip`. The PDF uses ReportLab with Matplotlib equations/plots and PyMuPDF page renders; no unavailable LaTeX compiler success is claimed. Every final page is subject to explicit image inspection recorded in the pass. The bundle has relative members and a content manifest; a separate SHA-256 sidecar and extraction-test record avoid self-referential hashing. Excluded weights/checkpoints/raw datasets are explicitly unavailable in bundle-level verification.

One shared CPU, one learning seed, small filtered AI-feedback data, unknown pretraining overlap, previously examined test outputs, three timing repetitions and unscored short generations limit interpretation. Owner code/results/license review and release approval remain required.
''',encoding='utf8')
    (ROOT/'docs/PAPER_AUDIT.md').write_text('''# Paper audit

Mechanical checks: regenerate from shared structured blocks; recompute raw per-example losses/margins and strict-positive/tie counts; recalculate medians and paired percentages; check all run work counts, stage sums and residuals; verify evidence manifests and source identities; inspect every rendered final page. Rendering is distinct from visual approval and TeX source emission is distinct from successful TeX compilation. Final page hashes and inspection scope are recorded separately.

Natural-language checks: historical and follow-up cohorts remain separate; training block, complete benchmark and actual learning workflow are named wherever a headline duration appears. Three timing trials are not three learning seeds. No significance, novelty, state-of-the-art or general model-quality claim is made. Original pretrained FP32 tolerance failures remain visible. The revised empirical gradient bound is not presented as a theorem or a preregistered threshold. FP64 scoring diagnostics are not described as full FP64 model training. Function-level parity is not end-to-end Trainer equivalence.

The original historical checkpoint files were independently rehashed and all saved fields compared, supporting that narrow byte-identity statement. New adapter-only checks identify the unchanged base separately. Canonical hash metadata includes tensor names/shapes/dtypes/bytes and tied aliases; shared-base checks include nonpersistent buffers. Bundle verification cannot recheck excluded tensor files and says so.

The old projected one-pass disadvantage remains alongside the actual workflow advantage. New stage timings are not assigned retrospectively to old runs. Single-seed loss and margin outcomes do not imply better human preference. Generation ratings remain empty. Bibliography uses a pinned original-code commit rather than a moving API endpoint. PEFT documentation 0.17.0 and installed 0.17.1 are distinguished.
''',encoding='utf8')
    (ROOT/'docs/INTERVIEW_NOTES.md').write_text('''# Implementation and interpretation notes

DPO uses L=softplus(-beta*m), where m is the policy chosen-minus-rejected score difference minus the same reference difference. Each score is a sum over completion tokens. At equality L=log(2), but policy-score derivatives are -beta/2 and +beta/2. The worked scores policy [-2,-4], reference [-3,-3] give m=2 and softplus(-0.2). A fixed-reference float64 gradcheck and analytic checks support the objective.

Shift logits one position against completion targets. The first completion token is predicted from the last prompt position; the assistant end token is included. Prompt context remains visible but is not directly scored. Pad/end ID aliasing requires positional attention/completion masks. Select valid positions before log-softmax/gather; zero times NaN is not safe masking. Completion sums, not length averages, define this objective.

LoRA learns low-rank updates while the base is frozen. Only verified q_proj/v_proj adapter parameters enter AdamW. With an identity initial adapter and a base reference, disabling adapters gives the reference. That assumption fails for a different trained reference adapter. The context restores enablement, active adapters, requires_grad, per-module modes and existing gradient objects even on exception. Reference scoring occurs before the policy backward graph in the new loop, under no_grad. Base parameters and all buffers are hashed before/after; adapters demonstrably change.

Cache identity includes IDs, all token IDs/masks, fixed reference identity, model/tokenizer revision, template/end-token rule, length, precision/backend, microbatch and follow-up software/thread settings. JSON uses full Python float precision and atomic persistence. Reordering is safe by ID; duplicate/missing IDs, changed content, corruption and partial writes fail. Cached training has zero reference forwards. A changed reference or tokenization invalidates its cache.

Accumulation weights each microbatch by pairs/actual window size; the partial final window is tested. AdamW can amplify near-zero FP32 gradient differences, which motivated a float64 accumulation-invariance fixture. Checkpoints write to a temporary file, fsync, then replace; a simulated failed writer preserves the previous file. Resume restores model, optimizer, integer schedule position, data position and Python/NumPy/Torch RNG states at update boundaries.

Real debugging evidence: the original FP32 batched accumulation comparison exposed near-zero gradient amplification; the implementation-invariant test uses double precision. Windows temporary-directory/flush issues were fixed and logged. This pass's first test runner omitted a temporary parent, and the first TRL hook assumed keyword inputs. Both bounded repairs preserve failure records. The actual pretrained parity failure is more consequential: identical logits yield different FP32 reductions. A selected-token arithmetic diagnostic finds differences of about 1.8e-5 per token between CPU log_softmax and selected-logit-minus-logsumexp. Long sums exceed the original absolute score tolerance. A separate full-forward FP64-reduction diagnostic restores score/loss agreement and identical adapter gradients. Original failures remain; one explicit revised bound is qualified.

Distinguish file byte identity; canonical full-model tensor identity; adapter identity plus unchanged base; and numerical closeness. A hash over adapter weights says nothing by itself about optimizer, RNG, scheduler or serialized-file metadata. The two historical learning files were actually compared at all those scopes.

Runtime reduction is 100*(baseline-comparison)/baseline. Throughput increase has a different denominator. Separate the 20-update measured block, 25-update benchmark process and 16-update learning workflow. A/B compares reference residency, B/C compares caching against shared base, A/C combines changes. Cold C constructs a new cache in every process. Warm cache reuse is a different historical arm. Medians and paired-trial percentages are separate statistics.

Amortization: Tu=Su+E*U, Tc=Sc+P+E*C, with Sc excluding P. If U>C, caching wins for E>(Sc-Su+P)/(U-C). This stationary training-only projection omits evaluations/generation/checkpoint costs. The historical 7.505 s projected disadvantage and 109.093 s workflow advantage are not interchangeable results. Recorded stage aggregates reconcile the old workflow total; finer phase and per-split cache attribution is unresolved. New instrumented replay measures only itself.

One seed and 64 test pairs do not support broad quality claims. Initial margins are ties because policy equals reference, not because the language model lacks ability. Short greedy examples have no human ratings.
''',encoding='utf8')
    (ROOT/'docs/OWNER_REVIEW.md').write_text('''# Owner review and release approval

Review the implementation, evidence and reported limitations using the checklist below.

- [ ] Review and explain the original-sigmoid derivation, score sums and causal masks.
- [ ] Inspect shared-base freezing, optimizer membership and exception restoration.
- [ ] Review the actual pretrained original-tolerance failures and numerical adjudication.
- [ ] Check the raw timing boundaries, all trials, exclusions and residuals.
- [ ] Review the one-seed learning outcomes and unsupported quality claims to avoid.
- [ ] Inspect checkpoint/file/tensor equality scopes and clean-reproduction limits.
- [ ] Review model, dataset, source-code and documentation licenses/notices.
- [ ] Read every page of the final report and independently inspect key artifacts.
- [ ] Confirm the portable archive contains only intended release material.
- [ ] Authorize release/publication explicitly, if desired.

Reviewer name: not recorded. Review date: not recorded. Approval: not recorded. These fields must not be checked automatically by an agent.
''',encoding='utf8')
