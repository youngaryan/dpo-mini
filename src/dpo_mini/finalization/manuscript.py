"""One structured manuscript drives Markdown, TeX, figures, tables and PDF."""
import csv
import html
import re
from pathlib import Path
from ..provenance import ROOT,read_json,atomic_json,file_hash,now
from .report import ARM_LABELS


def fmt(x,n=3):return 'unavailable' if x is None else f'{x:.{n}f}'
def reduction(a,b):return 100*(a-b)/a


def write(folder,s,h,follow):
    paper=ROOT/'paper';paper.mkdir(exist_ok=True)
    (paper/'tables').mkdir(exist_ok=True);(paper/'figures').mkdir(exist_ok=True)
    blocks=[];claims=[]
    def heading(text):blocks.append({'kind':'heading','text':text})
    def p(text):blocks.append({'kind':'paragraph','text':text})
    def equation(latex):blocks.append({'kind':'equation','latex':latex})
    def table(name,title,columns,rows,origins):
        blocks.append({'kind':'table','name':name,'title':title,'columns':columns,'rows':rows})
        with (paper/'tables'/(name+'.csv')).open('w',newline='',encoding='utf8') as f:
            writer=csv.writer(f);writer.writerow(columns);writer.writerows(rows)
        for origin in origins:
            path=(folder/origin['path']) if origin['path'].startswith('evidence/') else (ROOT/origin['path'])
            if path.is_file():
                actual=file_hash(path)
                if 'sha256' in origin:assert origin['sha256']==actual
                else:origin['sha256']=actual
        claims.append({'id':'table:'+name,'status':'verified computation from retained evidence','table_cells':rows,
            'columns':columns,'sources':origins,'selection':'All complete compatible trials; interrupted attempts excluded, retained in audit',
            'rounding':'Only displayed cells rounded; full precision remains in evidence and summary',
            'csv_path':'paper/tables/'+name+'.csv','csv_sha256':file_hash(paper/'tables'/(name+'.csv'))})
    def figure(name,caption):blocks.append({'kind':'figure','path':'figures/'+name+'.png','caption':caption})
    old=h['timing'];u=old['uncached_reference'];c=old['cached_reference'];w=old['warm_cached_reference']
    learning={r['arm']:r for r in s['historical']['learning']}
    lu,lc=learning['uncached_reference'],learning['cached_reference']
    new=s['followup']['timing'];complete=s['cohort_complete']
    parity=s['checks']['parity'];precision=s['checks']['precision']['results']
    env=read_json(folder/'evidence/environment.json');resources=s['resources']
    sources=read_json(folder/'evidence/sources.json')
    old_origins=[{'path':r['artifact'],'sha256':r['artifact_sha256'],'run_id':r['run_id'],
        'fields':['training_seconds','whole_process_seconds','training_memory.sampled_peak_rss_bytes'],
        'formula':'median/min/max over three completed trials; RSS bytes / 2^20'} for r in h['runs'] if r['kind']=='benchmark']
    new_origins=[{'path':r['index']['path']+'/metrics.json','sha256':r['index']['metrics_sha256'],
        'fields':['training_seconds','training_memory.sampled_peak_rss_bytes','stages_seconds'],
        'parent_path':r['index']['path']+'/parent-timing.json','parent_sha256':r['index']['parent_timing_sha256'],
        'formula':'Median/min/max over three trials of each arm; pairwise reduction=100*(baseline-comparison)/baseline'} for r in follow if r['metrics']['kind']=='benchmark']
    blocks.append({'kind':'title','text':'DPO-Mini: A Reproducible Correctness and Reference-Caching Study of Direct Preference Optimization'})
    p('Technical report. Aryan Golbaghi. Public edition: 16 October 2026. Experiments completed 15 October 2026.')
    heading('Abstract')
    abstract=(f'We audit an independent original-sigmoid DPO implementation on SmolLM2-135M-Instruct with rank-16 LoRA, FP32 eager CPU execution, four threads, and a filtered 256/64/64 preference-pair split. The historical three-trial benchmark used five warm-up plus twenty measured updates per arm. Separate-reference versus cold-cache median training blocks were {u["training_seconds"]["median"]:.3f} and {c["training_seconds"]["median"]:.3f} s; complete benchmark processes were {u["whole_process_seconds"]["median"]:.3f} and {c["whole_process_seconds"]["median"]:.3f} s. These boundaries differ from the single-seed, sixteen-update learning workflows ({lu["whole_process_seconds"]:.3f} and {lc["whole_process_seconds"]:.3f} s). ')
    if complete:
        nb,nc=new['shared_base_reference'],new['cached_shared_base']
        abstract+=(f'The new balanced three-trial comparison gives shared-base versus cold shared-cache median training times of {nb["training_seconds"]["median"]:.3f} and {nc["training_seconds"]["median"]:.3f} s, and whole-process times of {nb["whole_process_seconds"]["median"]:.3f} and {nc["whole_process_seconds"]["median"]:.3f} s. ')
    abstract+=('Pretrained TRL function comparisons expose an FP32 tolerance failure, traced to distinct score-reduction kernels on identical logits; diagnostic FP64 reductions recover matching adapter gradients. Shared-base reference invariants and short trajectories pass. Independent raw-score and dataset audits support the historical numerical claims. Recorded historical aggregates reconcile the total runtime difference, while finer per-phase attribution remains unresolved. These small, single-machine measurements establish scoped computational behavior, not general model quality or end-to-end TRL trainer equivalence.')
    p(abstract)
    heading('1. Research questions and prior work')
    p('This study asks whether the objective and completion scores are implemented correctly, whether a fixed reference can be reused without changing the measured trajectory, and how reference residency and cache construction alter observed costs. A second uncached reference model is a useful simple baseline, but LoRA also permits a reference forward through the same frozen base with the policy adapter disabled. The follow-up therefore separates residency from caching. Neither DPO nor reference-score caching is proposed as a new method.')
    p('DPO derives a preference objective using a policy/reference log-probability ratio [R1]. The original public implementation supplies a primary implementation reference [R2]. TRL exposes DPO training and reference-log-probability precomputation [R3,R4]. LoRA represents a trainable update through low-rank matrices [R13]; the PEFT adapter API provides the disabled-adapter context used here [R12,R15]. This report distinguishes mathematical checks, executable function comparisons, and complete optimizer-loop experiments. Agreement at one level does not establish agreement at every level.')
    heading('2. Objective and scoring contract')
    equation(r's_\theta(x,y)=\sum_{t=1}^{|y|}\log p_\theta(y_t\mid x,y_{<t})')
    equation(r'm_i=(s_\theta(x_i,y_i^+)-s_0(x_i,y_i^+))-(s_\theta(x_i,y_i^-)-s_0(x_i,y_i^-))')
    equation(r'\mathcal{L}=\frac{1}{N}\sum_{i=1}^{N}\log(1+\exp(-\beta m_i)),\qquad\beta=0.1')
    p('Scores are completion-token sums. The first completion token is predicted from the final prompt position; the final assistant-ending token is also scored. The generation prompt is rendered once using the pinned chat template. Each completion is suffix-encoded without added special tokens and receives exactly one assistant end token. Empty responses, embedded ending tokens and sequences above the length cap are rejected. This construction avoids assuming that a fully rendered conversation has the same token prefix as a separately rendered generation prompt [R8].')
    p('Prompt tokens remain visible to attention and can receive contextual gradients, but are excluded from direct score reduction. Padding is excluded by position, including when padding and end IDs alias. Masked invalid labels and nonfinite logits are never gathered or multiplied by zero: valid positions are selected before log-softmax. The independent scorer uses log-softmax, gather and an indexed sum; the DPO loss uses stable negative log-sigmoid. These choices define a numerical implementation, not a promise of bitwise equivalence to every algebraically equivalent kernel.')
    p('When policy and reference scores are equal, the loss is log(2), and the derivatives with respect to chosen and rejected policy scores are -0.05 and +0.05. For policy scores [-2,-4] and reference scores [-3,-3], the relative margin is 2 and the loss is softplus(-0.2). The score derivatives are not zero at initialization. Gradcheck uses float64 and a fixed reference; separate tests cover shift, causal masking, unequal lengths, prompt context and partial accumulation windows.')
    heading('3. Data, model and execution methods')
    p('The model revision is 12fd25f77366fa6b3b4b768ec3050bf629380bac; the UltraFeedback Binarized revision is 3949bf5f8c17c394422ccfab0c31ea9c20bdeb85 [R6,R7]. Training and validation are drawn only from train_prefs; test rows come only from test_prefs. The retained population consists of compatible single-turn conversations with the same user context, nonempty distinct completions, and finite strictly ordered preference scores when those fields are supplied. Labels are not relabelled. This intentionally narrow population is not representative of the full dataset [R14].')
    p('Prompt groups use Unicode NFKC normalization, collapsed whitespace and casefolding. All valid official-test prompt groups, including over-length rows, are protected before selecting training data. Accepted rows are sorted by content ID, reduced to one pair per normalized prompt group and shuffled with split seed 314159. The first 64 training-pool rows form validation, the next 256 form training, and 64 eligible official-test rows form test. Both chosen and rejected formatted sequences fit 256 tokens; no completion is silently truncated. A separately written parquet audit reconstructs filtering, IDs, tokens, masks, grouping and selection and checks every retained row.')
    ds=h['data']['statistics']
    p('After the stated exclusions, one duplicate normalized-prompt group was removed; no training row required removal for an official-test prompt conflict. All three selected split memberships and disjoint prompt-group sets were reproduced exactly by the independent audit.')
    table('data-filtering','Table 1. Original filtering, independently reconstructed. Counts are sequential exclusions.',
        ['Official source','Original','Length excluded','Non-strict scores','Empty / identical','Eligible'],
        [[k,str(ds[k+':original']),str(ds[k+':excluded:length cap']),str(ds[k+':excluded:non-strict or reversed scores']),
          f'{ds[k+":excluded:empty text"]} / {ds[k+":excluded:identical completions"]}',str(ds[k+':eligible'])] for k in ['train_prefs','test_prefs']],
        [{'path':'evidence/historical.json','fields':['data.statistics','data.sources','data.split_ids'],
          'formula':'Independent sequential raw-row filters and deterministic normalized-prompt grouping'}])
    p('The historical and follow-up experiments use Windows 11, Python 3.12.10 and an AMD Ryzen 5 PRO 7540U with six physical cores, twelve logical processors and 30.67 GiB physical RAM. Four PyTorch computation threads, FP32 weights/computation/reductions, eager attention, deterministic algorithms and evaluation-mode dropout are fixed. No CUDA device is used. Hardware exclusivity is not established; unrelated processes are not inspected. Pinned versions are torch 2.9.0, transformers 4.57.1, PEFT 0.17.1, TRL 0.24.0, accelerate 1.10.1, datasets 4.3.0 and NumPy 2.2.6. Backend and interop settings are retained per run [R9-R11].')
    p('The model has 135,436,608 parameters, of which 921,600 are trainable LoRA parameters. Rank 16, alpha 32 and zero adapter dropout apply to verified q_proj/v_proj linear targets. Identity-preserving initialization makes the initial adapter policy equivalent to the base. AdamW uses learning rate 5e-5, betas (0.9,0.999), epsilon 1e-8, zero weight decay and clipping norm 1.0. One preference pair (two sequences) is the microbatch; sixteen pairs form an optimizer update. Losses are weighted by the actual accumulation-window size, including a partial final window.')
    p('The single learning seed is 42. The sixteen-update schedule has one warm-up update: updates 1 and 2 both use 5e-5, followed by linear decay to 5e-5/15 at update 16. The 25-update timing schedule also has one optimizer warm-up update, and ends at 5e-5/24. This optimizer schedule is distinct from excluding five benchmark warm-up updates from the measured block. A prior Qwen pilot [R5] did not fit the CPU budget; it was not retried in this pass. Timing repetitions do not add learning seeds.')
    heading('4. Historical audit and equality scope')
    p('The original paper, source, analysis and manifests were archived before edits. The reconstructed source archive matches the historical source hash 88d7f630bae217476e88af8e5362a82f86d3d7f139e52200e9d0c17c6fc79dae and plan 1ba6e9976791feb50acc9766f9ed6f6e731737fb7bdf3369f1809e050e22ada8. The audit verifies complete update/pair counts, the actual integer schedule, split identities and independently recomputes DPO losses and margins from saved per-example scores using Python double arithmetic. Differences from saved FP32 arithmetic stay below 1e-7. This is stronger than checking a manuscript against its own summary.')
    p('One historical cold-cache timing attempt ended after 23 of 25 total updates (18 of 20 measured updates). It has no completed metrics and is excluded from all complete-trial medians. Its exact exit cause and time are unknown; the original interruption observation, partial trace and separately identified retry remain available. The previous budget charges an upper elapsed-time allowance rather than assuming the missing interval was free. No failed attempt is reclassified as a completed observation.')
    p('The two historical learning checkpoint files have identical SHA-256 values and their complete saved fields were compared: model tensors, optimizer state, scheduler, scaler placeholder, update/data position, trace and RNG state. Thus byte identity is supported for those particular files. It is not inferred from a model hash. The old model hash included names and bytes but did not explicitly encode shapes or dtypes. The new canonical digest sorts names, includes shape and dtype and length-delimited contiguous bytes, and includes tied aliases. Shared-base checks additionally cover all named buffers, including nonpersistent buffers. Adapter-only equality is always stated together with a separately identified unchanged base.')
    table('historical-runtime','Table 2. Historical complete-trial medians (range); three timing trials per arm. RSS is sampled during measured training.',
        ['Historical arm','20-update training, s','Benchmark process, s','Training RSS, MiB'],
        [[label]+[f'{v[key]["median"]:.3f} ({v[key]["min"]:.3f}-{v[key]["max"]:.3f})' for key in ['training_seconds','whole_process_seconds','training_rss_mib']]
         for label,v in [('Separate reference',u),('Cold separate-built cache',c),('Warm existing cache',w)]],old_origins)
    p(f'The ratio-of-medians reduction is {reduction(u["training_seconds"]["median"],c["training_seconds"]["median"]):.2f}% for the measured training block, {reduction(u["whole_process_seconds"]["median"],c["whole_process_seconds"]["median"]):.2f}% for the cold benchmark process, and {reduction(lu["whole_process_seconds"],lc["whole_process_seconds"]):.2f}% for the single executed learning workflow. Throughput increase uses a different denominator: the historical training-block value is 39.79%. The benchmark contains 400 processed pairs, including 80 warm-up pairs; the measured 320 pairs contain 40,761 scored completion tokens. The learning workflow processes 256 pairs in sixteen updates with evaluations, generations and a final checkpoint.')
    figure('historical-boundaries','Figure 1. Three historical timing boundaries, deliberately separated. Learning bars are single executions, not three-trial medians. Cold cache includes construction; warm cache excludes an already paid construction.')
    heading('5. Correctness and the pretrained upstream discrepancy')
    p(f'The final working core suite contains {s["checks"]["core"]["passed"]} passing tests, with {s["checks"]["core"]["failures"]} failures, {s["checks"]["core"]["errors"]} errors and {s["checks"]["core"]["skipped"]} skips in its successful run. Added cases check atomic checkpoint failure preservation, canonical identity scope and adapter-context restoration after exceptions. The first runner attempt had four fixture-setup errors from a missing temporary parent directory; fixing the runner made the complete suite pass. This was not a model-correctness failure. Offline collection does not download models. Pretrained integrations remain separate commands.')
    p('The installed TRL comparison executes the real 0.24.0 concatenated_forward and original sigmoid loss functions with reverse KL, beta 0.1, zero label smoothing, no reference synchronization, no weighting, no auxiliary/RPO losses and no alternative length normalization. The same detached reference-score tensor, computed locally from the frozen reference, enters both loss functions to isolate policy-scoring and objective arithmetic. Four fixed training rows are selected by preserved order before scoring, with 4-130 completion tokens. One-pair batches match production padding and positional handling. Every named trainable LoRA gradient and its presence pattern is checked. The second fixture uses two deterministic training-only updates; its adapter tensors and outputs change. A deliberately changed completion token is detected by the negative control.')
    table('correctness','Table 3. Maximum absolute differences. Original pretrained thresholds were exceeded and are retained as failures; diagnostics are a separate scope.',
        ['Comparison','Score error','Loss error','Gradient error','Interpretation'],
        [['FP64 score loss / TRL','n/a','0','0','Original bounds pass'],
         ['Historical tiny GPT-2 / TRL','0','0','2.2352e-8','Tiny function scope'],
         ['Pretrained initial, FP32',f'{parity["identity"]["max_score_error"]:.4e}',f'{parity["identity"]["max_loss_error"]:.4e}',f'{parity["identity"]["max_gradient_error"]:.4e}','Original score/loss bounds fail'],
         ['Pretrained trained, FP32',f'{parity["nontrivial"]["max_score_error"]:.4e}',f'{parity["nontrivial"]["max_loss_error"]:.4e}',f'{parity["nontrivial"]["max_gradient_error"]:.4e}','Original score/loss/gradient bounds fail'],
         ['Initial, FP64 reduction diagnostic',f'{precision["identity"]["max_score_error"]:.5g}',f'{precision["identity"]["max_loss_error"]:.5g}','0','FP32 weights/logits, FP64 reducers'],
         ['Trained, FP64 reduction diagnostic',f'{precision["nontrivial"]["max_score_error"]:.5g}',f'{precision["nontrivial"]["max_loss_error"]:.5g}','0','FP32 weights/logits, FP64 reducers'],
         ['Shared reference / separate','0','0','n/a','Before and after adapter updates']],
        [{'path':'evidence/checks.json','fields':['tiny_upstream','parity.identity','parity.nontrivial','precision.results','parity.shared'],
          'formula':'Elementwise absolute maxima and complete named-gradient comparisons; no selective example removal'}])
    p('The starting absolute bounds were 2e-5 for FP32 summed scores, 2e-6 for FP32 gradients, 1e-12 for float64 score loss and 1e-11 for float64 score gradients; an FP32 loss bound of 2e-6 was also declared. The trained-adapter gradient error exceeds 2e-6 by about 1.03e-8. The score discrepancy is not a tokenization mismatch: observed IDs, attention masks and model logits are bit-identical. A selected-token arithmetic diagnostic compares the local CPU log-softmax and installed TRL selected-logit-minus-logsumexp routines on a tokens-by-vocabulary matrix: they differ by up to 1.775e-5 per token in FP32 and at most 3.09e-14 in FP64. That diagnostic uses a different tensor shape from the full TRL forward. A separate full-forward diagnostic casts returned FP32 logits to FP64 while retaining both actual scoring methods; parameter gradients then match exactly for both fixtures.')
    p('One post-failure numerical adjudication uses a length-aware score bound N*2e-5 plus the standard FP32 summation term gamma_(N-1) times the absolute score sum, and a 1-Lipschitz softplus loss bound. A single empirical FP32 gradient allowance of 4e-6 is recorded, with its revised status explicit. This does not turn the original threshold failures into passes or establish a general error guarantee. Production scoring code and examples were not changed to make agreement look better. The result supports scoped numerical consistency of the implemented objective; complete TRL trainer equivalence is untested.')
    heading('6. Shared-base reference and new trial design')
    p('The shared reference runs the fixed pretrained base with the current LoRA adapter disabled through the inspected PEFT 0.17.1 implementation. The cited API documentation is version 0.17.0, a distinction recorded rather than hidden [R15]. The wrapper rejects merged adapters and trainable base parameters, runs reference inference without gradients, and restores adapter enablement, active adapters, per-module train/eval flags, requires_grad flags and existing gradient objects on normal and exceptional exits. Policy training retains its gradient path. This equivalence depends on an initially identity adapter and a base reference; disabling a policy adapter would not automatically reproduce a different trained reference adapter.')
    p('All named base parameters and all buffers remain unchanged after actual pretrained updates; optimizer parameter identities are exactly the intended adapter identities. Disabled-adapter scores match a separately resident frozen reference before and after nontrivial updates. Three-update matched trajectories (six pairs per strategy) produce identical loss traces and adapter tensors for separate reference, shared base and cached shared base. A second smoke check exercises the actual instrumented worker in fresh processes. The new optimizer loop computes reference scores before creating the policy backward graph; the original-arm accounting replay preserves the historical ordering and cache construction.')
    if complete:
        plan=s['plan']
        p(f'The follow-up plan was frozen locally after inspecting historical results, the numerical diagnosis and short feasibility checks. It is a new protocol, not an untouched confirmatory study. Three fresh-process trials per arm use {plan["warmup_updates"]} excluded warm-up updates plus {plan["measured_updates"]} measured updates, with sixteen pairs per update. The 256 training pairs are shuffled once with seed 42 and cycled to 400 processed pairs; the measured block covers positions 81-400, so processed-pair counts include repeated examples. Order is ABC, BCA, CAB. Each arm restores the same saved initial adapter/empty optimizer state and RNG state against an independently hashed base. Model state, data order, precision, schedule, progress logging and CPU thread settings are matched. Reference preparation cannot advance the restored training RNG.')
    else:p('The frozen follow-up plan and completed runs are retained. This intermediate rendering has not yet received all planned timing observations; missing arms are not imputed.')
    p('A keeps a policy and separate frozen reference resident. B keeps one base with a trainable adapter and recomputes reference scores with that adapter disabled. C constructs a new train-only score cache through B inside every trial process, persists and verifies it, and performs zero reference forwards during training. A versus B measures residency/implementation strategy; B versus C isolates caching against the stronger baseline; A versus C combines both changes. Historical cold C used a separately constructed reference and is a different cohort. No old and new timing trials are pooled.')
    p('After timed training, every new timing arm saves the same small adapter endpoint for named-tensor verification. This I/O and final identity checks are included in whole-process time and excluded from the measured training block. These endpoint files are not full learning-workflow checkpoints, and no held-out evaluation or generation is performed in the A/B/C timing processes.')
    heading('7. New timing and memory results')
    if complete:
        table('followup-runtime','Table 4. New A/B/C median (range), three trials each; 320 measured pairs after 80 warm-up pairs. Whole processes include all cold setup.',
            ['Arm','Training, s','Whole process, s','Training RSS, MiB'],
            [[ARM_LABELS[arm]]+[f'{new[arm][key]["median"]:.3f} ({new[arm][key]["min"]:.3f}-{new[arm][key]["max"]:.3f})'
             for key in ['training_seconds','whole_process_seconds','training_rss_mib']] for arm in ARM_LABELS],new_origins)
        for a,b in [('separate_reference','shared_base_reference'),('shared_base_reference','cached_shared_base'),('separate_reference','cached_shared_base')]:
            p(f'{ARM_LABELS[a]} to {ARM_LABELS[b]} has ratio-of-medians runtime reductions of {reduction(new[a]["training_seconds"]["median"],new[b]["training_seconds"]["median"]):.2f}% for measured training and {reduction(new[a]["whole_process_seconds"]["median"],new[b]["whole_process_seconds"]["median"]):.2f}% for the whole cold benchmark process.')
        p('Negative reduction means the comparison took longer. These are descriptive measurements on this machine, without a statistical-significance claim.')
        figure('followup-runtime','Figure 2. New per-trial training and whole-process durations. Every completed trial is shown. Different arm order balances one simple order effect; three trials cannot characterize all system variability.')
        p(f'Training sampled RSS medians are {new["separate_reference"]["training_rss_mib"]["median"]:.1f}, {new["shared_base_reference"]["training_rss_mib"]["median"]:.1f} and {new["cached_shared_base"]["training_rss_mib"]["median"]:.1f} MiB for A, B and C. Whole-process OS peak working-set medians, including cold construction, are {new["separate_reference"]["whole_os_peak_mib"]["median"]:.1f}, {new["shared_base_reference"]["whole_os_peak_mib"]["median"]:.1f} and {new["cached_shared_base"]["whole_os_peak_mib"]["median"]:.1f} MiB. C pays a measured train-cache precompute median of {new["cached_shared_base"]["precompute_seconds"]["median"]:.3f} s. Whole-process peak memory is not described as cache-only memory, and comparisons never substitute OS peaks for training RSS.')
        ab=new['separate_reference']['training_rss_mib']['median']-new['shared_base_reference']['training_rss_mib']['median']
        bc=new['shared_base_reference']['training_rss_mib']['median']-new['cached_shared_base']['training_rss_mib']['median']
        p(f'The stronger baseline changes the interpretation of memory savings. The difference between A and C medians decomposes into {ab:.1f} MiB from A minus B and {bc:.1f} MiB from B minus C. A-to-C alone therefore does not identify a cache-specific memory benefit. B-to-C is the relevant caching comparison when an identity-initialized adapter shares a frozen base. A-to-B also includes the disabled-adapter implementation overhead; these descriptive contrasts do not separate every low-level mechanism. Cold-process results, rather than the favorable training block alone, determine the observed benefit at this fixed work count.')
        p(f'All nine benchmark endpoints have matching canonical full model-state hashes: {s["followup"]["all_benchmark_final_model_hashes_equal"]}. Complete update traces match: {s["followup"]["all_benchmark_traces_equal"]}. These timing endpoints are not additional held-out learning checkpoints. Memory is sampled RSS at 10 ms intervals, including resident models; it can miss brief peaks. OS whole-process peak working set is a separate metric retained in raw records. GPU memory is unavailable, not recorded as zero. The cold-cache construction stage remains in process time and whole-process memory even if its measured training block is cheaper.')
    else:p('New timing results are pending completion; no projected value is substituted for an observation.')
    heading('8. Complete-workflow accounting and amortization')
    p(f'The historical twenty-update medians project to U={h["projection"]["U"]:.3f} s and C={h["projection"]["C"]:.3f} s per 256-pair training pass, with train-cache precomputation P={h["projection"]["P"]:.3f} s. Under equal other setup, one pass makes caching {h["projection"]["equal_setup_one_pass_cached_minus_uncached_seconds"]:.3f} s slower and breaks even after {h["projection"]["equal_setup_break_even_passes"]:.3f} passes. In contrast, the actual historical learning workflows differ by {lu["whole_process_seconds"]-lc["whole_process_seconds"]:.3f} s in favor of caching. The projection excludes held-out evaluations, generations and checkpoint writing and assumes stationary per-pair timing. It therefore does not predict the complete executed workflow.')
    delta=s['historical']['workflow_cached_minus_uncached']
    p(f'The recorded historical aggregate stages do reconcile the observed workflow difference: cached setup costs {delta["setup_including_cache_seconds"]:.3f} s more, training costs {-delta["training_seconds"]:.3f} s less, combined evaluation/generation costs {-delta["evaluation_and_generation_seconds"]:.3f} s less, and checkpoint writing costs {-delta["checkpoint_seconds"]:.3f} s less. The residual cached-minus-uncached difference is {delta["unallocated_parent_residual_seconds"]:.3f} s. These signed components sum to the observed {-delta["whole_process_seconds"]:.3f} s saving. They are recorded aggregates, not invented per-split or per-phase attribution. In particular, historical learning precomputed 384 pairs across three splits, whereas the benchmark projection uses a 256-pair train cache.')
    p('Old records lump evaluation and generation together and do not split cache construction by dataset split. This leaves finer historical per-phase attribution unresolved despite the aggregate reconciliation. The new replay uses the original sixteen-update protocol and original separate-reference cache design, including train/validation/test precomputation for cached learning. It measures nonoverlapping monotonic stages and a separate parent Popen-to-exit duration. Residual time is the measured difference between the parent duration and allocated stages; it includes imports, launch, output and interpreter exit without pretending that those costs were individually observed. Equivalent instrumentation is applied to both arms.')
    replays=s['followup']['replay']
    if len(replays)==2:
        by={r['metrics']['arm']:r for r in replays}
        ru,rc=by['separate_reference'],by['historical_cached_reference']
        keys=sorted(set(ru['metrics']['stages_seconds'])|set(rc['metrics']['stages_seconds']))
        rows=[[k.replace('_',' '),fmt(ru['metrics']['stages_seconds'].get(k,0)),fmt(rc['metrics']['stages_seconds'].get(k,0))] for k in keys]
        rows += [['Unallocated parent residual',fmt(ru['parent']['unallocated_residual_seconds']),fmt(rc['parent']['unallocated_residual_seconds'])],
                 ['Parent whole process',fmt(ru['parent']['whole_process_seconds']),fmt(rc['parent']['whole_process_seconds'])]]
        table('workflow-accounting','Table 5. Newly measured original-design replay, seconds. Exclusive stages sum with residual to the parent whole process; no training warm-up block is excluded.',
            ['Stage','Separate reference','Separate-built cache'],rows,
            [{'path':r['index']['path']+'/metrics.json','sha256':r['index']['metrics_sha256'],
              'parent_path':r['index']['path']+'/parent-timing.json','parent_sha256':r['index']['parent_timing_sha256'],
              'fields':['stages_seconds','parent.whole_process_seconds','parent.unallocated_residual_seconds'],
              'formula':'Residual=parent elapsed minus sum of mutually exclusive child stages'} for r in replays])
        p(f'In this new replay, whole-process times are {ru["parent"]["whole_process_seconds"]:.3f} s versus {rc["parent"]["whole_process_seconds"]:.3f} s, a signed reduction of {reduction(ru["parent"]["whole_process_seconds"],rc["parent"]["whole_process_seconds"]):.2f}%. Training itself takes {ru["metrics"]["training_seconds"]:.3f} s versus {rc["metrics"]["training_seconds"]:.3f} s. The breakdown explains these new observations, not the unrecorded split of older costs. Absolute runtime can move with concurrent system activity, thermal state and ordinary variation; exclusivity was not established.')
        figure('workflow-stages','Figure 3. New original-design learning replay. Cache construction, evaluation/generation and training are separate; the parent residual is shown explicitly. These are single paired workflow executions.')
    equation(r'T_u(E)=S_u+E U,\qquad T_c(E)=S_c+P+E C')
    equation(r'E>\frac{S_c-S_u+P}{U-C}\quad\mathrm{when}\quad U>C')
    p('Here S_c explicitly excludes reference precomputation P. The model assumes an unchanged reference and tokenized population, reusable cache, stationary per-pair cost and no evaluation/generation or full learning checkpoint. If U is no greater than C, extra passes do not yield a finite amortization point under these assumptions. Setup, cold construction and cache reuse are never charged twice. Projections describe these assumptions; they are not executed additional epochs.')
    if complete:
        pr=s['followup']['projection']
        p(f'Using the new B/C medians gives U={pr["U"]:.3f}, C={pr["C"]:.3f}, P={pr["P"]:.3f}, S_u={pr["S_u"]:.3f} and S_c={pr["S_c_excluding_P"]:.3f} seconds. The projected break-even is {fmt(pr["break_even_passes"])} passes, and projected cached-minus-shared time for one pass is {pr["one_pass_cached_minus_shared_seconds"]:.3f} s. The setup proxy includes measured nontraining identity/persistence checks and the unallocated launch/exit residual, excludes benchmark warm-up and P, and should not be confused with the learning-workflow boundary.')
    heading('9. Single-seed held-out outcomes')
    final=lu['final_evaluation'];initial=lu['initial_evaluation']
    table('learning-outcomes','Table 6. Historical single paired seed, 16 updates; both reference strategies give identical saved per-example outcomes.',
        ['Split','Initial loss','Final loss','Positive margins','Ties'],
        [[split,f'{initial[split]["dpo_loss"]:.6f}',f'{final[split]["dpo_loss"]:.6f}',
          f'{round(final[split]["strict_positive_accuracy"]*64)}/64 ({100*final[split]["strict_positive_accuracy"]:.2f}%)',
          f'{round(final[split]["tie_rate"]*64)}/64'] for split in ['validation','test']],
        [{'path':r['artifact'],'sha256':r['artifact_sha256'],'fields':['initial_evaluation.*.examples','final_evaluation.*.examples'],
          'formula':'Mean per-pair softplus(-0.1*m); count beta*m>1e-6 and abs(beta*m)<=1e-6'} for r in h['runs'] if r['kind']=='learning'])
    p('Strict positives use beta*m > 1e-6; ties use |beta*m| <= 1e-6. Initial policy/reference equality gives 64 ties in each held-out split, not evidence of zero language ability. After training, test positives are 39/64 and validation positives 42/64, with no ties. Initial/final losses are pair-weighted, not token-weighted. Double recomputation from stored scores supports the displayed rounded values. The paired learning trajectories and final outcomes agree for the single paired seed; this is not an average across seeds.')
    p('The eight reserved exploratory generations use greedy argmax and at most 24 new tokens, ending earlier at the assistant end token or remaining context cap. They are short examples without completed human ratings. No paid judge, preference survey or broad helpfulness/truthfulness study was run. The modest DPO-loss change and margin counts do not demonstrate reduced hallucination, improved general usefulness or a human-preference gain. Test outputs were examined during the historical study; this follow-up does not treat them as an untouched confirmatory evaluation.')
    heading('10. Limits, reproducibility and responsibility')
    p('This is one small pretrained model, one filtered short-response population, one learning seed and one nonexclusive CPU host. Three timing trials describe this workload and cannot establish statistical significance, between-seed uncertainty or GPU behavior. AI-feedback preference labels can contain noise and systematic bias, and pretraining overlap is unknown. Stronger language-quality claims would require different evidence. Cached scores are valid only while the fixed reference, tokenizer/template, IDs/masks, precision/backend and other declared identity fields remain compatible. Numerical equivalence on this hardware is not a promise of bitwise portability across versions or devices.')
    clean=s['checks'].get('clean',{})
    if clean.get('status')=='passed':
        p(f'A genuinely fresh project-local environment and scoped source copy were executed. The clean core suite passed {s["checks"]["clean_core"]["passed"]} tests with no failures/errors/skips. Installed dependency consistency and file RECORD hashes were checked, user/system site packages were excluded, and imports resolved inside the clean environment. Cached pinned package payloads and hash-verified public model/data inputs were reused offline. Pinned tiny and pretrained TRL checks, the three-strategy short trajectory, toy resume coverage, numeric raw-evidence verification and paper/table regeneration ran there. The complete timing and learning campaigns were not repeated in that environment.')
    else:p('Clean-environment verification is separately recorded in the finalization status. This rendering does not treat an unexecuted or pending clean check as a pass.')
    p('The implemented entry point is python scripts/finalize.py --offline --max-experiment-hours 6 --resume. --report-only regenerates manuscript and tables from evidence without training or downloads; --verify-only recalculates numeric checks without launching experiments. Existing raw evidence, immutable plans, result hashes and worker-source identities gate reuse. A fresh acquisition route remains python scripts/bootstrap.py --profile compact; use --offline only when all locked packages and public inputs are already available. Lightweight local tests are python -m pytest -q after installing the pinned environment. Exact Windows commands and the clean scope are recorded in the finalization report.')
    p(f'The available historical measured process total is {resources["historical_measured_process_seconds"]:.3f} s ({resources["historical_measured_process_seconds"]/3600:.3f} h); with documented allowances it is {resources["historical_accounted_seconds"]:.3f} s. This additional pass currently records {resources["new_measured_process_seconds"]:.3f} s ({resources["new_measured_process_seconds"]/3600:.3f} h) of leaf subprocess time, including failed attempts and reproduction checks. Parent waits are not added again. Actual elapsed wall time, storage and the finalized ledger are reported separately. The additional limit is six experiment-process hours, ninety minutes per child, five GiB of downloads and twelve GiB additional storage. No new model/data payloads were downloaded.')
    p('The review archive contains source, tests, locked requirements, documentation, the synchronized report and numeric evidence sufficient to regenerate tables and audit scores. It excludes model weights, giant checkpoints, environments, caches, full datasets, credentials and unrelated material. Bundle verification checks the included evidence; it cannot reopen excluded checkpoint tensors or retrain offline without separately obtained pinned inputs. Local checkpoint audits and their exact scope are retained as evidence. Upstream licenses/notices are referenced without inventing a license grant for this project or third-party data.')
    p('The supported conclusion is that fixed-reference strategies can preserve the observed local trajectory while changing measured costs, and that a shared-base baseline and complete accounting are necessary for interpreting those costs. The observed FP32 upstream discrepancy remains explicitly qualified.')
    heading('Appendix A. Per-trial values and audit trail')
    table('historical-paired-reductions','Table A1. Historical paired-trial runtime reductions; these are not the ratio of medians.',
        ['Trial','Training reduction, %','Cold process reduction, %'],
        [[r['trial'],fmt(r['training_percent']),fmt(r['whole_percent'])] for r in s['historical']['paired_reductions']],old_origins)
    if complete:
        table('followup-per-trial','Table A2. New measured per-trial values. All cold processes include setup; training excludes five warm-up updates.',
            ['Trial','Arm','Training, s','Whole process, s'],
            [[r['metrics']['trial'],ARM_LABELS[r['metrics']['arm']],fmt(r['metrics']['training_seconds']),fmt(r['parent']['whole_process_seconds'])]
             for r in follow if r['metrics']['kind']=='benchmark'],new_origins)
        table('followup-paired-reductions','Table A3. New paired-trial reductions, 100*(baseline-comparison)/baseline. Negative values indicate slower execution.',
            ['Trial','Comparison','Training, %','Whole process, %'],
            [[r['trial'],r['baseline'].split('_')[0]+' to '+r['comparison'].split('_')[0],fmt(r['training_percent']),fmt(r['whole_percent'])]
             for r in s['followup']['paired_reductions']],new_origins)
    p('Local evidence roots are artifacts/finalization/<pass-id>/audit, parity, diagnose, precision, runs and evidence. inventory.json records the preserved source and historical files; frozen-source.zip and plan.json identify the follow-up. Each run records a spec, provenance, progress, metrics and parent-timing file. claims.json maps report tables to source hashes, fields, selection rules and formulas. The portable evidence manifest hashes included numeric files. Earlier failed/interrupted attempts remain local and visible in the ledger; no completion is inferred from a directory name alone.')
    heading('References')
    used=sources
    for source in used:
        blocks.append({'kind':'reference','id':source['id'],'title':source['title'],
            'author':'; '.join(source['authors']) if source.get('authors') else source.get('author_organization',source.get('author','')),
            'year':source.get('year',''),'url':source['url'],
            'retrieved':source['retrieved_at'][:10],'revision':source.get('revision',source.get('version_note',''))})
    structured={'title':blocks[0]['text'],'blocks':blocks,'summary_sha256':file_hash(folder/'summary.json'),
                'generated_at':now(),'source':'Shared structured blocks for Markdown, TeX, PDF; figures and CSV tables from same summary'}
    atomic_json(paper/'report_blocks.json',structured)
    for name,source,fields,formula,status in [
        ('historical source and completion audit','historical.json',['original_source_sha256','original_plan_sha256','runs','incomplete_attempts'],
         'Archived source digest, completed update/pair counts and interrupted-attempt exclusion','confirmed'),
        ('dataset membership and exclusions','historical.json',['data.statistics','data.sources','data.split_ids'],
         'Independent parquet reconstruction, normalized prompt sets, fixed seed and all selected IDs/tokens/masks','confirmed'),
        ('historical checkpoint byte and tensor equality','historical.json',['checkpoint_equality'],
         'Both file SHA-256 values and recursively compared model/optimizer/scheduler/RNG/trace fields; canonical tensor digest includes shape/dtype','confirmed'),
        ('historical upstream scope','checks.json',['tiny_upstream.scope','parity','diagnosis','precision','adjudication'],
         'Separate tiny-model scope from pretrained initial/nontrivial fixtures; original thresholds fail; diagnostic and revised bounds explicit','corrected'),
        ('shared reference invariants','checks.json',['parity.shared'],
         'Disabled-adapter score comparison before/after training; all base parameters/buffers; restored flags/modes/gradients; three matched trajectories','confirmed'),
        ('test counts and clean reproduction','checks.json',['core','clean_core','clean','clean_isolation','clean_pretrained'],
         'JUnit totals minus failures/errors/skips, isolated module/dependency checks and exact clean scope','confirmed' if clean.get('status')=='passed' else 'pending'),
        ('old workflow accounting and attribution','historical.json',['projection','historical_attribution','runs.*.setup_including_cache_seconds','runs.*.training_seconds','runs.*.evaluation_and_generation_seconds','runs.*.checkpoint_seconds','runs.*.whole_process_seconds'],
         'Recorded aggregate components plus parent-minus-components residual reconcile total difference; old per-split precompute and separate eval/generation timing not recorded','aggregate accounting confirmed; fine-grained historical attribution unverifiable'),
        ('resource totals','resources.json',['ledger','historical_measured_process_seconds','historical_accounted_seconds','new_measured_process_seconds'],
         'Sum leaf subprocess elapsed including failed attempts; parent orchestration not added; historical allowances separate','confirmed at snapshot'),
        ('hardware and pinned software','environment.json',['cpu','physical_cores','logical_cpus','ram_total_bytes','versions'],
         'Recorded machine/runtime inspection; exclusivity not established','confirmed'),
        ('parameter counts and learning schedule','historical.json',['runs.*.parameters','runs.*.trace','runs.*.updates'],
         'Named parameter numel totals and actual integer learning-rate trace; one learning seed','confirmed')]:
        claims.append({'id':name,'status':status,'source':{'path':'evidence/'+source,'sha256':file_hash(folder/'evidence'/source),'fields':fields},
                       'formula':formula,'scope':'As stated; no stronger inference implied'})
    atomic_json(folder/'claims.json',{'claims':claims,'evidence_manifest_sha256':file_hash(folder/'evidence/manifest.json'),
        'scope':'All table cells have source artifacts/hashes, fields, selection rules and formulas; central narrative claims separately mapped'})
    plots(paper,s,h,follow)
    render(paper,blocks,used)
    documents(folder,s,h)


def plots(paper,s,h,follow):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150,'savefig.dpi':220})
    colors=['#245A81','#D0782C','#3E846B']
    fig,axs=plt.subplots(1,3,figsize=(8,2.8))
    old=h['timing'];arms=['uncached_reference','cached_reference','warm_cached_reference']
    for ax,key,title in zip(axs[:2],['training_seconds','whole_process_seconds'],['20 measured updates','Whole benchmark process']):
        vals=[old[a][key]['median'] for a in arms]
        ax.bar(['U','Cold C','Warm C'],vals,color=colors)
        ax.set_title(title);ax.set_ylabel('Seconds');ax.set_ylim(0,max(vals)*1.18)
        for i,v in enumerate(vals):ax.text(i,v+max(vals)*.025,f'{v:.0f}',ha='center',fontsize=8)
    learning={r['arm']:r for r in s['historical']['learning']}
    vals=[learning[a]['whole_process_seconds'] for a in ['uncached_reference','cached_reference']]
    axs[2].bar(['U','Cold C'],vals,color=colors[:2]);axs[2].set_title('16-update learning workflow');axs[2].set_ylabel('Seconds');axs[2].set_ylim(0,max(vals)*1.18)
    for i,v in enumerate(vals):axs[2].text(i,v+max(vals)*.025,f'{v:.0f}',ha='center',fontsize=8)
    fig.tight_layout();fig.savefig(paper/'figures/historical-boundaries.png');plt.close(fig)
    timing=[r for r in follow if r['metrics']['kind']=='benchmark']
    if s['cohort_complete']:
        fig,axs=plt.subplots(1,2,figsize=(8,3))
        for i,(arm,label) in enumerate(ARM_LABELS.items()):
            rows=sorted([r for r in timing if r['metrics']['arm']==arm],key=lambda r:r['metrics']['trial'])
            for ax,key in zip(axs,['training_seconds','whole_process_seconds']):
                vals=[r['metrics']['training_seconds'] if key=='training_seconds' else r['parent'][key] for r in rows]
                ax.plot([0,1,2],vals,marker=['o','s','^'][i],color=colors[i],label=label)
                ax.set_xticks([0,1,2]);ax.set_xlabel('Trial ID');ax.set_ylabel('Seconds')
        axs[0].set_title('Measured training block');axs[1].set_title('Whole cold process')
        handles,labels=axs[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',ncol=3,bbox_to_anchor=(.5,-.015),frameon=False,fontsize=9.5)
        fig.tight_layout(rect=[0,.1,1,1]);fig.savefig(paper/'figures/followup-runtime.png',bbox_inches='tight');plt.close(fig)
    replay=s['followup']['replay']
    if len(replay)==2:
        fig,ax=plt.subplots(figsize=(8,3.0));groups=['Load / identity','Precompute','Cache I/O','Initial eval / gen','Training','Final eval / gen','Checkpoint / checks','Residual']
        palette=['#7DA6C2','#D99948','#D0B988','#96C1B1','#245A81','#508E72','#8D80A6','#BBBBBB']
        for i,r in enumerate(replay):
            stages=r['metrics']['stages_seconds'];vals=[0.]*8
            for key,v in stages.items():
                if key.startswith('reference_precompute'):idx=1
                elif key=='cache_persist_verify_read':idx=2
                elif key.startswith('initial_evaluation') or key=='initial_generation':idx=3
                elif key=='measured_training':idx=4
                elif key.startswith('final_evaluation') or key=='final_generation':idx=5
                elif key in ['checkpoint_write','final_identity_validation','shutdown_release']:idx=6
                else:idx=0
                vals[idx]+=v
            vals[7]=r['parent']['unallocated_residual_seconds'];left=0
            for j,v in enumerate(vals):ax.barh(i,v,left=left,color=palette[j],label=groups[j] if i==0 else None);left+=v
        ax.set_yticks([0,1],['Separate reference','Separate-built cache']);ax.set_xlabel('Parent-observed seconds');ax.invert_yaxis()
        ax.legend(ncol=4,loc='upper center',bbox_to_anchor=(.5,-.22),fontsize=9.5,frameon=False)
        fig.tight_layout();fig.savefig(paper/'figures/workflow-stages.png',bbox_inches='tight');plt.close(fig)


def render(paper,blocks,sources):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,KeepTogether
    from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
    from reportlab.lib import colors
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.enums import TA_CENTER
    fonts=Path(matplotlib.get_data_path())/'fonts/ttf'
    for name,filename in [('Body','DejaVuSerif.ttf'),('BodyBold','DejaVuSerif-Bold.ttf'),('Sans','DejaVuSans.ttf'),('SansBold','DejaVuSans-Bold.ttf')]:
        pdfmetrics.registerFont(TTFont(name,str(fonts/filename)))
    styles=getSampleStyleSheet()
    body=ParagraphStyle('BodyStudy',fontName='Body',fontSize=9.6,leading=13.3,spaceAfter=7,allowWidows=0,allowOrphans=0)
    title=ParagraphStyle('TitleStudy',fontName='SansBold',fontSize=17,leading=22,spaceAfter=12)
    heading=ParagraphStyle('HeadStudy',fontName='SansBold',fontSize=11.5,leading=15,spaceBefore=11,spaceAfter=6,keepWithNext=True)
    caption=ParagraphStyle('CaptionStudy',fontName='Sans',fontSize=8.1,leading=10.5,spaceAfter=8)
    table_caption=ParagraphStyle('TableCaptionStudy',parent=caption,keepWithNext=True)
    small=ParagraphStyle('SmallStudy',fontName='Sans',fontSize=7.5,leading=10,spaceAfter=5)
    th=ParagraphStyle('THStudy',parent=small,fontName='SansBold',textColor=colors.white)
    story=[];md=[];tex=[];equation_count=0;equation_group=[]
    def escape(text):
        mapping={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}
        return ''.join(mapping.get(c,c) for c in str(text))
    for b in blocks:
        k=b['kind']
        if k!='equation' and equation_group:
            story.append(KeepTogether(equation_group));equation_group=[]
        if k in ['title','heading','paragraph']:
            style={'title':title,'heading':heading,'paragraph':body}[k]
            story.append(Paragraph(html.escape(b['text']),style))
            md.append(('# ' if k=='title' else '## ' if k=='heading' else '')+b['text'])
            tex.append((r'\section*{'+escape(b['text'])+'}' if k=='heading' else r'\maketitle' if k=='title' else escape(b['text'])+'\n'))
        elif k=='equation':
            equation_count+=1
            path=paper/'figures'/f'equation-{equation_count}.png'
            fig=plt.figure(figsize=(7, .55));fig.text(.5,.5,'$'+b['latex']+'$',ha='center',va='center',fontsize=13)
            fig.savefig(path,dpi=240,bbox_inches='tight',pad_inches=.08);plt.close(fig)
            im=Image(str(path));scale=min(72/240,470/im.imageWidth);im.drawWidth=im.imageWidth*scale;im.drawHeight=im.imageHeight*scale
            equation_group.extend([im,Spacer(1,7)]);md.append('$$\n'+b['latex']+'\n$$');tex.append('\\[\n'+b['latex']+'\n\\]')
        elif k=='table':
            n=len(b['columns']);width=487
            if n==6:ratios=[1.35,.7,1,1.05,1.05,.8]
            elif n==5:ratios=[2.2,1,1,1,2.0] if b['name']=='correctness' else [1.2,1,1,1.5,.6]
            elif n==4:ratios=[1.35,1,1,1]
            else:ratios=[1.8]+[1]*(n-1)
            widths=[width*x/sum(ratios) for x in ratios]
            cells=[[Paragraph(html.escape(b['title']),caption)]+['']*(n-1)]
            cells+=[[Paragraph(html.escape(str(x)),th) for x in b['columns']]]+[[Paragraph(html.escape(str(x)),small) for x in row] for row in b['rows']]
            t=Table(cells,colWidths=widths,repeatRows=2,hAlign='LEFT')
            t.setStyle(TableStyle([('SPAN',(0,0),(-1,0)),('BACKGROUND',(0,1),(-1,1),colors.HexColor('#245A81')),('VALIGN',(0,0),(-1,-1),'TOP'),
                ('ROWBACKGROUNDS',(0,2),(-1,-1),[colors.HexColor('#F1F4F6'),colors.white]),('BOTTOMPADDING',(0,0),(-1,-1),5),
                ('TOPPADDING',(0,0),(-1,-1),5),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),
                ('LEFTPADDING',(0,0),(-1,0),0),('RIGHTPADDING',(0,0),(-1,0),0),('TOPPADDING',(0,0),(-1,0),0),('BOTTOMPADDING',(0,0),(-1,0),8)]))
            story.extend([t,Spacer(1,9)])
            md.append(b['title']+'\n\n|'+'|'.join(b['columns'])+'|\n|'+'|'.join(['---']*n)+'|\n'+'\n'.join('|'+'|'.join(str(x) for x in row)+'|' for row in b['rows']))
            colspec=''.join('p{'+f'{.90*r/sum(ratios):.3f}'+r'\linewidth}' for r in ratios)
            tex.append(r'\begin{table}[htbp]\centering\footnotesize\caption{'+escape(b['title'])+'}\n'+r'\begin{tabular}{'+colspec+'}\n\\toprule\n'+
                ' & '.join(escape(x) for x in b['columns'])+r' \\ \midrule'+'\n'+'\n'.join(' & '.join(escape(x) for x in row)+r' \\' for row in b['rows'])+'\n'+r'\bottomrule\end{tabular}\end{table}')
        elif k=='figure':
            path=paper/b['path']
            if not path.exists():continue
            im=Image(str(path));scale=487/im.imageWidth;im.drawWidth=487;im.drawHeight=im.imageHeight*scale
            story.append(KeepTogether([im,Paragraph(html.escape(b['caption']),caption)]))
            md.append('!['+b['caption']+']('+b['path']+')')
            tex.append(r'\begin{figure}[htbp]\centering\includegraphics[width=\linewidth]{'+b['path']+r'}\caption{'+escape(b['caption'])+r'}\end{figure}')
        elif k=='reference':
            author=b['author']+(f' ({b["year"]})' if b.get('year') else '')
            text=f'[{b["id"]}] {author}. {b["title"]}. Retrieved {b["retrieved"]}.'+(f' Version/revision: {b["revision"]}.' if b['revision'] else '')
            story.append(Paragraph(html.escape(text)+f' <link href="{html.escape(b["url"],quote=True)}" color="#245A81">Primary source</link>.',small))
            md.append(f'[{b["id"]}] {author}. [{b["title"]}]({b["url"]}). Retrieved {b["retrieved"]}. '+b['revision'])
            tex.append(r'\noindent '+escape(text)+r' \href{'+b['url']+r'}{Primary source}.\par')
    if equation_group:story.append(KeepTogether(equation_group))
    (paper/'paper.md').write_text('\n\n'.join(md)+'\n',encoding='utf8')
    preamble=r'''\documentclass[10pt]{article}
\usepackage[a4paper,margin=19mm]{geometry}
\usepackage{amsmath,amssymb,graphicx,booktabs,hyperref,array}
\hypersetup{colorlinks=true,urlcolor=blue}
\setlength{\parindent}{0pt}\setlength{\parskip}{6pt}
\setlength{\tabcolsep}{3pt}
\title{'''+escape(blocks[0]['text'])+r'''}
\author{Aryan Golbaghi}
\date{16 October 2026}
\begin{document}
'''
    (paper/'main.tex').write_text(preamble+'\n\n'.join(tex)+'\n\\end{document}\n',encoding='utf8')
    bibliography=[]
    for ref in sources:
        author=' and '.join(ref['authors']) if ref.get('authors') else '{'+ref.get('author_organization',ref.get('author',''))+'}'
        year='\n  year={'+ref['year']+'},' if ref.get('year') else ''
        bibliography.append('@misc{'+ref['id']+',\n  title={'+ref['title']+'},\n  author={'+author+'},'+year+'\n  howpublished={\\url{'+ref['url']+'}},\n  note={Retrieved '+ref['retrieved_at'][:10]+'; version/provenance in source registry}\n}')
    (paper/'references.bib').write_text('\n\n'.join(bibliography)+'\n',encoding='utf8')
    def page(canvas,doc):
        canvas.saveState();canvas.setFont('Sans',7.5);canvas.setFillColor(colors.HexColor('#58616A'))
        canvas.drawString(54,818,'DPO-Mini | Correctness and reference-cost study')
        canvas.drawRightString(541,25,str(doc.page));canvas.restoreState()
    doc=SimpleDocTemplate(str(paper/'paper.pdf'),pagesize=(595.28,841.89),leftMargin=54,rightMargin=54,topMargin=42,bottomMargin=40,
        title=blocks[0]['text'],author='Aryan Golbaghi')
    doc.build(story,onFirstPage=page,onLaterPages=page)
    import fitz
    pdf=fitz.open(paper/'paper.pdf');preview=paper/'preview';preview.mkdir(exist_ok=True)
    for i,page in enumerate(pdf):page.get_pixmap(matrix=fitz.Matrix(1.4,1.4)).save(preview/f'page-{i+1:02d}.png')
    atomic_json(paper/'render-record.json',{'renderer':'ReportLab; equations/plots Matplotlib; page PNGs PyMuPDF',
        'pages':len(pdf),'pdf_sha256':file_hash(paper/'paper.pdf'),'visual_inspection':'pending explicit image review',
        'tex_status':'Synchronized editable source emitted; not claimed compiled by LaTeX'})


def documents(folder,s,h):
    from .documentation import write
    write(folder,s,h)
