"""One structured report -> Markdown, LaTeX/BibTeX, and local ReportLab PDF."""
import _common
import html
import re
from pathlib import Path
from dpo_mini.provenance import ROOT,atomic_json,read_json,now


def build_blocks(s):
    cfg=s["plan"]["config"];toy=cfg["profile"]=="toy"
    timing=s["timing"];learn=s["learning"];up=s["upstream"]
    blocks=[]
    def add(kind,text=None,**kw):blocks.append({"kind":kind,"text":text,**kw})
    def section(title):add("page");add("heading",title)
    title="DPO-Mini: A Reproducible Correctness and Reference-Caching Study of Direct Preference Optimization"
    if toy:title+=" (Toy Validation Only)"
    add("title",title)
    add("paragraph","Technical report. Aryan Golbaghi.")
    add("paragraph",f'Executed scope: {s["completion_gate"]}. Report generated {now()[:10]}.')
    add("heading","Abstract")
    if timing and learn:
        u=timing["uncached_reference"];c=timing["cached_reference"]
        mean_loss=sum(r["final_loss_uncached"] for r in learn)/len(learn)
        delta=s["percentages"]["training_seconds"]["percent_reduction"]
        cold=s["percentages"]["whole_process_seconds"]["percent_reduction"]
        abstract=(f'We independently implement the original sigmoid DPO objective and compare recomputed and cached fixed-reference scores using {cfg["model_id"]} on a CPU. '
          f'The study uses {cfg["train_pairs"]} training, {cfg["validation_pairs"]} validation and {cfg["test_pairs"]} test pairs, {len(learn)} paired training seeds, one epoch, and three paired timing trials. '
          f'Median measured training time is {u["training_seconds"]["median"]:.3f} s uncached and {c["training_seconds"]["median"]:.3f} s cached, a {delta:.2f}% runtime reduction. '
          f'The cold whole-process runtime reduction is {cold:.2f}%; cache construction is included. '
          f'The final test DPO loss averages {mean_loss:.6f} across seeds. '
          f'Installed TRL score-level loss/gradient errors are {up.get("score_loss_error",{}).get("max_absolute",float("nan")):.1e}/{up.get("score_gradient_error",{}).get("max_absolute",float("nan")):.1e}; '
          f'the tiny-model gradient comparison has maximum absolute error {up.get("model_parameter_gradient_error",{}).get("max_absolute",float("nan")):.3e}. '
          'These measurements establish bounded implementation and systems evidence; preference margins do not establish general response quality. Evidence: claims C1-C6 and linked raw runs.')
    else:
        abstract='The implementation and reporting pipeline is available, but a complete matched learning study was not finished. No pretrained performance or quality conclusion is supported. The status archive identifies the incomplete stages.'
    add("paragraph",abstract)
    add("heading","1. Introduction and research questions")
    add("paragraph","DPO expresses preference learning through policy/reference log-probability differences [R1]. A small independent implementation is useful only if token boundaries, gradient behavior and measurement scope are auditable. Reference-score precomputation already exists in TRL [R3,R4]; this report claims neither a new alignment objective nor a new caching technique.")
    add("paragraph","We ask whether independent scores, losses and gradients match analytic calculations and upstream functions; whether fixed-reference caching preserves the matched optimization trajectory; and whether preprocessing-inclusive runtime or memory improves on the available hardware. A negative speed result is a valid answer.")
    section("2. Objective, implementation and correctness")
    add("paragraph","The score is the sum of log probabilities of completion tokens, including the assistant end token. Logits at position t-1 predict token t. Thus the first answer token is scored from the final prompt position. Prompt positions are excluded from direct score reduction but remain visible to attention and can receive contextual gradients.")
    add("equation","s(theta; x,y) = sum_t log p_theta(y_t | x,y_<t)")
    add("equation","m = [s_pi(chosen)-s_ref(chosen)] - [s_pi(rejected)-s_ref(rejected)]")
    add("equation","L = mean softplus(-beta*m), beta = 0.1")
    add("paragraph","The scorer selects scored prediction positions before log-softmax/gather, avoiding invalid masked target IDs and zero-times-infinity artifacts. Half-precision inputs would be promoted for reduction; float64 is preserved. The executed workload uses FP32. Dropout is zero, the reference is frozen and excluded from AdamW, and the cached path performs zero reference forwards during training.")
    add("paragraph","Mandatory local tests cover log(2) at equality with nonzero policy-score gradients, the analytic margin-two example, gradient direction/gradcheck, first and final response tokens, unequal lengths and padding/EOS aliasing, causal masking, prompt gradients, frozen reference, cache corruption and ID mismatches, partial accumulation and resume. At beta=0.1, scores (-2,-4) versus (-3,-3) give margin 2 and loss log(1+exp(-0.2)). Equality gives chosen/rejected score derivatives -0.05/+0.05 for one pair.")
    error_rows=[["Comparison","Max absolute error"]]
    for label,key in [("TRL float64 score loss","score_loss_error"),("TRL float64 score gradients","score_gradient_error"),("TRL FP32 completion scores","completion_score_error"),("TRL FP32 model gradients","model_parameter_gradient_error")]:
        error_rows.append([label,f'{up[key]["max_absolute"]:.4e}' if key in up else "not measured"])
    add("table",rows=error_rows)
    add("paragraph","The independent comparator executes installed TRL 0.24.0 dpo_loss and concatenated_forward. The latter uses a locally initialized two-layer GPT-2 (width 32, four heads), identical weights, token IDs, positions and masks. This is function-level model/gradient parity, not full Trainer or pretrained adapter parity. FP64 tolerances are 1e-12 for loss and 1e-11 for gradients; FP32 tolerances are 2e-5 for scores and 2e-6 for gradients. Relative gradient error uses a 1e-12 denominator floor; large ratios near zero are interpreted with the absolute errors.")
    add("paragraph","A real debugging finding was Windows fsync failure on a read-only checkpoint descriptor; the implementation now flushes the writable stream before atomic replacement. An early FP32 accumulated-versus-batched Adam comparison differed by 3.53645e-5 in weights. The accumulation invariant is checked in float64 at 1e-10, while paired experiments preserve the same microbatch ordering. Failed logs remain in the archive.")
    section("3. Experimental protocol and provenance")
    env=s["environment"]
    add("paragraph",f'Hardware: {env["cpu"]}; {env["physical_cores"]} physical cores, {env["logical_cpus"]} logical processors; {env["ram_total_bytes"]/1024**3:.2f} GiB physical RAM. Windows and Python versions are {env["os"]} and {env["python"]}. Measurements use {cfg["threads"]} CPU threads and no CUDA device. Hardware exclusivity was not established. RAM availability was sampled; no private process inventory was collected.')
    add("paragraph","Resolved packages: "+", ".join(f'{k} {v}' for k,v in env["versions"].items())+". Fully pinned transitive versions are in requirements.lock. Setup and public downloads are excluded from compute comparisons and logged separately. Local files include pinned source/model/data revisions and hashes.")
    add("paragraph",f'Profile {cfg["profile"]}; model revision {cfg["revision"]}; dataset revision {cfg["dataset_revision"]}. The plan hash is {s["plan"]["sha256"]}. This is a frozen local plan, not a public preregistration. A paired pilot precedes the freeze; larger failed or infeasible profiles are retained in the pilot archive.')
    if s["runs"]:
        parameters=s["runs"][0]["parameters"]
        add("paragraph",f'The instantiated policy contains {parameters["total"]:,} parameters, of which {parameters["trainable"]:,} are trainable. These are actual runtime counts, including adapters where used.')
    if toy:
        add("paragraph","Synthetic data uses distinct numbered prompts requesting yes, with yes preferred over no or never. A byte tokenizer and locally initialized two-layer width-64/four-head transformer avoid model downloads. There is no evidence here about pretrained language ability, human preferences, or general alignment.")
    else:
        data_stats=read_json(ROOT/"artifacts/prepared"/cfg["profile"]/"manifest.json")["statistics"]
        add("paragraph",f'The pinned model card is [{"R6" if cfg["profile"]=="compact" else "R5"}]. Original preference split counts were {data_stats["train_prefs:original"]:,} train and {data_stats["test_prefs:original"]:,} test. Length filtering excluded {data_stats.get("train_prefs:excluded:length cap",0):,} and {data_stats.get("test_prefs:excluded:length cap",0):,}, respectively; non-strict or reversed supplied scores excluded {data_stats.get("train_prefs:excluded:non-strict or reversed scores",0):,} and {data_stats.get("test_prefs:excluded:non-strict or reversed scores",0):,}. Other exclusions and group deduplication are itemized in the manifest.')
        add("paragraph","UltraFeedback Binarized supplies AI-feedback labels [R7,R14]. Only two-message user/assistant pairs with verified common context, nonempty distinct completions and strict provided score ordering are eligible. Training and validation come from train_prefs; final test examples retain test_prefs membership. Unicode NFKC, whitespace collapse and casefold define prompt groups. All valid official test groups take precedence; one pair per normalized prompt group is selected. This cannot establish absence of pretraining overlap.")
        add("paragraph",f'Both formatted sequences must fit {cfg["max_length"]} tokens; no completion is silently truncated. Filtering counts, original schema, IDs, group hashes and length distributions are in data_manifest.json. The final split has {cfg["train_pairs"]}/{cfg["validation_pairs"]}/{cfg["test_pairs"]} pairs. Length selection restricts the population to which results apply.')
        add("paragraph","The documented generation prompt is tokenized once, then response tokens and one assistant ending are appended by construction [R8]. This avoids assuming whole-chat and generation-prefix tokenization agree. Qwen pilots disable thinking, including any template-generated empty thinking prefix in the conditioning prompt. Compact uses its pinned chat template. Representative decoded sequences and masks are archived before training.")
    add("paragraph",f'Training uses one epoch and seeds {s["plan"]["seeds"]}. A microbatch is {cfg["microbatch_pairs"]} preference pair(s), each containing two sequences; the effective batch is {cfg["effective_batch_pairs"]} pairs. AdamW uses learning rate {cfg["learning_rate"]}, betas (0.9,0.999), epsilon 1e-8, zero decay, norm clipping at 1.0 and approximately 3% linear warm-up followed by decay. Final partial windows divide by their actual pair count. Pretrained policy uses LoRA rank16/alpha32 on verified q_proj/v_proj modules, zero dropout and identity-preserving initialization [R12,R13].')
    section("4. Systems measurements and cache validity")
    add("paragraph","The uncached arm holds a distinct frozen reference. The cached arm builds two scores per pair and releases the reference before optimization. A cache identity covers the reference state, model/tokenizer revision, template, end policy, preprocessing, dtypes, attention backend, microbatch and actual IDs/masks. Lookups check content by stable ID. Atomic replacement, counts and checksums detect stale, partial or corrupt caches; reordering cannot silently remap rows.")
    add("paragraph",f'Each timing trial runs in fresh processes from equal initial weights and RNG state. Cold cached versus uncached order alternates by trial; a warm-cache run then reuses that trial\'s verified cache. Warm runs always follow cold construction, an order limitation. Each arm performs {s["plan"]["warmup_updates"]} warm-up and {s["plan"]["measured_updates"]} measured updates. Training rows cycle if needed for this fixed timing workload; timing-only endpoints are not learning checkpoints. Timing repeats are not independent learning seeds. The timed block includes collation/transfers, policy work, reference work or lookup, backward, clipping and optimizer updates. No profiler runs inside reported measurements [R9,R10].')
    if s["plan"].get("seed_reduction_reason"):
        add("paragraph",s["plan"]["seed_reduction_reason"]+". No multi-seed quality uncertainty is estimated.")
    if s["plan"].get("timing_reduction_reason"):
        add("paragraph",s["plan"]["timing_reduction_reason"]+". Shorter blocks are more sensitive to transient timing noise.")
    add("paragraph","Each optimizer update atomically writes a small local progress record in both arms; this logging is inside the timed block. Interrupted runs retain the number of completed updates, while incomplete accumulation windows are not called completed training.")
    runtime=[["Arm","Training s","Whole process s","Train RSS MiB"]]
    names={"uncached_reference":"Uncached","cached_reference":"Cached cold","warm_cached_reference":"Cached warm"}
    for arm in ["uncached_reference","cached_reference","warm_cached_reference"]:
        if arm not in timing:continue
        m=timing[arm]
        runtime.append([names[arm],f'{m["training_seconds"]["median"]:.3f}',f'{m["whole_process_seconds"]["median"]:.3f}',f'{m["training_rss_mib"]["median"]:.1f}'])
    add("table",rows=runtime)
    add("figure",path="figures/runtime.png",caption="Figure 1. Three paired trials: median and observed range. Whole process includes imports, setup and warm-up; the measured block excludes them.")
    add("paragraph","Cold whole-process time includes interpreter/import costs, model/data loading, identity checks, reference construction or cache construction, warm-up and the timed training block. Timing-only runs declare no evaluation, generation or checkpoint work. Learning workflows separately include initial/final held-out evaluation, generation and a final checkpoint. Cold does not mean a cold OS filesystem cache. Warm time includes cache validation/read and excludes its previously paid construction cost.")
    section("5. Memory, cache amortization and uncertainty")
    add("figure",path="figures/memory.png",caption="Figure 2. CPU RSS scopes differ: training sampled every 10 ms versus Windows whole-process peak working set. No GPU memory measurement is available.")
    add("paragraph","Memory results retain resident model memory. Sampling can miss short training peaks, while Windows process peak includes imports and setup. A released reference may leave allocator pages resident, so theoretical parameter sizes are not substituted for measured RSS. GPU allocated/reserved metrics are explicitly unavailable. Lower training memory need not imply lower cold whole-workflow peak.")
    be=s["break_even"]
    if be["status"]=="projection":
        add("paragraph",f'Full-pass times projected from measured training throughput are U={be["uncached_pass_seconds"]:.3f} s and C={be["cached_pass_seconds"]:.3f} s. Measured reference precompute P={be["precompute_seconds"]:.3f} s gives the simplified equal-setup threshold E>P/(U-C)={be["equal_setup_threshold_passes"]:.3f} passes. Including the observed setup difference gives E>{be["measured_setup_threshold_passes"]:.3f}. These are projections, not executed extra epochs. Cache reuse requires the same reference, tokenizer, templates, IDs, masks and numerical settings.')
        add("equation","uncached(E) = Su + E*U; cached(E) = Sc + E*C")
    else:add("paragraph","No supported positive per-pass advantage was available for a break-even projection.")
    add("paragraph","Raw trial observations and median/range summaries are retained. No significance test is claimed from three timing repetitions. Runtime reduction uses (uncached-cached)/uncached; throughput increase uses (cached-throughput minus uncached-throughput)/uncached-throughput. These percentages differ. Seed variability, example-sampling uncertainty and timing noise are separate quantities; this small study does not estimate broad model-quality uncertainty.")
    add("paragraph","The one-epoch learning workflows are the direct end-to-end measurements including evaluation and checkpoint costs. Timing-block extrapolation assumes stable per-pair costs and repeated reuse; neither a favorable projected break-even nor a warm-cache gain proves a cold single-use gain. Full per-seed workflow times are stored alongside all checkpoints and metric rows.")
    section("6. Held-out outcomes and exploratory generations")
    lr=[["Seed","Initial loss","Final loss U/C","Positive %","Ties %"]]
    for r in learn:
        lr.append([str(r["seed"]),f'{r["initial_loss"]:.6f}',f'{r["final_loss_uncached"]:.6f} / {r["final_loss_cached"]:.6f}',f'{100*r["strict_positive_accuracy"]:.2f}',f'{100*r["tie_rate"]:.2f}'])
    add("table",rows=lr)
    if learn:
        add("paragraph",f'Across {len(learn)} seeds, the maximum paired test-margin difference is {max(r["max_test_margin_difference"] for r in learn):.4e}; the maximum paired training-loss difference is {max(r["max_trace_loss_difference"] for r in learn):.4e}. Final state hashes match for {sum(r["final_state_hash_equal"] for r in learn)}/{len(learn)} pairs. Per-example scores, losses, margins and reference values permit aggregate verification.')
        for row in learn:
            add("paragraph",f'The executed one-epoch workflow for seed {row["seed"]} took {row["uncached_workflow_seconds"]:.3f} s uncached and {row["cached_workflow_seconds"]:.3f} s cached, including initial/final evaluation, generation and final checkpoint writing. Each arm completed {row["updates"]} updates over {row["training_pairs"]} training pairs.')
    add("figure",path="figures/trajectory.png",caption="Figure 3. Training-window loss by update for the uncached arms. Paired cached differences are reported separately; these curves do not represent independent test evaluations.")
    add("paragraph","Held-out loss is a pair-weighted mean. Positive-margin accuracy counts beta*m>1e-6, and ties satisfy |beta*m|<=1e-6. When policy equals reference, all margins are ties: a strict-positive rate of zero is not zero language ability. Margin distributions are retained as min/quartiles/max. Test examples are evaluated only after the plan is frozen; final checkpoints are fixed, with no favorable seed or checkpoint selection.")
    add("paragraph","Eight reserved prompts receive baseline and final-policy greedy decoding with identical end rules and a maximum of 24 new tokens, further bounded by context capacity. These short examples are exploratory and cannot support a general helpfulness, truthfulness or hallucination claim. A blinded A/B CSV and separate mapping key have empty preference and notes fields. No human ratings or judge-model win rates were collected.")
    section("7. Discussion, reproducibility and responsibility")
    add("paragraph","The evidence separates objective correctness, matched trajectory agreement and systems cost. Analytic invariants and an executed upstream comparison cover a narrow precision/model scope. Caching is valid only for a fixed reference and identical scored inputs. CPU-specific timings on a shared laptop are not predictions for GPUs, other models, longer data or a shared-base disabled-adapter reference. The primary memory baseline is a separate model, so its savings cannot be transferred unchanged to that alternative design.")
    add("paragraph","Limitations include small seed/trial counts, aggressive length and single-turn filtering, AI-feedback labels, unknown pretraining overlap, no human evaluation, short generation caps and one local hardware environment. Deterministic settings improve local repeatability but do not promise cross-platform bitwise identity [R11]. The implementation has no packing, quantization, distributed training, paid judge, cloud execution or publication step.")
    add("heading","Reproduction and evidence map")
    add("code","python scripts/bootstrap.py --profile auto --max-experiment-hours 6\npython scripts/reproduce.py --profile toy --offline\npython scripts/reproduce.py --profile auto --offline --resume\npython scripts/reproduce.py --report-only\npython scripts/verify_artifacts.py")
    add("paragraph","Run commands in the project root, using the project .venv interpreter after bootstrap. Raw run specs, source hashes, parent plan hashes, commands, logs, metrics and checkpoint paths are in artifacts/runs. The immutable study plan and status live in artifacts/studies. artifacts/analysis/summary.json regenerates the tables and plots; claims.json maps central claims and cells to hashed metric artifacts. Source registry entries distinguish retrieval failures from retrieved sources.")
    add("paragraph","The manuscript, numerical tables and figures are generated from the same structured report blocks. A lightweight local ReportLab renderer produces the PDF because a terminal LaTeX compiler was unavailable. LaTeX/BibTeX sources remain provided. The original authors' implementation and library/model/data sources are attributed, with downloaded artifacts excluded from the code repository. Review licenses and derived examples before redistribution.")
    add("heading","Conclusion and next steps")
    add("paragraph","This executed scope supports only the reported correctness checks, paired learning results and local cost measurements. Next steps are owner review of the code, mathematics, raw evidence and licenses; a fresh reproduction from the lock; and, if warranted, a separately planned accelerator replication and genuine human evaluation. An abstention-training extension remains future work. No originality, publication, employment or peer-review claim follows from this report.")
    section("References and artifact audit")
    sources=read_json(ROOT/"artifacts/source_registry.json") if (ROOT/"artifacts/source_registry.json").exists() else []
    for r in sources:
        add("paragraph",f'[{r["id"]}] {r["author_organization"]}. {r["title"]}. {r["url"]} (retrieval: {r["status"]}; {r["retrieved_at"][:10]}).')
    add("paragraph","The artifact verifier checks required paths, plan/run hashes, metric regeneration, example aggregates, table regeneration, figure references, bibliography keys and declared scope. These mechanical checks do not prove every natural-language statement; a separate textual audit is recorded in docs/PAPER_AUDIT.md. Failed preparation/tests and negative outcomes are retained in the experiment log.")
    return blocks


def escape_tex(text):
    text=text.replace("\\",r"\textbackslash{}")
    for char,replacement in [("&",r"\&"),("%",r"\%"),("$",r"\$"),("#",r"\#"),("_",r"\_"),("{",r"\{"),("}",r"\}")]:
        text=text.replace(char,replacement)
    return text


def main():
    s=read_json(ROOT/"artifacts/analysis/summary.json")
    blocks=build_blocks(s)
    atomic_json(ROOT/"paper/report_blocks.json",blocks)
    markdown=[];tex=[r"\documentclass[10pt]{article}",r"\usepackage[margin=0.9in]{geometry}",r"\usepackage{graphicx,booktabs,url,hyperref}",r"\usepackage[T1]{fontenc}",r"\begin{document}"]
    for b in blocks:
        kind,text=b["kind"],b.get("text")
        if kind=="page":tex.append(r"\clearpage");continue
        if kind=="title":markdown.append("# "+text);tex.append(r"\title{"+escape_tex(text)+r"}\author{Aryan Golbaghi (intended owner; review pending)}\date{}\maketitle")
        elif kind=="heading":markdown.append("## "+text);tex.append(r"\section*{"+escape_tex(text)+"}")
        elif kind=="paragraph":
            markdown.append(text)
            tx=escape_tex(text)
            tx=re.sub(r"\[(R\d+(?:,R\d+)*)\]",lambda m:r"\cite{"+m[1]+"}",tx)
            tex.append(tx+"\n")
        elif kind in ["equation","code"]:
            markdown.append("```\n"+text+"\n```")
            tex.append("\\begin{verbatim}\n"+text+"\n\\end{verbatim}")
        elif kind=="table":
            rows=b["rows"];markdown.append("\n".join(["| "+" | ".join(rows[0])+" |","| "+" | ".join(["---"]*len(rows[0]))+" |"]+["| "+" | ".join(r)+" |" for r in rows[1:]]))
            tex.append(r"\begin{center}\small\begin{tabular}{"+"l"*len(rows[0])+r"}\toprule")
            tex.extend(" & ".join(escape_tex(v) for v in row)+r" \\" for row in rows)
            tex.append(r"\bottomrule\end{tabular}\end{center}")
        elif kind=="figure":
            if not (ROOT/"paper"/b["path"]).exists():continue
            markdown.append(f'![{b["caption"]}]({b["path"]})')
            tex.append(r"\begin{center}\includegraphics[width=0.94\linewidth]{"+b["path"]+"}\\\\\n"+escape_tex(b["caption"])+r"\end{center}")
    tex.extend([r"\bibliographystyle{plain}",r"\bibliography{references}",r"\end{document}"])
    (ROOT/"paper/paper.md").write_text("\n\n".join(markdown)+"\n",encoding="utf8")
    (ROOT/"paper/main.tex").write_text("\n".join(tex)+"\n",encoding="utf8")
    sources=read_json(ROOT/"artifacts/source_registry.json")
    bib=[]
    for r in sources:
        bib.append("@misc{"+r["id"]+",\n  title={"+r["title"]+"},\n  author={{"+r["author_organization"]+"}},\n  howpublished={\\url{"+r["url"]+"}},\n  note={Retrieved "+r["retrieved_at"][:10]+"; "+r["status"]+"}\n}")
    (ROOT/"paper/references.bib").write_text("\n\n".join(bib),encoding="utf8")
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,Image,Table,TableStyle,Preformatted,KeepTogether
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name="BodyMini",fontName="Times-Roman",fontSize=10,leading=13,spaceAfter=7,splitLongWords=True))
    styles.add(ParagraphStyle(name="CellMini",fontName="Helvetica",fontSize=8,leading=10,splitLongWords=True))
    story=[]
    def para(text,style="BodyMini"):
        return Paragraph(html.escape(text),styles[style])
    for b in blocks:
        kind,text=b["kind"],b.get("text")
        if kind=="page":story.append(PageBreak())
        elif kind=="title":story.extend([para(text,"Title"),Spacer(1,10)])
        elif kind=="heading":story.append(para(text,"Heading2"))
        elif kind=="paragraph":story.append(para(text))
        elif kind in ["equation","code"]:
            story.append(Preformatted(text,ParagraphStyle("MonoMini",fontName="Courier",fontSize=7.4,leading=10,spaceAfter=8),maxLineLength=95))
        elif kind=="table":
            rows=b["rows"];n=len(rows[0]);width=6.8*inch/n
            table=Table([[para(v,"CellMini") for v in row] for row in rows],colWidths=[width]*n,repeatRows=1,hAlign="LEFT")
            table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#E6EDF3")),("VALIGN",(0,0),(-1,-1),"TOP"),
                ("BOTTOMPADDING",(0,0),(-1,-1),6),("TOPPADDING",(0,0),(-1,-1),6),("LINEBELOW",(0,0),(-1,0),.6,colors.grey),
                ("LINEBELOW",(0,-1),(-1,-1),.4,colors.grey)]))
            story.extend([table,Spacer(1,8)])
        elif kind=="figure" and (ROOT/"paper"/b["path"]).exists():
            from PIL import Image as PILImage
            with PILImage.open(ROOT/"paper"/b["path"]) as im:w,h=im.size
            width=6.5*inch
            story.append(KeepTogether([Image(str(ROOT/"paper"/b["path"]),width=width,height=width*h/w),para(b["caption"],"CellMini"),Spacer(1,8)]))
    def footer(canvas,doc):
        canvas.setFont("Helvetica",8);canvas.setFillColor(colors.grey)
        canvas.drawString(.8*inch,.42*inch,"DPO-Mini | Correctness and reference-cost study")
        canvas.drawRightString(7.55*inch,.42*inch,str(doc.page))
    pdf=ROOT/"paper/paper.pdf"
    SimpleDocTemplate(str(pdf),pagesize=(8.5*inch,11*inch),rightMargin=.8*inch,leftMargin=.8*inch,
                      topMargin=.65*inch,bottomMargin=.65*inch,title=blocks[0]["text"],author="Aryan Golbaghi (intended owner; review pending)").build(story,onFirstPage=footer,onLaterPages=footer)
    import fitz
    document=fitz.open(pdf);render_dir=ROOT/"paper/preview";render_dir.mkdir(exist_ok=True)
    pages=[]
    for i,page in enumerate(document):
        page.get_pixmap(matrix=fitz.Matrix(1.2,1.2)).save(str(render_dir/f"page-{i+1:02d}.png"))
        pages.append({"page":i+1,"text_characters":len(page.get_text()),"images":len(page.get_images()),"width":page.rect.width,"height":page.rect.height})
    atomic_json(ROOT/"artifacts/analysis/pdf-inspection.json",{"renderer":"ReportLab; PyMuPDF raster QA","pages":pages,"page_count":len(document)})
    import shutil
    archive=ROOT/s["study"]/"reports"/now().replace(":","")
    shutil.copytree(ROOT/"paper",archive)
    print("Report written",pdf,"pages",len(document))


if __name__=="__main__":main()
