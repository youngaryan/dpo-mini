"""Generate all numerical summaries, tables, figures and empirical claim links."""
import _common
import argparse
import csv
import math
import statistics
from pathlib import Path
from dpo_mini.provenance import ROOT,atomic_json,digest,file_hash,read_json


def summarize(values):
    return {"n":len(values),"median":statistics.median(values),"min":min(values),"max":max(values)}


def compute(study: Path) -> dict:
    plan=read_json(study/"experiment_plan.json")
    status=read_json(study/"status.json")
    rows=[]
    for folder in status["runs"]:
        path=ROOT/folder/"metrics.json"
        m=read_json(path)
        if m["kind"]=="pilot":continue
        if read_json(ROOT/folder/"provenance.json")["parent_plan_hash"]!=plan["sha256"]:
            raise ValueError("Run belongs to another plan")
        rows.append({**m,"artifact":path.relative_to(ROOT).as_posix(),"artifact_sha256":file_hash(path)})
    timing={}
    for name in ["uncached_reference","cached_reference","warm_cached_reference"]:
        group=[r for r in rows if r["kind"]=="benchmark" and
               ((name=="warm_cached_reference" and r["cache_is_warm"]) or
                (r["arm"]==name and not r["cache_is_warm"]))]
        if not group:continue
        timing[name]={k:summarize([r[k] for r in group]) for k in
            ["training_seconds","whole_process_seconds","cache_build_or_load_seconds","reference_precompute_seconds","setup_including_cache_seconds","pairs_per_second","response_tokens_per_second"]}
        timing[name]["training_rss_mib"]=summarize([r["training_memory"]["sampled_peak_rss_bytes"]/1024**2 for r in group])
        timing[name]["workflow_rss_mib"]=summarize([r["whole_workflow_memory"]["os_process_peak_rss_bytes"]/1024**2 for r in group
                                                  if r["whole_workflow_memory"]["os_process_peak_rss_bytes"] is not None])
    learning=[]
    for seed in plan["seeds"]:
        pair={r["arm"]:r for r in rows if r["kind"]=="learning" and r["seed"]==seed}
        if len(pair)!=2:continue
        u,c=pair["uncached_reference"],pair["cached_reference"]
        e1=u["final_evaluation"]["test"];e2=c["final_evaluation"]["test"]
        by_id={r["id"]:r for r in e2["examples"]}
        learning.append({"seed":seed,"initial_loss":u["initial_evaluation"]["test"]["dpo_loss"],
            "final_loss_uncached":e1["dpo_loss"],"final_loss_cached":e2["dpo_loss"],
            "strict_positive_accuracy":e1["strict_positive_accuracy"],"tie_rate":e1["tie_rate"],
            "margin_mean":e1["margin_mean"],"margin_quantiles":e1["margin_quantiles"],
            "final_state_hash_equal":u["final_state_sha256"]==c["final_state_sha256"],
            "max_test_margin_difference":max(abs(r["reward_margin"]-by_id[r["id"]]["reward_margin"]) for r in e1["examples"]),
            "max_trace_loss_difference":max(abs(a["loss"]-b["loss"]) for a,b in zip(u["trace"],c["trace"])),
            "updates":u["updates"],"training_pairs":u["pairs"],
            "uncached_workflow_seconds":u["whole_process_seconds"],"cached_workflow_seconds":c["whole_process_seconds"],
            "uncached_artifact":u["artifact"],"cached_artifact":c["artifact"]})
    percentages={};breakeven={"status":"not measured"}
    if all(n in timing for n in ["uncached_reference","cached_reference"]):
        u=timing["uncached_reference"];c=timing["cached_reference"]
        for key in ["training_seconds","whole_process_seconds","training_rss_mib","workflow_rss_mib"]:
            a,b=u[key]["median"],c[key]["median"]
            percentages[key]={"uncached":a,"cached":b,"numerator":a-b,"denominator":a,
                              "percent_reduction":100*(a-b)/a,"formula":"100*(uncached-cached)/uncached"}
        a,b=u["pairs_per_second"]["median"],c["pairs_per_second"]["median"]
        percentages["throughput"]={"numerator":b-a,"denominator":a,"percent_increase":100*(b-a)/a}
        measured_pairs=plan["measured_updates"]*plan["config"]["effective_batch_pairs"]
        U=u["training_seconds"]["median"]*plan["config"]["train_pairs"]/measured_pairs
        C=c["training_seconds"]["median"]*plan["config"]["train_pairs"]/measured_pairs
        P=c["reference_precompute_seconds"]["median"]
        setup_diff=c["setup_including_cache_seconds"]["median"]-u["setup_including_cache_seconds"]["median"]
        breakeven={"status":"projection" if U>C else "no per-pass advantage", "uncached_pass_seconds":U,
                   "cached_pass_seconds":C,"precompute_seconds":P,"setup_difference_seconds":setup_diff,
                   "equal_setup_threshold_passes":P/(U-C) if U>C else None,
                   "measured_setup_threshold_passes":setup_diff/(U-C) if U>C else None,
                   "formula":"Sc+E*C < Su+E*U iff E>(Sc-Su)/(U-C), when U>C",
                   "assumptions":"fixed reference/tokenized training set; cache reusable; stationary per-pair cost; excludes evaluation and checkpoints; projected full passes, not executed epochs"}
    return {"study":study.relative_to(ROOT).as_posix(),"plan":plan,"timing":timing,"learning":learning,
            "percentages":percentages,"break_even":breakeven,"runs":rows,
            "upstream":read_json(ROOT/"artifacts/upstream.json"),"environment":read_json(ROOT/"artifacts/environment.json"),
            "execution_status":status,"completion_gate":"Executed pretrained study" if learning and plan["config"]["profile"]!="toy" else
            ("Executed toy validation study" if learning else "Implementation/report sources only")}


def tables(summary):
    rows=[["Arm","n","Training s median [range]","Whole process s median [range]","Training RSS MiB median"]]
    for arm in ["uncached_reference","cached_reference","warm_cached_reference"]:
        if arm not in summary["timing"]:continue
        m=summary["timing"][arm]
        fmt=lambda key:f'{m[key]["median"]:.3f} [{m[key]["min"]:.3f}, {m[key]["max"]:.3f}]'
        rows.append([arm,str(m["training_seconds"]["n"]),fmt("training_seconds"),fmt("whole_process_seconds"),f'{m["training_rss_mib"]["median"]:.1f}'])
    learning=[["Seed","Initial test loss","Final uncached loss","Final cached loss","Positive margin %","Ties %","Updates"]]
    for r in summary["learning"]:
        learning.append([str(r["seed"]),f'{r["initial_loss"]:.6f}',f'{r["final_loss_uncached"]:.6f}',
            f'{r["final_loss_cached"]:.6f}',f'{100*r["strict_positive_accuracy"]:.2f}',f'{100*r["tie_rate"]:.2f}',str(r["updates"])])
    return {"runtime":rows,"learning":learning}


def main():
    p=argparse.ArgumentParser();p.add_argument("--study",type=Path,required=True);args=p.parse_args()
    s=compute(args.study)
    analysis=ROOT/"artifacts/analysis";analysis.mkdir(parents=True,exist_ok=True)
    atomic_json(analysis/"summary.json",s)
    t=tables(s);atomic_json(analysis/"tables.json",t)
    for name,rows in t.items():
        path=ROOT/"paper/tables"/(name+".csv");path.parent.mkdir(exist_ok=True,parents=True)
        with path.open("w",newline="",encoding="utf8") as f:csv.writer(f).writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False})
    figures=ROOT/"paper/figures";figures.mkdir(exist_ok=True,parents=True)
    arms=[a for a in ["uncached_reference","cached_reference","warm_cached_reference"] if a in s["timing"]]
    labels=["Uncached","Cached cold","Cached warm"][:len(arms)]
    colors=["#0072B2","#D55E00","#009E73"][:len(arms)]
    if arms:
        fig,ax=plt.subplots(1,2,figsize=(9,3.1))
        for a,key,title in [(ax[0],"training_seconds","Measured training block"),(ax[1],"whole_process_seconds","Whole process, including warm-up")]:
            med=[s["timing"][r][key]["median"] for r in arms]
            low=[m-s["timing"][r][key]["min"] for r,m in zip(arms,med)]
            high=[s["timing"][r][key]["max"]-m for r,m in zip(arms,med)]
            a.bar(labels,med,color=colors,yerr=[low,high],capsize=4)
            a.set(ylabel="Seconds",title=title,ylim=(0,None))
        fig.suptitle(f'{s["plan"]["config"]["profile"]}, CPU; {s["plan"]["timing_trials"]} paired trials; median and range')
        fig.tight_layout();fig.savefig(figures/"runtime.png",dpi=180);plt.close(fig)
        fig,ax=plt.subplots(figsize=(7,3))
        for i,key in enumerate(["training_rss_mib","workflow_rss_mib"]):
            xs=[j+(i-.5)*.35 for j in range(len(arms))]
            ax.bar(xs,[s["timing"][r][key]["median"] for r in arms],width=.35,label=["Training sampled RSS","Whole-process OS peak RSS"][i],color=["#0072B2","#D55E00"][i])
        ax.set(xticks=range(len(arms)),xticklabels=labels,ylabel="MiB (CPU resident memory)",title="Memory scopes; medians of three trials")
        ax.legend();fig.tight_layout();fig.savefig(figures/"memory.png",dpi=180);plt.close(fig)
    if s["learning"]:
        fig,ax=plt.subplots(figsize=(7,3))
        for row in s["runs"]:
            if row["kind"]=="learning" and row["arm"]=="uncached_reference":
                ax.plot([v["update"] for v in row["trace"]],[v["loss"] for v in row["trace"]],label=f'seed {row["seed"]}')
        ax.set(xlabel="Optimizer update",ylabel="Training-window DPO loss",title="One-epoch trajectories; each point averages its preference pairs")
        ax.legend();fig.tight_layout();fig.savefig(figures/"trajectory.png",dpi=180);plt.close(fig)
        examples=next(r for r in s["runs"] if r["kind"]=="learning" and r["arm"]=="uncached_reference")
        sheet=[];key=[]
        for i,(a,b) in enumerate(zip(examples["initial_generations"],examples["final_generations"])):
            swap=int(digest(a["id"])[0],16)%2
            first,second=(b,a) if swap else (a,b)
            sheet.append([i+1,a["prompt"],first["text"],second["text"],"",""])
            key.append({"review_id":i+1,"example_id":a["id"],"A":"trained" if swap else "baseline","B":"baseline" if swap else "trained","seed":examples["seed"]})
        with (analysis/"blinded-review.csv").open("w",newline="",encoding="utf8") as f:
            csv.writer(f).writerows([["review_id","prompt","A","B","preference_A_B_tie","notes"],*sheet])
        atomic_json(analysis/"blinded-mapping-key.json",key)
    claims=[]
    def claim(id,text,key,formula=None):
        claims.append({"id":id,"claim":text,"artifact":"artifacts/analysis/summary.json","artifact_sha256":file_hash(analysis/"summary.json"),"metric_key":key,"transformation":formula})
    claim("C1","Timing cells are medians/ranges of actual fresh-process trials","timing","group by arm and cache state; median/min/max")
    claim("C2","Test outcomes come from fixed final checkpoints for every planned seed","learning")
    claim("C3","TRL comparison scope and errors","upstream")
    claim("C4","Runtime percentage reductions use uncached runtime as denominator","percentages")
    claim("C5","Break-even is a projection using measured setup and per-pair costs","break_even")
    claim("C6","Cached and uncached learning agreement is checked per seed","learning","final state hashes and maximum paired trajectory/margin differences")
    for row in s["runs"]:
        claims.append({"id":"RUN-"+digest(row["artifact"])[:12],"claim":"Raw observation used by summary","artifact":row["artifact"],"artifact_sha256":row["artifact_sha256"],"metric_key":"*"})
    atomic_json(ROOT/"artifacts/claims.json",claims)
    print(s["completion_gate"],"timing",list(s["timing"]),"learning seeds",[r["seed"] for r in s["learning"]])


if __name__=="__main__":main()
