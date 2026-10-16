"""Mechanical audit; intentionally not a proof of all manuscript prose."""
import _common
import csv
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from dpo_mini.provenance import ROOT,atomic_json,digest,file_hash,read_json,source_manifest,now
from analyze import compute,tables


def main():
    final_latest=ROOT/'artifacts/finalization/latest.json'
    if final_latest.exists():
        folder=ROOT/read_json(final_latest)['path']
        if (folder/'evidence/historical.json').exists():
            historical=read_json(folder/'evidence/historical.json')
            latest_path=ROOT/'artifacts/latest_study.json'
            latest_id=Path(read_json(latest_path)['path']).name if latest_path.exists() else None
            if latest_id is None or all(r['run_id'].startswith(latest_id) for r in historical['runs']):
                from dpo_mini.finalization.report import verify
                verify(folder)
                return
    required=["README.md","requirements.lock","artifacts/environment.json","artifacts/dependency_resolution.log",
        "artifacts/source_registry.json","artifacts/experiment_plan.json","artifacts/data_manifest.json",
        "artifacts/claims.json","artifacts/analysis/summary.json","paper/paper.md","paper/main.tex","paper/references.bib","paper/paper.pdf",
        "docs/DECISIONS.md","docs/EXPERIMENT_LOG.md","docs/INTERVIEW_NOTES.md"]
    for name in required:
        assert (ROOT/name).is_file(),f"Missing {name}"
    s=read_json(ROOT/"artifacts/analysis/summary.json")
    study=ROOT/s["study"]
    plan=read_json(study/"experiment_plan.json")
    body={k:v for k,v in plan.items() if k!="sha256"}
    assert digest(body)==plan["sha256"],"Frozen plan hash mismatch"
    assert plan["code_hash"]==source_manifest()["sha256"],"Source changed after freeze; declare a new plan revision"
    assert digest(read_json(ROOT/"artifacts/data_manifest.json"))==plan["data_manifest_sha256"]
    assert file_hash(ROOT/f'artifacts/prepared/{plan["config"]["profile"]}/data.json')==plan["data_file_sha256"]
    tests=ET.parse(study/"core.xml").getroot()
    suites=[tests] if tests.tag=="testsuite" else list(tests)
    assert sum(int(x.get("failures",0))+int(x.get("errors",0)) for x in suites)==0,"Core failure invalidates performance conclusions"
    count=sum(int(x.get("tests",0)) for x in suites)
    assert count>=14,"Mandatory core suite incomplete"
    rebuilt=compute(study)
    for key in ["timing","learning","percentages","break_even"]:
        assert s[key]==rebuilt[key],f"Metric regeneration differs: {key}"
    assert tables(s)==read_json(ROOT/"artifacts/analysis/tables.json"),"Table regeneration mismatch"
    for name,rows in tables(s).items():
        with (ROOT/"paper/tables"/(name+".csv")).open(encoding="utf8",newline="") as f:
            assert list(csv.reader(f))==rows
    for c in read_json(ROOT/"artifacts/claims.json"):
        assert file_hash(ROOT/c["artifact"])==c["artifact_sha256"],f"Claim artifact changed: {c['id']}"
    for source in read_json(ROOT/"artifacts/source_registry.json"):
        if source["status"]=="retrieved":
            assert file_hash(ROOT/source["artifact"])==source["sha256"]
    for run in s["runs"]:
        provenance=read_json((ROOT/run["artifact"]).parent/"provenance.json")
        assert provenance["status"]=="completed" and provenance["parent_plan_hash"]==plan["sha256"]
        assert provenance["source"]["sha256"]==plan["code_hash"]
        if run["arm"]=="cached_reference":assert run["reference_training_forward_calls"]==0
        if run["kind"]=="learning":
            assert run["pairs"]==plan["config"]["train_pairs"]
            assert sum(x["pairs"] for x in run["trace"])==run["pairs"]
            assert (ROOT/run["checkpoint"]).exists()
            for stage in ["initial_evaluation","final_evaluation"]:
                for split,e in run[stage].items():
                    assert len(e["examples"])==e["pairs"]
                    assert math.isclose(sum(x["loss"] for x in e["examples"])/e["pairs"],e["dpo_loss"],abs_tol=1e-12)
                    if stage=="initial_evaluation":
                        assert math.isclose(e["dpo_loss"],math.log(2),abs_tol=1e-6)
                        assert e["tie_rate"]==1
        else:
            assert run["pairs"]==plan["measured_updates"]*plan["config"]["effective_batch_pairs"]
    assert len(s["learning"])==len(plan["seeds"]),"Incomplete paired learning seeds"
    assert all(t["training_seconds"]["n"]==plan["timing_trials"] for t in s["timing"].values())
    md=(ROOT/"paper/paper.md").read_text(encoding="utf8")
    for ref in re.findall(r"!\[[^\]]*\]\(([^)]+)\)",md):assert (ROOT/"paper"/ref).exists(),ref
    bib=(ROOT/"paper/references.bib").read_text(encoding="utf8")
    for citation in re.findall(r"\[(R\d+(?:,R\d+)*)\]",md):
        for key in citation.split(","):assert "@misc{"+key+"," in bib,key
    assert s["completion_gate"] in md
    assert "pending author review" in md
    if plan["config"]["profile"]=="toy":assert "Toy Validation Only" in md
    assert not re.search(r"statistically significant|state-of-the-art|guaranteed publication",md,re.I)
    with (ROOT/"artifacts/analysis/blinded-review.csv").open(encoding="utf8",newline="") as f:
        for row in csv.DictReader(f):assert row["preference_A_B_tie"]==row["notes"]==""
    import fitz
    pdf=fitz.open(ROOT/"paper/paper.pdf")
    assert len(pdf)>=3 and all(len(p.get_text())>100 for p in pdf)
    result={"status":"passed","checked_at":now(),"core_tests":count,"raw_runs":len(s["runs"]),
            "paired_training_seeds":len(s["learning"]),"pdf_pages":len(pdf),
            "scope":s["completion_gate"],"limitation":"Mechanical verification does not prove every natural-language claim"}
    atomic_json(ROOT/"artifacts/verification.json",result)
    print(result)


if __name__=="__main__":main()
