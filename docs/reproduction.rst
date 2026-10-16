Reproducing and auditing DPO-Mini
================================

The lightweight route
---------------------

Create an isolated Python 3.12 environment, install ``requirements.lock``, then
run ``python -m pytest -q`` using that environment. These 19 core tests do not
download models during import or collection. Model-level integrations are
separate commands and require the pinned model cache.

``python scripts/finalize.py --offline --verify-only`` checks included evidence
hashes, frozen measurement source, trial compatibility, table values and
independently recomputed per-example metrics. It explicitly reports that omitted
weights and full checkpoints cannot be reopened in this distribution.

``python scripts/finalize.py --offline --report-only`` regenerates the report,
tables and figures from the recorded evidence without training or downloading.
Generated Markdown working documents are ignored by Git. The public README and
this guide remain separate from those generated documents.

Evidence map
------------

``artifacts/finalization/latest.json`` identifies the study directory. Inside it,
``evidence/manifest.json`` hashes the portable raw numeric evidence;
``evidence/followup.json`` contains all nine timing runs and both workflow replays;
``evidence/historical.json`` retains original measurements and per-example
evaluation scores. ``claims.json`` maps report claims to fields and calculations.
``summary.json`` retains full-precision derived values.

``paper/report_blocks.json`` is the shared structured manuscript.
``paper/tables/`` holds the published CSV tables. ``paper/main.tex`` and
``paper/references.bib`` are editable sources. The PDF is generated with ReportLab;
LaTeX compilation is not claimed.

Clean-environment scope
-----------------------

The completed study created a genuinely fresh isolated environment and installed
the pinned packages from verified cached payloads. It checked dependency
consistency, module origins and installed file hashes, and ran the complete core
suite, pinned tiny-model TRL comparison, pretrained initial/trained-adapter
comparison, short three-strategy trajectory, toy resume tests, raw-evidence
verification and report regeneration. Full timing and learning campaigns were
not repeated in that clean environment.

The public edition changes presentation only. The frozen numerical worker
source and all evidence files retain their original hashes. Publication checks
are recorded separately from the original clean-environment records.

Fresh experiments
-----------------

``python scripts/bootstrap.py --profile toy`` creates a local environment and
runs the small synthetic protocol. ``--offline`` is suitable only when its
dependencies are already cached.

``python scripts/bootstrap.py --profile compact`` installs pinned dependencies,
acquires the public model/data inputs and runs the original compact study.
This is a computational experiment, not the lightweight report check.
The frozen follow-up finalizer requires the local full-study manifests and
initial-state artifacts; those large execution inputs are intentionally omitted
from the public evidence repository. Do not use ``--resume`` on the lightweight
distribution as a substitute for reconstructing them.

Pinned inputs are HuggingFaceTB/SmolLM2-135M-Instruct at revision
``12fd25f77366fa6b3b4b768ec3050bf629380bac`` and
HuggingFaceH4/ultrafeedback_binarized at revision
``3949bf5f8c17c394422ccfab0c31ea9c20bdeb85``. Model and data terms apply separately.

Interpretation
--------------

Use completion-token sums and the original sigmoid DPO objective, beta 0.1.
The reference remains fixed. Shared-base reference scoring disables the policy
adapter temporarily, restores its state and does not enter the gradient graph.
Caches are valid only while model, reference, examples, tokenizer, masks,
precision and declared execution identity remain compatible.

Runtime reduction is ``100 * (baseline - comparison) / baseline``. A training
block, a cold benchmark process and a complete learning workflow have different
boundaries. The historical workflow's 109.093-second saving reconciles with its
recorded aggregate stages; finer historical evaluation/generation and per-split
cache attribution remains unresolved. The instrumented replay measures its own
costs and does not retroactively supply missing historical timings.
