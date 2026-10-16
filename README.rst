DPO-Mini
========

A reproducible correctness and reference-caching study of Direct Preference
Optimization on SmolLM2-135M-Instruct with LoRA.

The project implements the original sigmoid DPO objective and tests when a
fixed reference can be shared or cached without changing the observed training
trajectory. It measures training time, complete benchmark-process time and
memory separately, including cold cache construction.

`Technical report <paper/paper.pdf>`_ |
`Research case study <https://aryangolbaghi.com/research/dpo-mini>`_ |
`Reproduction guide <docs/reproduction.rst>`_

Results
-------

Three fresh-process trials per strategy on an AMD Ryzen 5 PRO 7540U, with four
PyTorch computation threads and FP32 CPU execution. Each trial runs five warm-up
updates followed by twenty measured updates, with sixteen preference pairs per
update. The measured block contains 320 processed pairs; seed 42 is fixed.

.. list-table:: Median results
   :header-rows: 1
   :widths: 34 22 22 22

   * - Reference strategy
     - Training block (s)
     - Whole process (s)
     - Training RSS (MiB)
   * - Separate frozen reference
     - 901.831
     - 1151.478
     - 2632.4
   * - Shared frozen base
     - 902.774
     - 1147.964
     - 2098.9
   * - Cold shared-base cache
     - 609.977
     - 1013.617
     - 2095.7

Relative to the shared-base baseline, caching reduced the measured training
block by **32.43%** and the whole benchmark process by **11.70%**. Cold train-cache
construction took a median 228.976 seconds and is included in process time.
The training RSS difference between those strategies was only 3.2 MiB: most of
the memory saving against a separate reference came from sharing the base.

Whole benchmark time includes loading, warm-up, cache construction and endpoint
output. It excludes held-out evaluation, generation and full learning
checkpoints. All nine benchmark runs have matching update traces and canonical
model-state hashes. Three timing trials describe this host and workload; they
do not establish statistical significance or variability across learning seeds.

Correctness and the numerical qualification
------------------------------------------

* The working and fresh-environment core suites each passed 19 tests, with zero
  failures, errors or skips. Coverage includes causal masks, completion-token
  sums, accumulation, cache integrity, frozen references and deterministic resume.
* Shared-base checks cover 274 named base parameters/buffers, all 120 trainable
  adapter gradients, state restoration and matched short training trajectories.
* Actual pretrained comparisons against installed TRL 0.24.0 **exceeded the
  original FP32 absolute tolerances**. Maximum score/loss differences reached
  6.4087e-4 / 1.6809e-5; the trained gradient difference was 2.0103e-6 against a
  2e-6 bound. Identical input logits isolate the discrepancy to score-reduction
  arithmetic. Separate FP64-reduction diagnostics recovered identical adapter
  gradients. Original failures and the subsequent numerical adjudication remain
  in the evidence. Full TRL Trainer equivalence is not established.

The historical learning experiment used one paired seed, sixteen updates and a
256/64/64 training/validation/test split. Final test DPO loss was 0.690726, with
39/64 strict-positive margins and no ties. These are objective-level measurements,
not human preference or general language-quality results. Historical and new
timing cohorts are reported separately in the paper.

Install and verify
------------------

Use Python 3.12 and a project-local virtual environment. The lockfile records the
tested dependency versions. On Windows::

    python -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements.lock
    .\.venv\Scripts\python.exe -m pytest -q
    .\.venv\Scripts\python.exe scripts/finalize.py --offline --verify-only
    .\.venv\Scripts\python.exe scripts/finalize.py --offline --report-only

On Linux/macOS, substitute ``.venv/bin/python`` for the Windows executable.
Dependency installation can use the network; the two finalizer commands above
do not download or train. Report regeneration may create ignored Markdown
intermediates. The public repository itself contains no tracked Markdown files.

The repository includes source, tests, pinned configuration, numeric evidence,
tables, figures, PDF, LaTeX and bibliography. Model weights, large checkpoints,
full datasets, environments, caches and internal working logs are excluded.
Verification reports those limits explicitly. Full pretrained reruns require
separately obtaining the pinned public inputs; this is not a self-contained
offline training distribution.

Study scope and attribution
---------------------------

This is a correctness and systems-measurement study. DPO and reference-score
caching are established techniques. One nonexclusive CPU host, one learning
seed, a small filtered AI-feedback dataset, unknown pretraining overlap and no
human quality ratings limit generalisation.

The public edition was prepared
on 16 October 2026 from the completed 15 October study; archived status records
retain their original dates and scope.

See `third-party notices <THIRD_PARTY_NOTICES.txt>`_ and the paper's primary
references. Upstream licenses are retained. No additional project-wide license
grant or right to redistribute excluded model/data materials is implied.
