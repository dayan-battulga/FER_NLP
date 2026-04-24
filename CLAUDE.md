# CLAUDE.md

Working context for the FiNER-ORD Financial NER project. Keep this file current. Update the "Current Status" section first when the repo state changes.

---

## Project

Take-home challenge for the ML/NLP Research Engineer internship at Dunedain. Build a Financial NER model on the `gtfintechlab/finer-ord` dataset. The task is BIO-tagged token classification with 7 labels across PER / LOC / ORG entities.

**Primary metric:** entity-level F1 via `seqeval` (confirmed with Daniel).  
**Bar:** Daniel's benchmark is about `0.9` entity F1 in 5 epochs. He explicitly said "F1 or efficiency" wins.  
**Current strategy:** the original `teacher_crf` run is the current leading teacher candidate on 3-seed mean, while the 5-epoch efficient-training run is the strongest efficiency story. Distillation, quantization, latency, and Pareto reporting are still planned work, not finished work.  
**Deliverables:** clean repo, stronger `README.md`, `REPORT.md` (6-8 pages), Pareto chart, best student checkpoint.

---

## Current Status

- [x] Initial one-seed baselines were run earlier (BERT-base, RoBERTa-base, RoBERTa-large). Best of that pass: RoBERTa-large at `0.8471` entity F1 on seed `88`.
- [x] Metric confirmed with Daniel (entity-level `seqeval`).
- [x] Repo scaffolded around config-driven vanilla and CRF training.
- [x] `src/data.py`, `src/evaluate.py`, `src/train.py`, `src/crf_model.py`, and `scripts/bio_repair.py` are written.
- [x] Configs exist for smoke test, vanilla teacher, CRF teacher, vanilla student, distilled student, efficient training, and label smoothing.
- [x] Smoke test passed. `smoke_test_seed88` completed and wrote `summary.json`, `predictions.json`, and CSV rows.
- [x] Phase B vanilla teacher completed for 3 seeds on CUDA. Aggregate test entity F1 is `0.8485 +/- 0.0022` (30 epochs); best committed seed is `phase_b_teacher_seed5768` at `0.8510`.
- [x] Phase B CRF comparison completed for 3 seeds. Aggregate test entity F1 is `0.8521 +/- 0.0018` (30 epochs); best committed seed is `teacher_crf_seed88` at `0.8540`.
- [x] Phase D efficient-training ablation completed for 3 seeds. Aggregate test entity F1 is `0.8481 +/- 0.0066` (5 epochs), with mean train time about `7.31` minutes; best seed is `efficient_training_seed78516` at `0.8553`.
- [x] DeBERTa-v3-large efficient configuration set up (5 epochs, `label_all_subwords=true`, `llrd_decay=0.9`, `head_lr=5e-5`, `warmup_ratio=0.1`, `lr_scheduler_type=cosine`). Smoke test ran locally on Mac but was aborted due to platform limits; all DeBERTa training moved to Colab. Seeds `88` and `5768` completed on Colab; seed `5768` test entity F1 is approximately `0.839`. Seed `78516` is pending. A 3-seed aggregate for DeBERTa efficient is not yet computed.
- [x] BIO repair was analyzed on the committed vanilla and CRF predictions. It produced essentially no entity-F1 gain.
- [x] Analysis notebooks now exist in `notebooks/`, including `compare_runs.ipynb`, `bio_fillter_testing.ipynb`, and `teacher_0_9_gap_analysis.ipynb`.
- [x] The stricter CRF variant with valid-token-only packing and hard BIO constraints was tried locally, performed worse than the original CRF, and was reverted. The checked-in `src/crf_model.py` is back on the original `teacher_crf` behavior.
- [x] Per-run artifacts are committed for the vanilla teacher, original CRF teacher, and efficient-training runs.
- [ ] Teacher is not officially locked yet. The current choice is between using the original CRF as the teacher or leaning harder into the efficiency story.
- [ ] Phase C vanilla student has not been run yet.
- [ ] Phase C distilled student is not runnable yet. `src/distill.py` is still empty, and `src.train.py` raises `NotImplementedError` when `use_distillation: true`.
- [ ] Quantization, latency, and Pareto chart scripts do not exist yet.
- [ ] `README.md` is still a placeholder.
- [ ] `REPORT.md` does not exist yet.
- [ ] Review call with Daniel is not tracked in-repo.

---

## Critical Rules

These are easy to get wrong. Do not.

1. **Tokenizer always uses `add_prefix_space=True` when `is_split_into_words=True`.** This applies uniformly to RoBERTa (BPE) and DeBERTa-v3 (SentencePiece) tokenizers; the data pipeline passes it unconditionally.

2. **Teacher and student must share a tokenizer for distillation.** DistilRoBERTa-base is the planned student because it matches the RoBERTa tokenizer family. Do not mix families.

3. **The original `teacher_crf` run is the current leading teacher candidate, but the teacher is not officially locked yet.** After that decision, do not change the teacher recipe.

4. **Label mapping integers preserve the HF dataset's original ordering.** `0 -> O`, `1 -> B-PER`, `2 -> I-PER`, `3 -> B-LOC`, `4 -> I-LOC`, `5 -> B-ORG`, `6 -> I-ORG`.

5. **Sanity check label mapping before any long run.** `python -m src.data` should show sensible entities such as `"Obama" -> B-PER` and continuous multi-token entities.

6. **Mask `-100` positions before the KL call in distillation.** Applying the mask after the KL call can produce NaNs.

7. **The current checked-in CRF implementation is the original one.** It uses `attention_mask` as the CRF mask and replaces `-100` with `0` before the CRF forward pass. Do not assume the stricter packed / hard-BIO variant is active.

8. **Save `predictions.json` per run.** BIO repair and error analysis depend on it.

9. **Three seeds for headline numbers.** Use `88, 5768, 78516` and report mean +/- std.

10. **Corrected hyperparameters, not the paper's:** `num_epochs=30`, `early_stopping_threshold=1e-3`, `fp16=True` on CUDA, `save_total_limit=2`.

11. **Do not try to run `use_distillation: true` configs through `src.train.py` until `src/distill.py` is implemented.**

12. **Do not headline a single lucky seed.** `efficient_training_seed78516` is higher than the best CRF seed, but the efficient-training 3-seed mean is not.

13. **Always annotate F1 comparisons with their epoch budget.** Vanilla teacher and CRF teacher ran for 30 epochs; `efficient_training` and `deberta_efficient` ran for 5 epochs. Comparing a 5-epoch result to a 30-epoch result without that context is misleading and has caused at least one planning mistake.

---

## Repo Map

```text
finer-ord/
├── CLAUDE.md                    # this file
├── README.md                    # placeholder right now
├── requirements.txt             # pinned deps
├── docs/
│   └── PROJECT_CONTEXT.MD       # deeper technical background
├── configs/
│   ├── README.md
│   ├── baseline_teacher.yaml
│   ├── smoke_test.yaml
│   └── baseline/
│       ├── teacher_crf.yaml
│       ├── student_vanilla.yaml
│       ├── student_distilled.yaml
│       ├── efficient_training.yaml
│       ├── label_smoothing.yaml
│       ├── deberta_smoke.yaml
│       └── deberta_efficient.yaml
├── notebooks/
│   ├── compare_runs.ipynb
│   ├── bio_fillter_testing.ipynb
│   ├── teacher_0_9_gap_analysis.ipynb
│   └── colab_runner.ipynb
├── src/
│   ├── data.py                  # dataset loading, grouping, tokenization
│   ├── evaluate.py              # seqeval + token metrics + confusion matrices
│   ├── train.py                 # vanilla multi-seed training entrypoint
│   ├── crf_model.py             # original CRF training path (restored)
│   ├── distill.py               # placeholder, currently empty
│   └── finer-ord.py             # HF dataset inspection helper
├── scripts/
│   ├── bio_repair.py            # post-processing for saved predictions
│   └── run_all.sh               # placeholder, currently empty
└── results/
    ├── results.csv
    ├── results_detailed.csv
    ├── phase_b_teacher_aggregate.json
    ├── teacher_crf_aggregate.json
    ├── efficient_training_aggregate.json
    ├── smoke_test_seed88/
    ├── phase_b_teacher_seed88/
    ├── phase_b_teacher_seed5768/
    ├── phase_b_teacher_seed78516/
    ├── teacher_crf_seed88/
    ├── teacher_crf_seed5768/
    ├── teacher_crf_seed78516/
    ├── efficient_training_seed88/
    ├── efficient_training_seed5768/
    ├── efficient_training_seed78516/
    ├── deberta_efficient_seed88/
    └── deberta_efficient_seed5768/
```

**Single source of truth for label mapping:** `src/data.py` defines `ID2LABEL`, `LABEL2ID`, and `NUM_LABELS`.

**Single entry point for metrics:** `src/evaluate.py::compute_detailed_metrics()` returns a JSON-serializable dict with token F1, entity F1, both confusion matrices, and per-class F1.

---

## Run Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
wandb login

# Verify data and pipeline
python -m src.data
python -m src.train --config configs/smoke_test.yaml --run-checks

# Phase B vanilla teacher
python -m src.train --config configs/baseline_teacher.yaml

# Phase B CRF comparison
python -m src.train --config configs/baseline/teacher_crf.yaml

# 5-epoch efficient-training ablation
python -m src.train --config configs/baseline/efficient_training.yaml

# BIO repair analysis on saved predictions
python scripts/bio_repair.py --predictions results/phase_b_teacher_seed5768/predictions.json --save
python scripts/bio_repair.py --predictions results/teacher_crf_seed88/predictions.json --save

# Phase C step 1, runnable today
python -m src.train --config configs/baseline/student_vanilla.yaml

# Phase C step 2, not runnable yet
# python -m src.train --config configs/baseline/student_distilled.yaml

# Optional ablation after distillation exists
# python -m src.train --config configs/baseline/label_smoothing.yaml
```

Add `--no-wandb` to `src.train` commands when logging is not desired.

---

## Key Technical Choices

**Models:** current leading teacher candidate is the original `roberta-large + CRF`; planned student is `distilroberta-base`.

**Implemented training paths:** `src.train.py` handles vanilla multi-seed training and dispatches to `src.crf_model.py` when `use_crf: true`.

**Teacher results so far:**
- Vanilla teacher mean test entity F1: `0.8485 +/- 0.0022` (30 epochs)
- Original CRF mean test entity F1: `0.8521 +/- 0.0018` (30 epochs)
- Efficient-training mean test entity F1: `0.8481 +/- 0.0066` (5 epochs)
- DeBERTa-v3-large efficient: 2 of 3 seeds complete, seed `5768` at approximately `0.839` test entity F1 (5 epochs). Aggregate pending.

**30-epoch recipe (vanilla teacher, CRF teacher):** `lr=1e-5`, `batch_size=8`, `num_epochs=30`, `patience=7`, `threshold=1e-3`, `weight_decay=0.01`, `fp16=True` on CUDA.

**5-epoch RoBERTa efficient recipe (`efficient_training`):** `lr=3e-5`, `num_epochs=5`, `warmup_ratio=0.1`, `lr_scheduler_type=cosine`.

**5-epoch DeBERTa efficient recipe (`deberta_efficient`):** `model=microsoft/deberta-v3-large`, `head_lr=5e-5`, `llrd_decay=0.9`, `num_epochs=5`, `warmup_ratio=0.1`, `lr_scheduler_type=cosine`, `label_all_subwords=true`, `fp16=True` on CUDA.

**CRF recipe currently checked in:** backbone LR `1e-5`, CRF head LR `1e-4`, `pytorch-crf==0.7.2`, `attention_mask` as CRF mask, and `-100 -> 0` before CRF forward.

**Distillation state:** config fields exist, but the actual trainer is not implemented yet.

**Subword alignment:** Both first-subword-only and label-all-subwords alignment are implemented in `src/data.py`. Behavior is controlled by the `label_all_subwords` config flag (default `false` for backward compatibility). DeBERTa efficient runs use `label_all_subwords=true`; all other committed runs use the default.

**Tracking:** W&B project `finer-ord`, committed `results/results.csv`, committed `results/results_detailed.csv`, and aggregate JSONs for vanilla teacher, original CRF, and efficient training.

**Pinned versions that matter:** `transformers==4.44.2`, `seqeval==1.2.2`, `numpy==1.26.4`, `torch==2.3.1`, `sentencepiece==0.2.0`. Do not upgrade blindly.

**Platform notes from committed runs:** the smoke test summary came from `mps` with fp16 disabled; the 3-seed vanilla teacher, original CRF, and efficient-training runs came from `cuda` with fp16 enabled. DeBERTa efficient runs came from Colab with per-run GPU type logged in `summary.json` under `runtime.gpu_type`; cross-seed wall-clock comparisons should account for possible GPU heterogeneity.

---

## Known Error Patterns

From the saved prediction analysis and strict span confusion notebooks:

- **Validation-to-test gap is large.** Vanilla and CRF both drop by about `0.067` entity F1 from validation to test.
- **Token F1 is already very high.** Both vanilla and CRF are around `0.985` token weighted F1, so the remaining problem is strict entity-span quality, not generic token classification.
- **ORG is the main bottleneck.** Missed and spurious ORG spans dominate the remaining error burden.
- **LOC is the second bottleneck.** Boundary and partial-overlap errors are still common.
- **Type confusion is small.** Exact same-span wrong-type mistakes are much less important than missed, spurious, and boundary-overlap errors.
- **BIO repair does not help.** On the committed vanilla and CRF predictions, post-hoc BIO repair produced essentially no entity-F1 gain.

Interventions tied to those patterns:

- **Original CRF head** -> completed, modestly better than vanilla on mean test entity F1, mostly through ORG.
- **Efficient training** -> completed, supports an efficiency story but not a better mean F1 story.
- **Label-all-subwords** -> implemented in `data.py` behind a config flag. Used by `deberta_efficient`. Its standalone effect on RoBERTa has not been tested in isolation.

---

## Style and Workflow Preferences

- **No em dashes.** Applies to report writing too.
- **Direct tone, grounded in specifics.** Avoid corporate jargon.
- **One deliverable per prompt, phase-gated.** Do not start distilled-student work until the teacher choice and the distillation implementation are both done.
- **Ship, then polish.** Get a working pipeline first, optimize second.
- **Honest about negative results.** A failed ablation is still useful evidence.

---

## Open Questions

- Should the original `teacher_crf` now be officially locked as the teacher, or is the gain over vanilla too small to justify the extra complexity?
- Is the better headline for Daniel now the modest CRF win, or the fact that `efficient_training` matches vanilla-level mean F1 in 5 epochs with much lower wall-clock? Note: `efficient_training` (5 epochs, `0.8481`) is statistically indistinguishable from vanilla teacher (30 epochs, `0.8485`), which is the cleanest form of the efficiency claim.
- If Phase C starts, should the teacher checkpoint be `results/teacher_crf_seed88/checkpoint-best`, or should the team prefer the best vanilla seed for simplicity?
- Should the stricter CRF attempt be mentioned in the report as a negative result, or kept as internal experimentation history only?
- What exact submission format Dunedain expects: repo link, PDF, or both?
- Does DeBERTa-v3-large efficient underperform RoBERTa efficient at 5 epochs because the `head_lr` of `5e-5` is too high, or because the 5-epoch budget itself is too tight for DeBERTa? This only matters if seed `78516` lands a 3-seed mean below `0.84` and we are choosing between retrying at a lower LR versus reverting to `teacher_crf` as the locked teacher.

---

## Notes for Future Sessions

- The deeper technical reference lives in `docs/PROJECT_CONTEXT.MD`. Keep `CLAUDE.md` short and operational.
- `results/phase_b_teacher_aggregate.json` and the committed per-seed summaries still reference the historical path `configs/phase_b_teacher.yaml`. The current checked-in config is `configs/baseline_teacher.yaml`.
- `configs/README.md` still shows the old `configs/phase_b_teacher.yaml` example.
- `configs/baseline/student_distilled.yaml` and `configs/baseline/label_smoothing.yaml` still point to placeholder teacher checkpoint paths. Update them after the teacher decision.
- `README.md`, `src/distill.py`, and `scripts/run_all.sh` are placeholders right now.
- `notebooks/bio_fillter_testing.ipynb` confirms that BIO repair did not move entity F1 on the committed vanilla or CRF runs.
- `notebooks/teacher_0_9_gap_analysis.ipynb` is the best artifact for explaining why the models are still below `0.90`.
- The current best single efficient-training seed (`0.8553`) is not a headline result by itself. Use 3-seed mean +/- std for claims.
- Results belong in the report. Keep this file focused on workflow, constraints, and current repo state.
- Any future model family beyond RoBERTa and DeBERTa-v3 should re-verify the label-all-subwords assumption by running `python -m src.data` with that family's tokenizer before headline experiments. SentencePiece and BPE split differently; behavior is tokenizer-specific.
