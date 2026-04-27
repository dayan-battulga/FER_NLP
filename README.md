# FiNER-ORD Financial NER

Financial named entity recognition on the `gtfintechlab/finer-ord` dataset.
The task is BIO token classification over 7 labels:

- `O`
- `B-PER`, `I-PER`
- `B-LOC`, `I-LOC`
- `B-ORG`, `I-ORG`

The headline metric is strict entity-level F1 from `seqeval`. Token weighted
F1 is useful context, but it is not the primary optimization target.

## Project Direction

This repo is framed around the take-home brief's "F1 or efficiency wins"
criterion. The current teacher is locked, and the active work is Phase C:
distillation, dynamic INT8 quantization, latency measurement, and a Pareto
frontier chart.

The locked teacher is:

- `efficient_after_dapt_logit_ensemble`
- RoBERTa-large with FiNER-only DAPT
- CRF fine-tuning
- 3-seed logit / emission ensemble
- Test entity F1: `0.8634`

Do not retrain or modify the locked teacher path unless the project direction
changes. Phase C reads existing teacher artifacts and builds a deployable
student story around them.

## Dataset

Source: `gtfintechlab/finer-ord` on Hugging Face.

Splits:

- Train: 135 articles, 3,262 sentences, 80,531 tokens
- Validation: 24 articles, 402 sentences, 10,233 tokens
- Test: 42 articles, 1,075 sentences, 25,957 tokens

Entity counts:

- Train: PER 821, LOC 966, ORG 2026
- Validation: PER 138, LOC 193, ORG 274
- Test: PER 284, LOC 300, ORG 544

The label mapping lives in `src/data.py` and preserves the dataset's integer
ordering. Run this before any long experiment if the environment is new:

```bash
python -m src.data
```

The sanity output should show `Obama` as `B-PER` and continuous multi-token
entities.

## Repository Layout

Core modules:

- `src/data.py`: dataset loading, grouping, tokenization, label alignment
- `src/evaluate.py`: strict entity F1, token F1, confusion matrices
- `src/train.py`: vanilla training, config loading, mode dispatch
- `src/crf_model.py`: locked CRF teacher training and emission saving
- `src/dapt.py`: FiNER-only masked language model continued pretraining
- `src/distill.py`: offline student distillation from teacher emissions
- `src/losses.py`: Dice loss used for an exploratory negative result

Phase C scripts:

- `scripts/extract_train_emissions.py`
- `scripts/measure_latency.py`
- `scripts/quantize_student.py`
- `scripts/build_pareto.py`
- `scripts/ensemble_logits.py`

Documentation:

- `CLAUDE.md`: current operational context
- `docs/PROJECT_CONTEXT.MD`: detailed technical reference
- `docs/figures/`: Pareto chart outputs
- `REPORT.md`: deferred until real Phase C numbers exist

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional W&B logging is configured through the YAML files. Add `--no-wandb`
to training commands when logging is not desired.

The pinned versions are in `requirements.txt`. Do not upgrade the ML stack
casually because saved checkpoints and tokenizer behavior are part of the
experiment contract.

## Current Results

The source of truth for completed runs is:

- `results/results.csv`
- `results/results_detailed.csv`
- per-run `results/<run_id>/summary.json`

Do not invent metrics in documentation. Pull F1 values from the CSV or summary
files, and pull latency values from `results/latency/*.json` after measurements
have been run.

Known locked teacher values:

- Best single-model family: `efficient_after_dapt`
- 3-seed mean test entity F1: `0.8548 +/- 0.0038`
- Locked teacher ensemble: `efficient_after_dapt_logit_ensemble`
- Ensemble test entity F1: `0.8634`

## Command Flow

### 1. Verify data and smoke test

```bash
python -m src.data
python -m src.train --config configs/smoke_test.yaml --run-checks
```

### 2. Historical baselines

These are complete and do not need to be rerun for Phase C:

```bash
python -m src.train --config configs/baseline_teacher.yaml
python -m src.train --config configs/baseline/teacher_crf.yaml
python -m src.train --config configs/baseline/efficient_training.yaml
```

### 3. Locked DAPT teacher path

This path is complete. Do not retrain it during Phase C:

```bash
python -m src.dapt --config configs/baseline/dapt_roberta_large.yaml
python -m src.train --config configs/baseline/efficient_after_dapt.yaml
```

The fine-tuned teacher runs are:

- `results/efficient_after_dapt_seed88`
- `results/efficient_after_dapt_seed5768`
- `results/efficient_after_dapt_seed78516`

### 4. Locked teacher ensemble

The headline teacher is the CRF logit / emission ensemble:

```bash
python scripts/ensemble_logits.py \
  --runs efficient_after_dapt_seed88 efficient_after_dapt_seed5768 efficient_after_dapt_seed78516 \
  --mode logit \
  --use-crf \
  --output-name efficient_after_dapt_logit_ensemble
```

### 5. Extract train emissions for distillation

Student distillation is offline. Extract train emissions once and reuse them:

```bash
python scripts/extract_train_emissions.py \
  --runs efficient_after_dapt_seed88 efficient_after_dapt_seed5768 efficient_after_dapt_seed78516
```

Each run receives:

```text
results/<run_id>/train_emissions.npz
```

The file schema matches the existing validation and test emission dumps:

- `emissions`
- `attention_mask`
- `labels`

### 6. Distill the student

The student config is `configs/baseline/student_distilled.yaml`.

Smoke check:

```bash
python -m src.distill --config configs/baseline/student_distilled.yaml --smoke --no-wandb
```

Full 3-seed run:

```bash
python -m src.distill --config configs/baseline/student_distilled.yaml --no-wandb
```

Expected run directories:

- `results/student_distilled_seed88`
- `results/student_distilled_seed5768`
- `results/student_distilled_seed78516`

The student is `distilroberta-base`, uses first-subword-only alignment, and
does not use a CRF.

### 7. Ensemble the student

Student ensembling uses vanilla logits, not CRF emissions:

```bash
python scripts/ensemble_logits.py \
  --runs student_distilled_seed88 student_distilled_seed5768 student_distilled_seed78516 \
  --mode logit \
  --output-name student_distilled_logit_ensemble
```

### 8. Measure latency

Latency measurements write JSON files under `results/latency/`.

GPU FP32 examples:

```bash
python scripts/measure_latency.py \
  --runs efficient_after_dapt_seed88 student_distilled_seed88 \
  --device cuda
```

CPU FP32 examples:

```bash
python scripts/measure_latency.py \
  --runs efficient_after_dapt_seed88 student_distilled_seed88 \
  --device cpu
```

The script reports:

- torch version
- CPU model
- GPU model when available
- parameter count
- checkpoint size
- median latency at batch sizes 1 and 8
- p95 latency at batch sizes 1 and 8
- throughput for batch size 8

### 9. Quantize the student

Dynamic INT8 quantization is CPU-targeted and uses `nn.Linear` modules only:

```bash
python scripts/quantize_student.py \
  --runs student_distilled_seed88 student_distilled_seed5768 student_distilled_seed78516
```

Each student run receives:

```text
results/<student_run>/checkpoint-best-int8/
results/<student_run>/summary_int8.json
```

Measure INT8 latency separately on CPU:

```bash
python scripts/measure_latency.py \
  --runs student_distilled_seed88_int8 \
  --device cpu
```

### 10. Build the Pareto chart

Once F1, latency, and INT8 summaries exist:

```bash
python scripts/build_pareto.py
```

Outputs:

- `docs/figures/pareto.png`
- `docs/figures/pareto_data.csv`

The chart uses test entity F1 on the y-axis. The latency subplot uses median
single-example latency at batch size 1. The size subplot uses checkpoint size
in MB. The backing CSV includes throughput values where available.

## Phase C Acceptance Targets

Targets are goals, not assumed results:

- Distilled student single-model mean at or above `0.82` test entity F1
- Distilled student logit ensemble at or above `0.83` test entity F1
- INT8 student within `0.01` F1 of the FP32 student
- Student points down and left of the teacher on the Pareto chart

Use 3-seed mean plus standard deviation for headline single-model claims.
Never headline a single lucky seed.

## Important Constraints

- Do not modify `src/crf_model.py`, `src/dapt.py`, `src/data.py`, or `src/evaluate.py` for Phase C.
- Do not modify `configs/baseline/efficient_after_dapt.yaml`.
- Do not modify `configs/baseline/dapt_roberta_large.yaml`.
- Do not retrain the locked teacher checkpoints.
- Do not delete existing artifacts under `results/`.
- Do not add new dependencies silently.
- Keep `REPORT.md` deferred until real Phase C numbers exist.

## Optional Teacher-Side Experiments

These are not on the critical path:

- Packed 512-token re-inference via `scripts/reinfer_packed.py`
- 6-model multi-recipe ensemble if old CRF emissions are re-extracted
- Single-seed Dice loss spike via `configs/baseline/efficient_dice_seed88.yaml`

Run validation before test for packed re-inference. Do not let optional teacher
experiments block Phase C.
