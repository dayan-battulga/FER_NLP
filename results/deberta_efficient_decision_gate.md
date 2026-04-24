# DeBERTa Efficient 3-Seed Decision Gate

## DeBERTa Efficient (5 epochs)
- Test entity F1 (mean ± std): **0.837422 ± 0.001777**
- Test token weighted F1 (mean ± std): **0.983401 ± 0.000336**
- Test per-class F1 (mean ± std):
  - PER: **0.939324 ± 0.000895**
  - LOC: **0.837375 ± 0.005631**
  - ORG: **0.787997 ± 0.004032**

## Aggregate Comparison
| Experiment | Epoch Budget | Test Entity F1 (mean ± std) | Test ORG F1 (mean ± std) |
|---|---:|---:|---:|
| phase_b_teacher | 30 | 0.848451 ± 0.002227 | 0.795355 ± 0.006075 |
| teacher_crf | 30 | 0.852132 ± 0.001783 | 0.803161 ± 0.004182 |
| efficient_training | 5 | 0.848141 ± 0.006585 | 0.802187 ± 0.007078 |
| deberta_efficient | 5 | 0.837422 ± 0.001777 | 0.787997 ± 0.004032 |

## GPU Heterogeneity
- Seed 88: `NVIDIA A100-SXM4-40GB`
- Seed 5768: `NVIDIA A100-SXM4-40GB`
- Seed 78516: `NVIDIA A100-SXM4-40GB`
- All three seeds used the same GPU type; wall-clock comparisons are less noisy.

## Decision Gate Categorization
- Path A threshold: mean >= 0.855
- Path B threshold: 0.840 <= mean < 0.855
- Path C threshold: mean < 0.840
- Current categorization from observed mean: **Path C (revert to teacher_crf as locked teacher)**
