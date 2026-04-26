# Experiment configs

Each YAML file defines a single experiment cohort. Run with:

    python -m src.train --config configs/baseline/efficient_after_dapt.yaml

Multi-seed runs are defined inside the config (`seeds: [88, 5768, 78516]`).

## Naming convention
- `baseline/*.yaml` — baseline and ablation recipes
- `smoke_test.yaml` — quick validation before committing to long runs

## Reproducibility
- All seeds are fixed and tracked
- W&B tags map 1:1 to config file name
- Results land in `./results/<run_id>/` and are aggregated by seed after all runs complete