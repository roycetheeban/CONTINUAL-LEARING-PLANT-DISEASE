# Cat B Method Selection and Transfer Plan (Case 1)

## Goal
Finalize a strong Cat B (new class introduction 5 -> 7) training method that can outperform naive baselines, then transfer the same approach to other cases.

## Compared Experiment Sets
- `catB_new_classes` (normal)
- `catB_new_classes tuned` (tuned)

## Current Best Numbers (from summaries)

### Baseline to Beat
- `cat_b_naive_full_retrain_head_expand` (normal)
  - Cycle 1: `Old F1=0.6450`, `New F1=0.3228`, `Old Acc=0.8522`, `New Acc=0.9439`
  - Cycle 2: `Old F1=0.6539`, `New F1=0.3207`, `Old Acc=0.8707`, `New Acc=0.9342`

### Best CL Candidates
- `Replay (tuned)` cycle 2: `Old F1=0.6394`, `New F1=0.3229`, `Old Acc=0.8443`, `New Acc=0.9478`
- `EWC (normal)` cycle 2: `Old F1=0.6375`, `New F1=0.3163`, `Old Acc=0.8408`, `New Acc=0.9149`

## Selection Decision
- Keep: `EWC`, `Replay`
- Drop from optimization target: `Isolation` (good retention, weak new-class plasticity in these runs)
- Keep as reference baseline only: `Naive FT`, `Naive Full Retrain`

## Important Wiring Issue Found
In `catB_new_classes tuned`, naive configs write to non-tuned output paths.  
Result: tuned summary shows `NA` for naive rows, even though naive configs exist.

## Final Optimization Direction (Cycle 1/2 only)
Do not retrain Part 1 base model. Only tune Cat B cycle configs:

1. Replay:
- Use `old_batch_size:new_batch_size = 16:16`
- Keep split head LR
- Reduce `lr_head_new` to `0.0008-0.0010`
- Keep lower LRs in cycle 2

2. EWC:
- Keep Fisher-before-head-expansion
- Use `old_batch_size:new_batch_size = 16:16`
- Sweep `lambda` in `400-1200`

## Acceptance Rule (to claim win)
At Cycle 2, CL run should beat `naive_full_retrain` old retention while keeping new class learning strong:
- Target minimum: `Old F1 > 0.6539`
- Maintain: `New F1 >= 0.3207` (or better)
- Primary ranking: `(Old F1 + New F1) / 2`

## Live Tracker (Update After Every New Training Comparison)

Use this section as a rolling tracker. Do not delete old rows; append new rows with date/time and run ID.

### A) Best-Per-Method Tracker (Case 1 Cat B)

| Method | Current Best Source | Cycle | Old F1 | New F1 | Old Acc | New Acc | Joint F1 `(Old+New)/2` | Status vs Naive Full Retrain |
|---|---|---:|---:|---:|---:|---:|---:|---|
| EWC | catB_new_classes (normal) | 2 | 0.6375 | 0.3163 | 0.8408 | 0.9149 | 0.4769 | Below baseline |
| Replay | catB_new_classes tuned | 2 | 0.6394 | 0.3229 | 0.8443 | 0.9478 | 0.4812 | Below old F1 baseline, above new F1 baseline |
| Naive Full Retrain (baseline) | catB_new_classes (normal) | 2 | 0.6539 | 0.3207 | 0.8707 | 0.9342 | 0.4873 | Baseline to beat |

### B) Run Comparison Log (Append Only)

| Date | Run Tag | Source Folder | Method | Cycle | Key Params | Old F1 | New F1 | Decision |
|---|---|---|---|---:|---|---:|---:|---|
| 2026-05-16 | initial-compare | normal+tuned summaries | Replay | 2 | old:new=12:20, split-LR | 0.6394 | 0.3229 | Keep candidate |
| 2026-05-16 | initial-compare | normal+tuned summaries | EWC | 2 | lambda=400 (tuned), best from normal | 0.6375 | 0.3163 | Keep candidate |
| 2026-05-16 | initial-compare | normal summary | Naive Full Retrain | 2 | old:new=16:16 | 0.6539 | 0.3207 | Baseline |

### C) Promotion Rules (When to Move Code into `catB_finalized`)

Promote a method/config as "finalized winner" only if all are true:
1. Beats `Naive Full Retrain` on `Old F1` at Cycle 2.
2. Keeps `New F1 >= 0.3207`.
3. Improves or matches joint F1 `(Old+New)/2` against baseline.
4. Has full artifacts (`metrics.json`, `train_log.csv`, checkpoint, plus Fisher/theta for EWC or replay manifest for Replay).

### D) Finalized Code Tracker

| Method | Finalized Config Path | Finalized Code Path | Last Updated | Reason |
|---|---|---|---|---|
| EWC | `catB_finalized/ewc_head_expand/configs/` | `catB_finalized/ewc_head_expand/code/train_catB_ewc.py` | 2026-05-16 | Current best CL candidate set |
| Replay | `catB_finalized/experience_replay_head_expand/configs/` | `catB_finalized/experience_replay_head_expand/code/train_catB_replay.py` | 2026-05-16 | Current best CL candidate set |

## Ready Prompt for Another LLM (Code Transfer + Run)
Use this prompt directly:

```text
Project path:
G:\AIOT_PHASE_03\CONTINUAL-LEARING-PLANT-DESEASE

Task:
Transfer the best-performing Cat B Case 1 CL implementations (Replay and EWC) to Case 2 and Case 3 continuation pipelines, preserving folder structure and reproducibility artifacts.

Context:
- Source implementations are in:
  - experiments/part1_case1_scratch/Continuation/catB_new_classes tuned/experience_replay_head_expand
  - experiments/part1_case1_scratch/Continuation/catB_new_classes tuned/ewc_head_expand
- Baseline to beat is naive_full_retrain cycle2 in Case 1:
  - Old F1=0.6539, New F1=0.3207
- We only tune Cat B cycle1/2 configs. Do NOT retrain Part 1 base models.

Requirements:
1) Copy/adapt Replay + EWC code/config to:
   - experiments/part1_case2_imagenet_finetune/Continuation/catB_new_classes/
   - experiments/part1_case3B_plantvillage_finetune/Continuation/catB_new_classes/
2) Keep these method rules:
   - Replay and EWC only (main optimization methods)
   - old_batch_size:new_batch_size = 16:16
   - split head LR retained
   - replay lr_head_new around 0.0008-0.0010
   - EWC lambda sweep 400-1200
   - Fisher computed before head expansion
3) Ensure outputs are written to the correct case-specific folders (no path leakage from Case 1).
4) Keep artifacts: metrics.json, train_log.csv, checkpoints, fisher/theta for EWC, replay manifest for Replay, figures.
5) Add or update a summary markdown per case with Cycle 1 and Cycle 2 results:
   - Old F1, New F1, Old Acc, New Acc, Forgetting (if available), Peak VRAM
6) Update tracker document after each run comparison:
   - `experiments/part1_case1_scratch/Continuation/CATB_METHOD_SELECTION_AND_TRANSFER.md`
   - Update sections: "Best-Per-Method Tracker", "Run Comparison Log", and "Finalized Code Tracker"
   - Append rows, do not delete historical rows
6) Return:
   - Exact files changed
   - Exact run commands for each case and cycle
   - A short comparison table vs naive baselines.

Implementation note:
If helper paths/imports are hardcoded, make them case-relative and robust.
```

## Next Action
Run Replay + EWC for Case 2 and Case 3 using this transferred setup, then compare against each case naive full retrain baseline.

## Execution Order (Must Follow)
1. Read this file first and use only selected methods (`EWC`, `Replay`) for transfer/tuning.
2. Run Case 2 and Case 3 Cat B Cycle 1 and Cycle 2 for `EWC` and `Replay`.
3. Compare against each case `Naive Full Retrain` baseline.
4. Update tracker tables in this file.
5. Clean low-performing outputs using archive-first policy (do not hard delete first).

## Commands (Template)

Use this sequence from repo root:

```powershell
# Case 2
python experiments/part1_case2_imagenet_finetune/Continuation/catB_new_classes/run_catB.py ewc 1
python experiments/part1_case2_imagenet_finetune/Continuation/catB_new_classes/run_catB.py ewc 2
python experiments/part1_case2_imagenet_finetune/Continuation/catB_new_classes/run_catB.py replay 1
python experiments/part1_case2_imagenet_finetune/Continuation/catB_new_classes/run_catB.py replay 2

# Case 3
python experiments/part1_case3B_plantvillage_finetune/Continuation/catB_new_classes/run_catB.py ewc 1
python experiments/part1_case3B_plantvillage_finetune/Continuation/catB_new_classes/run_catB.py ewc 2
python experiments/part1_case3B_plantvillage_finetune/Continuation/catB_new_classes/run_catB.py replay 1
python experiments/part1_case3B_plantvillage_finetune/Continuation/catB_new_classes/run_catB.py replay 2
```

## Low-Performer Cleanup Rule

After comparison for a case:
- Keep winners: best `EWC` + best `Replay` + `Naive Full Retrain` baseline.
- Low performers: archive them under `_archived_low_performers/` (same case folder), then optionally delete later.
- Never remove metrics/logs for winners or baseline.

### Archive-First Example
```powershell
# Example path pattern (adjust case/method)
New-Item -ItemType Directory -Force -Path "experiments/<case>/Continuation/catB_new_classes/_archived_low_performers" | Out-Null
Move-Item -Force "experiments/<case>/Continuation/catB_new_classes/<low_method>/outputs" "experiments/<case>/Continuation/catB_new_classes/_archived_low_performers/<low_method>_outputs"
```

## Tracker Update Requirement
Every time new training results are produced:
1. Update "Best-Per-Method Tracker"
2. Append "Run Comparison Log"
3. Update "Finalized Code Tracker" if a better config/code becomes winner
4. Record archived methods and reason in run log decision column
