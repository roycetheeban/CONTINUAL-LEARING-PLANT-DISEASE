# Cat B Finalized - Full Context and Current State

## 1) Big Picture
- Project goal: edge-optimized continual learning for plant disease classification.
- Backbone: MobileNetV3-Small.
- Research flow:
  - Part 1: initial training with 3 base cases.
  - Part 2 Cat B: new-class introduction (5 -> 7 classes) with CL methods.

## 2) Part 1 Base Cases
- Case 1: train from scratch on 5 tomato classes.
- Case 2: ImageNet fine-tune.
- Case 3: PlantVillage pretrain + fine-tune.

Continuation in this folder focuses on **Case 1 Cat B** method development and optimization patterns that can be transferred to Case 2/3.

## 3) Cat B Setup (Case 1)
- Old classes (T1-T5): initial tomato classes.
- New classes (T6-T7): introduced in Cat B.
- Cycle 1 and Cycle 2 stream structure:
  - old stream: `data/02_tomato_5cls/cl_cycle{1,2}_stream`
  - new stream: `data/03_tomato_new_2cls/cl_cycle{1,2}_stream`
  - replay memory: `data/02_tomato_5cls/replay_buffer`
- Validation/test:
  - old val/test from 5-class set
  - new val/test from 2-class set

## 4) Baseline and Why It Matters
- Baseline method: `naive_full_retrain_head_expand`.
- Key baseline (Cycle 2):
  - Old F1: `0.6539`
  - New F1: `0.3207`
  - Joint F1: `0.4873`

This is the score CL methods are trying to beat.

## 5) Methods in Finalized Folder
- `ewc_head_expand`
- `experience_replay_head_expand`
- `hybrid_ewc_replay_head_expand` (Replay + light EWC)

## 6) Training Design Used
### Stage A (main CL update)
- EWC: replay-informed Fisher + EWC penalty.
- Replay: mixed old/new batches.
- Hybrid: replay batches + light EWC penalty.

### Stage B (added in newer iterations)
- Head-only correction after Stage A.
- Freeze features (G1-G3), train classifier head only.
- Mixed old+new data.
- No EWC penalty in Stage B.
- Goal: recalibrate class boundaries after expansion.

## 7) Full Training Methods + Hyperparameters (Finalized)

### 7.1 EWC (`ewc_head_expand`)
Cycle 1:
- batch_size: `32`
- old_batch_size/new_batch_size: `16/16`
- max_epochs/min_epochs/patience: `15/5/5`
- old_f1_drop_tolerance: `0.05`
- unfreeze_g2: `true`
- lr_g2/lr_g3/lr_head: `5e-5 / 2e-4 / 5e-4`
- lr_head_old/lr_head_new: `1e-4 / 1e-3`
- weight_decay: `1e-4`
- EWC lambda: `700`
- fisher_max_batches: `150`
- Stage-B head correction:
  - enabled: `true`
  - epochs/patience: `max 8, min 3, patience 3`
  - correction batch split: `16/16`
  - correction lr_head: `2e-4`

Cycle 2:
- batch_size: `32`
- old_batch_size/new_batch_size: `16/16`
- max_epochs/min_epochs/patience: `12/5/5`
- old_f1_drop_tolerance: `0.05`
- unfreeze_g2: `true`
- lr_g2/lr_g3/lr_head: `1e-5 / 1e-4 / 2e-4`
- lr_head_old/lr_head_new: `5e-5 / 5e-4`
- weight_decay: `1e-4`
- EWC lambda: `700`
- fisher_max_batches: `150`
- Stage-B head correction:
  - enabled: `true`
  - epochs/patience: `max 8, min 3, patience 3`
  - correction batch split: `16/16`
  - correction lr_head: `1.5e-4`

### 7.2 Replay (`experience_replay_head_expand`)
Cycle 1:
- batch_size: `32`
- old_batch_size/new_batch_size: `16/16`
- max_epochs/min_epochs/patience: `15/5/5`
- old_f1_drop_tolerance: `0.05`
- unfreeze_g2: `true`
- lr_g2/lr_g3/lr_head: `5e-5 / 2e-4 / 5e-4`
- lr_head_old/lr_head_new: `1e-4 / 1e-3`
- weight_decay: `1e-4`
- Stage-B head correction:
  - enabled: `true`
  - epochs/patience: `max 8, min 3, patience 3`
  - correction batch split: `16/16`
  - correction lr_head: `2e-4`

Cycle 2:
- batch_size: `32`
- old_batch_size/new_batch_size: `16/16`
- max_epochs/min_epochs/patience: `12/5/5`
- old_f1_drop_tolerance: `0.05`
- unfreeze_g2: `true`
- lr_g2/lr_g3/lr_head: `1e-5 / 1e-4 / 2e-4`
- lr_head_old/lr_head_new: `8e-5 / 8e-4`
- weight_decay: `1e-4`
- Stage-B head correction:
  - enabled: `true`
  - epochs/patience: `max 8, min 3, patience 3`
  - correction batch split: `16/16`
  - correction lr_head: `1.5e-4`

### 7.3 Hybrid Replay+EWC (`hybrid_ewc_replay_head_expand`)
Cycle 1:
- batch_size: `32`
- old_batch_size/new_batch_size: `16/16`
- max_epochs/min_epochs/patience: `15/5/5`
- old_f1_drop_tolerance: `0.05`
- unfreeze_g2: `true`
- lr_g2/lr_g3/lr_head: `5e-5 / 2e-4 / 5e-4`
- lr_head_old/lr_head_new: `1e-4 / 1e-3`
- weight_decay: `1e-4`
- EWC lambda (light): `250`
- fisher_max_batches: `150`
- Stage-B head correction:
  - enabled: `true`
  - epochs/patience: `max 8, min 3, patience 3`
  - correction batch split: `16/16`
  - correction lr_head: `2e-4`

Cycle 2:
- batch_size: `32`
- old_batch_size/new_batch_size: `16/16`
- max_epochs/min_epochs/patience: `12/5/5`
- old_f1_drop_tolerance: `0.05`
- unfreeze_g2: `true`
- lr_g2/lr_g3/lr_head: `1e-5 / 1e-4 / 2e-4`
- lr_head_old/lr_head_new: `8e-5 / 8e-4`
- weight_decay: `1e-4`
- EWC lambda (light): `200`
- fisher_max_batches: `150`
- Stage-B head correction:
  - enabled: `true`
  - epochs/patience: `max 8, min 3, patience 3`
  - correction batch split: `16/16`
  - correction lr_head: `1.5e-4`

### 7.4 Naive Full Retrain Baseline (Reference)
Reference source: `catB_new_classes` baseline summary.

Cycle 1 (baseline reference):
- old_batch_size/new_batch_size: `16/16`
- lr_g2/lr_g3/lr_head: `5e-5 / 2e-4 / 5e-4`
- weight_decay: `1e-4`
- max_epochs/min_epochs/patience: `15/5/5`
- behavior: full retrain-style joint optimization on old+new stream without EWC/replay constraints.
- results:
  - Old F1: `0.6450`
  - New F1: `0.3228`
  - Old Acc: `0.8522`
  - New Acc: `0.9439`
  - Joint F1: `0.4839`

Cycle 2 (baseline target to beat):
- old_batch_size/new_batch_size: `16/16`
- lower cycle2 LR schedule (same pattern as CL pipeline style)
- weight_decay: `1e-4`
- max_epochs/min_epochs/patience: `12/5/5` (effective run setting family)
- results:
  - Old F1: `0.6539`
  - New F1: `0.3207`
  - Old Acc: `0.8707`
  - New Acc: `0.9342`
  - Joint F1: `0.4873`

## 8) Method Logic Summary
- EWC:
  - Compute Fisher on pre-expansion 5-class model.
  - Expand head 5->7.
  - Train with CE + EWC penalty (old classifier rows protected).
  - Run head-only correction stage.
- Replay:
  - Expand head 5->7.
  - Train with mixed old(replay+old stream)+new stream batches.
  - Use split-LR behavior via head-old/head-new settings.
  - Run head-only correction stage.
- Hybrid:
  - Replay-style mixed batches + light EWC penalty in same Stage A.
  - Run head-only correction stage.

## 9) Iteration Summary (new_01, new_02, new_03)
- `new_01`: solid improvement baseline for EWC/Replay tuning.
- `new_02`: tested stronger Fisher sampling idea; did not beat new_01 consistently.
- `new_03`: added Stage-B head correction + Hybrid method.

## 10) Current Best CL Results (from finalized new-series)
- Replay Cycle 2:
  - Old F1: `0.6424`
  - New F1: `0.3193`
  - Joint F1: `0.4809`
- Hybrid Cycle 2:
  - Joint F1: `0.4803`
- EWC Cycle 2:
  - Joint F1: `0.4800`

## 11) Current Conclusion
- CL methods improved significantly and are close.
- But baseline naive full retrain (`0.4873`) is still higher than best CL (`0.4809`).
- So the edge-optimized continual retraining journal goal is progressing, but Case 1 Cat B has not yet surpassed naive full retrain.
