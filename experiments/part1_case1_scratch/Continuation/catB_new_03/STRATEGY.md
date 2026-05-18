# Cat B New_03 Strategy

## Objective
Beat `naive_full_retrain` on Cycle 2 by improving old/new balance for:
- `EWC`
- `Replay`
- `Hybrid (Replay + light EWC)`

## Major Changes
1. Two-stage training per cycle for EWC and Replay:
- Stage A: method update (EWC or Replay).
- Stage B: head-only correction (classifier calibration, no EWC penalty).

2. New Hybrid method:
- Replay-style mixed old/new training batches
- Light EWC penalty (`lambda` lower than pure EWC)
- Head-only correction stage

3. Unified settings:
- old:new batch mix `16:16`
- conservative cycle2 LRs
- early stopping by joint val behavior

## Run Commands
```powershell
cd G:\AIOT_PHASE_03\CONTINUAL-LEARING-PLANT-DESEASE

# EWC
python experiments/part1_case1_scratch/Continuation/catB_new_03/ewc_head_expand/code/train_catB_ewc.py --config experiments/part1_case1_scratch/Continuation/catB_new_03/ewc_head_expand/configs/catB_ewc_cycle1.yaml
python experiments/part1_case1_scratch/Continuation/catB_new_03/ewc_head_expand/code/train_catB_ewc.py --config experiments/part1_case1_scratch/Continuation/catB_new_03/ewc_head_expand/configs/catB_ewc_cycle2.yaml

# Replay
python experiments/part1_case1_scratch/Continuation/catB_new_03/experience_replay_head_expand/code/train_catB_replay.py --config experiments/part1_case1_scratch/Continuation/catB_new_03/experience_replay_head_expand/configs/catB_replay_cycle1.yaml
python experiments/part1_case1_scratch/Continuation/catB_new_03/experience_replay_head_expand/code/train_catB_replay.py --config experiments/part1_case1_scratch/Continuation/catB_new_03/experience_replay_head_expand/configs/catB_replay_cycle2.yaml

# Hybrid
python experiments/part1_case1_scratch/Continuation/catB_new_03/hybrid_ewc_replay_head_expand/code/train_catB_hybrid.py --config experiments/part1_case1_scratch/Continuation/catB_new_03/hybrid_ewc_replay_head_expand/configs/catB_hybrid_cycle1.yaml
python experiments/part1_case1_scratch/Continuation/catB_new_03/hybrid_ewc_replay_head_expand/code/train_catB_hybrid.py --config experiments/part1_case1_scratch/Continuation/catB_new_03/hybrid_ewc_replay_head_expand/configs/catB_hybrid_cycle2.yaml
```

## Success Rule
Cycle2 target:
- Old F1 > `0.6539`
- New F1 >= `0.3207`
- Joint F1 > `0.4873`
