# Two-Stage Cat B Update Rationale (new_02)

## Change Added
For both `EWC` and `Replay` in Cat B cycles:
1. Stage A: normal CL update (existing method logic).
2. Stage B: short head-only correction (classifier only, backbone frozen), mixed old+new data, no EWC penalty.

## Why This Should Help
- In class-incremental setups, much of the final error comes from output-layer boundary misalignment after head expansion.
- Stage A learns features + task update; Stage B recalibrates logits with stable features.
- This is consistent with continual-learning practice where replay/regularization stabilizes representation and a final classifier alignment step improves old/new balance.

## Practical Configuration
- `head_correction.enabled: true`
- `max_epochs: 8`, `min_epochs: 3`, `patience: 3`
- balanced batches (`old_batch_size: 16`, `new_batch_size: 16`)
- smaller LR than main stage (`lr_head: 2e-4` for cycle1, `1.5e-4` for cycle2)

## Files Updated
- `ewc_head_expand/code/train_catB_ewc.py`
- `experience_replay_head_expand/code/train_catB_replay.py`
- corresponding cycle1/cycle2 YAMLs under both methods.
