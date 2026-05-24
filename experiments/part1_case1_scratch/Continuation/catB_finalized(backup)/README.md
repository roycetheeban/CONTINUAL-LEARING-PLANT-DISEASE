# Cat B Finalized (Case 1)

This folder contains finalized Cat B methods selected for transfer and further optimization:

- `ewc_head_expand`
- `experience_replay_head_expand`
- `_shared` (common utilities)

## Source
Copied from:
- `Continuation/catB_new_classes tuned`

## Why These Methods
- They are the strongest CL candidates for balanced old/new performance.
- Goal is to beat `naive_full_retrain` baseline with CL constraints.

## Run
Use:

```bash
python run_catB_finalized.py ewc 1
python run_catB_finalized.py ewc 2
python run_catB_finalized.py replay 1
python run_catB_finalized.py replay 2
```

## Notes
- This folder intentionally excludes naive and isolation methods.
- Only Cat B cycle configs are tuned; Part 1 base model is unchanged.
