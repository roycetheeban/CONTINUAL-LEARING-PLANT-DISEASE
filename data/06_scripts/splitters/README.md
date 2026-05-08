# Splitter Scripts

Use case: scripts for class filtering, split creation, and replay-buffer preparation.

Scripts:
- `_split_utils.py`
- `split_pretrain_26.py`
- `split_tomato_5cls.py`
- `split_tomato_new_2cls.py`
- `organize_unused_classes.py`
- `build_replay_buffer.py`
- `run_all_splits.py`

Run all splits (default copy mode):

```bash
python data/06_scripts/splitters/run_all_splits.py --seed 42
```

Optional destructive mode:

```bash
python data/06_scripts/splitters/run_all_splits.py --mode move --seed 42
```
