# 03 Tomato New 2 Classes

Use case: new-class introduction stream for Category-B continual learning.

Subfolders:
- `cl_cycle1_stream/`
- `cl_cycle2_stream/`
- `val/`
- `test/`

Split policy (per class):
- `test/`: 15%
- `val/`: 10%
- `cl_cycle1_stream/`: 37.5%
- `cl_cycle2_stream/`: 37.5%

Notes:
- No `initial_train/` in this folder.
- Files are copied from `raw_segmented` by default.
