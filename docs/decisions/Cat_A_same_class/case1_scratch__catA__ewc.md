# Part 1 - Case 1 (Scratch) - CatA EWC Summary

Generated: 2026-05-10 16:49:40

## Base Training Reference
- Base checkpoint: `experiments/part1_case1_scratch/outputs/case1_scratch_weighted/checkpoints/model.pth`
- Base test accuracy: **0.9745**
- Base test macro-F1: **0.9712**

## Cycle Performance
| Stage | Test Acc | Test Macro-F1 | Delta Acc vs Base | Delta F1 vs Base |
|---|---:|---:|---:|---:|
| Cycle 1 | 0.9780 | 0.9755 | 0.0035 | 0.0043 |
| Cycle 2 | 0.9736 | 0.9705 | -0.0009 | -0.0006 |

## Hyperparameters
### Cycle 1
- Batch size: `32`
- Max epochs: `15`
- Min epochs: `5`
- Patience: `5`
- LR G3: `0.0001`
- LR Head: `0.0005`
- EWC lambda: `1000.0`
- Fisher max batches: `100`
### Cycle 2
- Batch size: `32`
- Max epochs: `12`
- Min epochs: `5`
- Patience: `5`
- LR G3: `5e-05`
- LR Head: `0.0002`
- EWC lambda: `1000.0`
- Fisher max batches: `100`

## Resource and Runtime
| Stage | Train Wall Time (s) | Total Wall Time (s) | Peak VRAM (MB) | RAM End (MB) | Model Size (MB) |
|---|---:|---:|---:|---:|---:|
| Cycle 1 | 127.8198 | 138.4872 | 169.5100 | 1293.58 | 5.8097 |
| Cycle 2 | 173.9541 | 184.3347 | 169.5100 | 1298.66 | 5.8097 |

## Artifact Checklist
- Fisher matrix saved: `Yes`
- Theta star saved: `Yes`