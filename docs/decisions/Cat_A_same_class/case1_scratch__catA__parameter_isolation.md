# Part 1 - Case 1 (Scratch) - CatA Parameter Isolation Summary

Generated: 2026-05-10 16:49:40

## Base Training Reference
- Base checkpoint: `experiments/part1_case1_scratch/outputs/case1_scratch_weighted/checkpoints/model.pth`
- Base test accuracy: **0.9745**
- Base test macro-F1: **0.9712**

## Cycle Performance
| Stage | Test Acc | Test Macro-F1 | Delta Acc vs Base | Delta F1 vs Base |
|---|---:|---:|---:|---:|
| Cycle 1 | 0.9631 | 0.9563 | -0.0114 | -0.0149 |
| Cycle 2 | 0.9657 | 0.9596 | -0.0088 | -0.0116 |

## Hyperparameters
### Cycle 1
- Batch size: `32`
- Max epochs: `15`
- Min epochs: `5`
- Patience: `5`
- LR Adapter: `0.001`
### Cycle 2
- Batch size: `32`
- Max epochs: `12`
- Min epochs: `5`
- Patience: `5`
- LR Adapter: `0.0005`

## Resource and Runtime
| Stage | Train Wall Time (s) | Total Wall Time (s) | Peak VRAM (MB) | RAM End (MB) | Model Size (MB) |
|---|---:|---:|---:|---:|---:|
| Cycle 1 | 88.7860 | 95.9211 | 122.6300 | 1213.07 | 5.8277 |
| Cycle 2 | 78.6674 | 85.0337 | 122.1400 | 1225.1 | 5.8456 |

## Artifact Checklist
- Frozen mask saved: `Yes`
- Model size log saved: `Yes`