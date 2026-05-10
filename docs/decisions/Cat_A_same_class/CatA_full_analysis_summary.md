# Category A Same-Class Continual Learning: Full Analysis Summary

## Case-Wise Cycle2 Outcomes
| Case | Method | Base Acc | Base F1 | Cycle2 Acc | Cycle2 F1 | Delta Acc | Delta F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Case1 Scratch (M1) | EWC | 0.9745 | 0.9712 | 0.9736 | 0.9705 | -0.0009 | -0.0006 |
| Case1 Scratch (M1) | Experience Replay | 0.9745 | 0.9712 | 0.9719 | 0.9666 | -0.0026 | -0.0045 |
| Case1 Scratch (M1) | Naive FT | 0.9745 | 0.9712 | 0.9754 | 0.9713 | +0.0009 | +0.0001 |
| Case1 Scratch (M1) | Parameter Isolation | 0.9745 | 0.9712 | 0.9657 | 0.9596 | -0.0088 | -0.0116 |
| Case2 ImageNet FT (M2) | EWC | 0.9727 | 0.9701 | 0.9815 | 0.9798 | +0.0088 | +0.0096 |
| Case2 ImageNet FT (M2) | Experience Replay | 0.9727 | 0.9701 | 0.9842 | 0.9827 | +0.0114 | +0.0125 |
| Case2 ImageNet FT (M2) | Naive FT | 0.9727 | 0.9701 | 0.9921 | 0.9908 | +0.0193 | +0.0207 |
| Case2 ImageNet FT (M2) | Parameter Isolation | 0.9727 | 0.9701 | 0.9727 | 0.9697 | +0.0000 | -0.0005 |
| Case3B PlantVillage FT (M3) | EWC | 0.9683 | 0.9652 | 0.9666 | 0.9635 | -0.0018 | -0.0017 |
| Case3B PlantVillage FT (M3) | Experience Replay | 0.9683 | 0.9652 | 0.9850 | 0.9835 | +0.0167 | +0.0184 |
| Case3B PlantVillage FT (M3) | Naive FT | 0.9683 | 0.9652 | 0.9824 | 0.9798 | +0.0141 | +0.0147 |
| Case3B PlantVillage FT (M3) | Parameter Isolation | 0.9683 | 0.9652 | 0.9666 | 0.9633 | -0.0018 | -0.0018 |

## Method Aggregate (Mean Across 3 Base Cases)
| Method | Mean C2 Acc | Mean C2 F1 | Mean Delta Acc | Mean Delta F1 | Mean Runtime(s) | Mean C2 Size(MB) |
|---|---:|---:|---:|---:|---:|---:|
| EWC | 0.9739 | 0.9713 | +0.0021 | +0.0024 | 238.46 | 5.8097 |
| Experience Replay | 0.9804 | 0.9776 | +0.0085 | +0.0088 | 564.03 | 5.8097 |
| Naive FT | 0.9833 | 0.9806 | +0.0114 | +0.0118 | 286.42 | 5.8097 |
| Parameter Isolation | 0.9683 | 0.9642 | -0.0035 | -0.0046 | 233.13 | 5.8456 |

## Deep Analysis and Recommendations
- Experience Replay is the safest default for same-class CL when you need robust retention with stable adaptation.
- EWC is strong and often close to replay, but it needs lambda tuning for best performance under drift changes.
- Naive FT remains essential as lower-bound evidence and for fast experimentation, but should not be default in production CL.
- Parameter Isolation is best when retention safety is strict, but you should account for model growth and reduced plasticity.

## Use-Case Guidance
- Edge device with tight memory but acceptable tuning effort: EWC.
- Non-stationary field stream and highest practical robustness: Experience Replay.
- Safety-critical old-class retention priority: Parameter Isolation.
- Quick baseline/ablation only: Naive FT.

## Conclusion
For CatA same-class continual learning across M1/M2/M3, replay and tuned EWC give the best stability-plasticity balance. Naive FT is the forgetting baseline, while isolation is the retention-maximizing option when model growth is acceptable. This should be the decision basis for journal reporting and edge deployment method selection.