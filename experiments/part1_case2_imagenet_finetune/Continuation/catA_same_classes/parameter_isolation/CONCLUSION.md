# Cat A Parameter Isolation Conclusion (M2 Tuned Start)

## Scope
- Method: Parameter Isolation (Adapter-based)
- Task: Cat A (same 5 classes, new data stream)
- Runs:
  - Cycle 1: `outputs/catA_isolation_cycle1`
  - Cycle 2: `outputs/catA_isolation_cycle2`

## Method Summary
1. The original model (backbone + classifier) is frozen to protect old knowledge.
2. A lightweight adapter is inserted at the G3 boundary (`features[11] -> adapter -> features[12]`).
3. Cycle 1 trains only Adapter C1.
4. Cycle 2 stacks Adapter C2 and trains only C2.
5. Old model parameters remain unchanged across both cycles.

## Layer/Training Policy Used
- Old model params: frozen (`requires_grad=False`)
- Classifier head: frozen
- Trainable params:
  - Cycle 1: adapter C1 only, `lr=1e-3`
  - Cycle 2: adapter C2 only, `lr=5e-4`
- Adapter init: identity-style (zero residual at start via zero-initialized `up` projection)

## Output Artifacts (Per Cycle)
- `checkpoints/model_expanded.pth`
- `logs/train_log.csv`
- `metrics/metrics.json`
- `metrics/classification_report.csv`
- `metrics/frozen_mask.pkl`
- `metrics/model_size_mb.txt`
- `figures/train_val_curves.png`
- `figures/confusion_matrix.png`

## Results Summary

| Cycle | Test Accuracy | Test Macro-F1 | Val Macro-F1 | Best Epoch |
|---|---:|---:|---:|---:|
| Cycle 1 | 0.9719 | 0.9686 | 0.9443 | 1 |
| Cycle 2 | 0.9727 | 0.9697 | 0.9390 | 12 |

## Resource Summary

| Cycle | Total Time (s) | Train Time (s) | Peak VRAM (MB) | End RAM (MB) | Model Size (MB) |
|---|---:|---:|---:|---:|---:|
| Cycle 1 | 101.3614 | 92.7013 | 122.63 | 1228.20 | 5.8277 |
| Cycle 2 | 184.8460 | 178.1234 | 122.14 | 1232.38 | 5.8456 |

## Key Interpretation
1. Isolation minimizes forgetting risk by never modifying old parameters.
2. Adaptation happens through newly added adapter capacity.
3. The expected tradeoff is model growth over cycles due to added adapters.
4. This method prioritizes stability/retention over compactness and backward transfer.

## Final Conclusion from This Run
1. **Performance remained stable across cycles** with a slight improvement in test metrics:
   - Accuracy: `0.9719 -> 0.9727` (`+0.0009`)
   - Macro-F1: `0.9686 -> 0.9697` (`+0.0010`)
2. **No forgetting spike was observed** in this Cat A same-class stream, consistent with isolation behavior.
3. **Model size increased** from `5.8277 MB` to `5.8456 MB` due to adapter stacking (`num_adapters: 1 -> 2`).
4. **Compute cost increased in Cycle 2** (longer train time), while memory remained low and stable (peak VRAM ~122 MB).
5. For Cat A same-class CL, this isolation setup provides strong retention with a modest size-growth tradeoff.

## Notes for Final Report
- Report Cycle 1 -> Cycle 2 performance trend on fixed `test_old`.
- Report model size growth explicitly (required for edge comparison).
- Compare directly against EWC and Replay under the same test protocol.
