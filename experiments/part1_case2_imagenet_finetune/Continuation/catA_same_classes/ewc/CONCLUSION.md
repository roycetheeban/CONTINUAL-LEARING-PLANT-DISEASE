# Cat A EWC Conclusion (Cycle 1 Lambda Sweep)

## Scope
This conclusion is based on the completed **Cycle 1** EWC lambda sweep for:
- Model: MobileNetV3-Small (M2 tuned start)
- Task: Cat A (same 5 classes, new data stream)
- Dataset split: `cl_cycle1_stream` train, `val`, `test`

## Cycle 1 Results Summary
Ranked by `test_macro_f1`:

| Lambda | Test Accuracy | Test Macro-F1 |
|---:|---:|---:|
| 10000 | 0.9798 | 0.9779 |
| 15000 | 0.9789 | 0.9773 |
| 8000 | 0.9789 | 0.9769 |
| 3000 | 0.9780 | 0.9756 |
| 1000 | 0.9771 | 0.9740 |
| 5000 | 0.9763 | 0.9736 |
| 300 | 0.9763 | 0.9731 |

## Selected Lambda for Continuation
**Chosen for Cycle 2 seed point:** `lambda = 10000`

Reason:
1. It gives the best Cycle 1 `test_macro_f1`.
2. It also gives the best Cycle 1 `test_accuracy`.
3. The improvement is consistent against all previously tested lambdas.

## Research Interpretation
- In this run setup, stronger regularization up to `10000` improved retention/performance in Cycle 1.
- This does **not** guarantee the same optimum in Cycle 2, because optimization dynamics change after the first CL update.

## Cycle 2 Plan
Run Cycle 2 with the same expanded lambda sweep for fair comparison:
- `300, 1000, 3000, 5000, 8000, 10000, 15000`

Each Cycle 2 run should initialize from its matching Cycle 1 checkpoint:
- `cycle2_lambdaX` starts from `outputs/cycle1_lambdaX/checkpoints/model.pth`

This preserves methodological consistency and allows direct Cycle 1 -> Cycle 2 lambda behavior analysis.

## Final Cross-Cycle Conclusion (After Cycle 2 Sweep)

Cycle 2 was executed with the full lambda set:
- `300, 1000, 3000, 5000, 8000, 10000, 15000`

Best Cycle 2 result by `test_macro_f1`:
- **lambda = 3000**
- Test accuracy: **0.9824**
- Test macro-F1: **0.9807**

Second-best (very close):
- lambda = 10000
- Test accuracy: 0.9824
- Test macro-F1: 0.9807 (slightly lower than lambda 3000)

## Practical Research Takeaways

1. **Best lambda is cycle-dependent** in this experiment, not globally fixed.
   - Cycle 1 best: `10000`
   - Cycle 2 best: `3000`

2. **Interpretation:** early continual updates benefited from stronger regularization, while the later cycle favored a moderately lower penalty for a better stability-plasticity balance.

3. **Resource-aware note:** some higher lambdas in Cycle 2 increased runtime without improving top test performance, so final lambda choice should consider both accuracy/F1 and update cost.

## Recommended Setting from This Run

- If selecting a single best checkpoint after full Cat A EWC continuation: use the **Cycle 2, lambda 3000** result.
- For future CL runs, keep per-cycle lambda retuning enabled instead of assuming one lambda is optimal for all cycles.

## Resource Usage Summary (From Stored Outputs)

### Resource Pattern Observed
- `peak_vram_mb` was effectively constant across runs: **169.51 MB**
- `model_size_mb` was constant across runs: **5.8097 MB**
- `fisher_batches_used` was constant across runs: **17**
- Main resource variation came from **runtime** and **best epoch** behavior.

### Cycle 1 Resource Snapshot
- Typical `total_wall_time_sec`: about **84.7 to 88.7 sec**
- Typical `train_wall_time_sec`: about **72.4 to 76.7 sec**
- Best Cycle 1 lambda (`10000`) runtime:
  - total: **85.19 sec**
  - train: **74.43 sec**

### Cycle 2 Resource Snapshot
- Most runs (300/1000/3000/5000/10000):
  - `total_wall_time_sec`: about **94.97 to 98.01 sec**
  - `train_wall_time_sec`: about **84.40 to 87.27 sec**
  - `best_epoch`: **1**
- High-lambda slower runs (8000/15000):
  - `cycle2_lambda8000`: total **165.04 sec**, train **154.55 sec**, best_epoch **6**
  - `cycle2_lambda15000`: total **167.75 sec**, train **156.55 sec**, best_epoch **6**
  - These were slower but not top performers.

## Insight Summary (All Available Outputs)

1. **Cycle-dependent optimum is confirmed**
   - Cycle 1 best: `lambda=10000`
   - Cycle 2 best: `lambda=3000`
   - A single fixed lambda is not optimal across both cycles.

2. **Performance gain is clear vs lower lambdas**
   - In Cycle 1, higher lambda improved test macro-F1 up to 10000.
   - In Cycle 2, mid-high lambda (`3000`) gave best macro-F1, while very high lambda did not improve further.

3. **Efficiency-performance tradeoff appears at high Cycle 2 lambdas**
   - `8000` and `15000` required much longer training time.
   - Despite extra time, they did not beat `3000` on test macro-F1.

4. **Practical operating point from this full sweep**
   - For final reported Cat A EWC checkpoint: **Cycle 2, lambda 3000**.
   - For future experiments: keep cycle-wise retuning and include runtime in model selection criteria, not accuracy/F1 alone.
