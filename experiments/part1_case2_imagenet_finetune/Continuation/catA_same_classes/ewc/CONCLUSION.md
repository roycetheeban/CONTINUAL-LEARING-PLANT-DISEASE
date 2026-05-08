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
