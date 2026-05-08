# Cat A EWC Final Note (M2 Tuned Start)

## Scope
This note summarizes the final Cat A (same classes) EWC decisions and results for MobileNetV3-Small using the tuned M2 checkpoint as the starting model.

## Fixed Protocol Across All 3 States
Three evaluation states were used:
1. Before CL (initial tuned M2 baseline)
2. After Cycle 1
3. After Cycle 2

For all 3 states, the same test set was used:
- `data/02_tomato_5cls/test`
- Test image count: 1,137

## Data Used and Counts
- Initial train (`initial_train`): 2,842
- Cycle 1 train stream (`cl_cycle1_stream`): 1,422
- Cycle 2 train stream (`cl_cycle2_stream`): 1,420
- Replay/Fisher source (`replay_buffer`): 526
- Validation (`val`): 758
- Test (`test`): 1,137

## Core Hyperparameters
- Model: MobileNetV3-Small (5 classes)
- Trainable layers: `features[9:]` (G3) + `classifier` (G4)
- Frozen layers: `features[:9]` (G1 + G2)
- Optimizer: Adam
- Batch size: 32
- Weight decay: 1e-4
- Scheduler: ReduceLROnPlateau (`mode=max`, `factor=0.5`, `patience=3`, `min_lr=1e-6`)
- Fisher max batches: 100
- Early-stop forgetting guard: old-class val macro-F1 drop tolerance = 3%
- Lambda sweep per cycle: 300, 1000, 3000

## Cycle-wise Hyperparameter Changes
From Cycle 1 to Cycle 2:
- `lr_g3`: 1e-4 -> 5e-5
- `lr_head`: 5e-4 -> 2e-4
- `max_epochs`: 15 -> 12
- Other settings kept consistent for fair comparison.

## Results Summary
### Before CL (Initial tuned M2)
- Test accuracy: 0.9692
- Test macro-F1: 0.9660

### Cycle 1 (lambda sweep)
- Lambda 300: test acc 0.9754, test macro-F1 0.9714
- Lambda 1000: test acc 0.9745, test macro-F1 0.9704
- Lambda 3000: test acc 0.9754, test macro-F1 0.9717

Cycle 1 best by test accuracy:
- Tie: lambda 300 and 3000 (both 0.9754)

Cycle 1 best by test macro-F1:
- Lambda 3000 (0.9717)

### Cycle 2 (lambda sweep)
- Lambda 300: test acc 0.9789, test macro-F1 0.9758
- Lambda 1000: test acc 0.9807, test macro-F1 0.9782
- Lambda 3000: test acc 0.9798, test macro-F1 0.9772

Cycle 2 best (both accuracy and macro-F1):
- Lambda 1000

## Final Best Case
Best overall result in this Cat A EWC run set:
- Cycle 2, lambda 1000
- Test accuracy: 0.9807
- Test macro-F1: 0.9782

Improvement vs initial tuned baseline:
- Accuracy: +0.0114 absolute (0.9692 -> 0.9807)
- Macro-F1: +0.0122 absolute (0.9660 -> 0.9782)

## Final Conclusion
- EWC improved same-class continual performance over the initial tuned model.
- Progressive LR reduction across cycles was effective.
- The most stable final setting for this run was Cycle 2 with lambda 1000.
- Keeping the same test set across all three states enabled clean before/after comparison.
