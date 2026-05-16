# CatB Case1 Tuned Summary

Generated: 2026-05-12 22:14:22

Base path: G:\AIOT_PHASE_03\CONTINUAL-LEARING-PLANT-DESEASE\experiments\part1_case1_scratch\Continuation\catB_new_classes tuned

| Method | Cycle | Old F1 | New F1 | Old Acc | New Acc | Forgetting Score | Peak VRAM MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| cat_b_ewc_head_expand | cycle1 | 0.5981 | 0.3804 | 0.7722 | 0.9168 | 0.3605 | 335.29 |
| cat_b_ewc_head_expand | cycle2 | 0.6259 | 0.3202 | 0.8179 | 0.9400 | -0.0317 | 335.29 |
| cat_b_replay_head_expand | cycle1 | 0.6227 | 0.3238 | 0.8135 | 0.9555 | NA | 325.39 |
| cat_b_replay_head_expand | cycle2 | 0.6394 | 0.3229 | 0.8443 | 0.9478 | NA | 325.39 |
| cat_b_naive_head_expand | cycle1 | NA | NA | NA | NA | NA | NA |
| cat_b_naive_head_expand | cycle2 | NA | NA | NA | NA | NA | NA |
| cat_b_isolation_new_branch | cycle1 | 0.8087 | 0.0363 | 0.9719 | 0.0716 | NA | 218.68 |
| cat_b_isolation_new_branch | cycle2 | 0.8085 | 0.0218 | 0.9719 | 0.0426 | NA | 218.68 |
| cat_b_naive_full_retrain_head_expand | cycle1 | NA | NA | NA | NA | NA | NA |
| cat_b_naive_full_retrain_head_expand | cycle2 | NA | NA | NA | NA | NA | NA |

## Artifact Check

| Method | Cycle | metrics.json | train_log.csv | checkpoint | fisher_matrix.pt | theta_star.pt | replay_manifest |
|---|---:|---|---|---|---|---|---|
| ewc_head_expand | cycle1 | True | True | True | True | True | False |
| ewc_head_expand | cycle2 | True | True | True | True | True | False |
| experience_replay_head_expand | cycle1 | True | True | True | False | False | True |
| experience_replay_head_expand | cycle2 | True | True | True | False | False | True |
| naive_finetune | cycle1 | False | False | False | False | False | False |
| naive_finetune | cycle2 | False | False | False | False | False | False |
| parameter_isolation_new_branch | cycle1 | True | True | True | False | False | False |
| parameter_isolation_new_branch | cycle2 | True | True | True | False | False | False |
| naive_full_retrain_head_expand | cycle1 | False | False | False | False | False | False |
| naive_full_retrain_head_expand | cycle2 | False | False | False | False | False | False |

## Notes

- NA means output/artifact was not found for that method-cycle.
- Forgetting Score appears only for methods/scripts that export it.
