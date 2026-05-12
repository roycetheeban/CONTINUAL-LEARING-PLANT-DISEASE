# CatB Case1 Summary

Generated: 2026-05-12 00:34:25

Base path: $base";
# CatB Case1 Summary  Generated: 2026-05-12 00:34:25  += ";
# CatB Case1 Summary  Generated: 2026-05-12 00:34:25  += 

| Method | Cycle | Old F1 | New F1 | Old Acc | New Acc | Forgetting Score | Peak VRAM MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| cat_b_ewc_head_expand | cycle1 | 0.6263 | 0.2724 | 0.8074 | 0.9246 | 0.3323 | 335.29 |
| cat_b_ewc_head_expand | cycle2 | 0.6375 | 0.3163 | 0.8408 | 0.9149 | -0.0295 | 335.29 |
| cat_b_replay_head_expand | cycle1 | 0.6257 | 0.3195 | 0.8039 | 0.9342 | NA | 325.39 |
| cat_b_replay_head_expand | cycle2 | 0.6408 | 0.3158 | 0.8470 | 0.9091 | NA | 325.39 |
| cat_b_naive_head_expand | cycle1 | 0.0000 | 0.9826 | 0.0000 | 0.9826 | 0.9587 | 325.71 |
| cat_b_naive_head_expand | cycle2 | 0.0000 | 0.9884 | 0.0000 | 0.9884 | 0.0000 | 325.71 |
| cat_b_isolation_new_branch | cycle1 | 0.8085 | 0.0563 | 0.9719 | 0.1238 | NA | 218.68 |
| cat_b_isolation_new_branch | cycle2 | 0.9712 | 0.0123 | 0.9745 | 0.0232 | NA | 218.68 |
| cat_b_naive_full_retrain_head_expand | cycle1 | 0.6450 | 0.3228 | 0.8522 | 0.9439 | NA | 325.39 |
| cat_b_naive_full_retrain_head_expand | cycle2 | 0.6539 | 0.3207 | 0.8707 | 0.9342 | NA | 325.39 |

## Artifact Check

| Method | Cycle | metrics.json | train_log.csv | checkpoint | fisher_matrix.pt | theta_star.pt | replay_manifest |
|---|---:|---|---|---|---|---|---|
| ewc_head_expand | cycle1 | True | True | True | True | True | False |
| ewc_head_expand | cycle2 | True | True | True | True | True | False |
| experience_replay_head_expand | cycle1 | True | True | True | False | False | True |
| experience_replay_head_expand | cycle2 | True | True | True | False | False | True |
| naive_finetune | cycle1 | True | True | True | False | False | False |
| naive_finetune | cycle2 | True | True | True | False | False | False |
| parameter_isolation_new_branch | cycle1 | True | True | True | False | False | False |
| parameter_isolation_new_branch | cycle2 | True | True | True | False | False | False |
| naive_full_retrain_head_expand | cycle1 | True | True | True | False | False | False |
| naive_full_retrain_head_expand | cycle2 | True | True | True | False | False | False |

## Notes

- NA means output/artifact was not found for that method-cycle.
- Forgetting Score appears only for methods/scripts that export it.
