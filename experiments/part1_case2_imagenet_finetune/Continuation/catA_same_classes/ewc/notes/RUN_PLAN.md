# Cat A (Same Classes) - EWC Lambda Robustness Sweep (M2 tuned start)

## Cycle 1 (base checkpoint = M2 tuned)
python experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/ewc/code/train_catA_ewc.py --config experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/ewc/configs/catA_ewc_cycle1_lambda300.yaml
python experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/ewc/code/train_catA_ewc.py --config experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/ewc/configs/catA_ewc_cycle1_lambda1000.yaml
python experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/ewc/code/train_catA_ewc.py --config experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/ewc/configs/catA_ewc_cycle1_lambda3000.yaml
python experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/ewc/code/train_catA_ewc.py --config experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/ewc/configs/catA_ewc_cycle1_lambda5000.yaml
python experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/ewc/code/train_catA_ewc.py --config experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/ewc/configs/catA_ewc_cycle1_lambda8000.yaml
python experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/ewc/code/train_catA_ewc.py --config experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/ewc/configs/catA_ewc_cycle1_lambda10000.yaml
python experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/ewc/code/train_catA_ewc.py --config experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/ewc/configs/catA_ewc_cycle1_lambda15000.yaml

## Cycle 2 (each lambda continues from its own cycle1 result)
python experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/ewc/code/train_catA_ewc.py --config experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/ewc/configs/catA_ewc_cycle2_lambda300.yaml
python experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/ewc/code/train_catA_ewc.py --config experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/ewc/configs/catA_ewc_cycle2_lambda1000.yaml
python experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/ewc/code/train_catA_ewc.py --config experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/ewc/configs/catA_ewc_cycle2_lambda3000.yaml
python experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/ewc/code/train_catA_ewc.py --config experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/ewc/configs/catA_ewc_cycle2_lambda5000.yaml
python experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/ewc/code/train_catA_ewc.py --config experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/ewc/configs/catA_ewc_cycle2_lambda8000.yaml
python experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/ewc/code/train_catA_ewc.py --config experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/ewc/configs/catA_ewc_cycle2_lambda10000.yaml
python experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/ewc/code/train_catA_ewc.py --config experiments/part1_case2_imagenet_finetune/Continuation/catA_same_classes/ewc/configs/catA_ewc_cycle2_lambda15000.yaml

## Output roots (auto-created)
- outputs/cycle1_lambda300
- outputs/cycle1_lambda1000
- outputs/cycle1_lambda3000
- outputs/cycle1_lambda5000
- outputs/cycle1_lambda8000
- outputs/cycle1_lambda10000
- outputs/cycle1_lambda15000
- outputs/cycle2_lambda300
- outputs/cycle2_lambda1000
- outputs/cycle2_lambda3000
- outputs/cycle2_lambda5000
- outputs/cycle2_lambda8000
- outputs/cycle2_lambda10000
- outputs/cycle2_lambda15000

## Stored per run
- checkpoints/model.pth
- checkpoints/theta_star.pt
- metrics/fisher_matrix.pt
- metrics/metrics.json
- logs/train_log.csv
