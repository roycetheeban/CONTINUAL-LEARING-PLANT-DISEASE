import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

import sys
sys.path.append(str(Path(__file__).resolve().parents[2] / '_shared' / 'code'))
from common_catb import set_seed, ensure_dirs, evaluate, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))

    set_seed(int(cfg['seed']))
    device = torch.device('cuda' if cfg['device']=='cuda' and torch.cuda.is_available() else 'cpu')
    dirs = ensure_dirs(Path(cfg['output']['root']))

    tfm = transforms.Compose([
        transforms.Resize((int(cfg['data']['image_size']), int(cfg['data']['image_size']))),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    old_test = datasets.ImageFolder(cfg['data']['old_test_dir'], transform=tfm)
    new_test = datasets.ImageFolder(cfg['data']['new_test_dir'], transform=tfm)

    bs = int(cfg['train']['batch_size']); nw = int(cfg['num_workers'])
    old_loader = DataLoader(old_test, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    new_loader = DataLoader(new_test, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)

    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    model.classifier[3] = nn.Linear(1024, int(cfg['model']['old_num_classes']))
    model.load_state_dict(torch.load(cfg['model']['base_checkpoint'], map_location=device), strict=True)
    model = model.to(device)

    yot,yop = evaluate(model, old_loader, device)
    ys_old = score(yot,yop)

    # evaluate new test with label remap into 0..4 impossible; use raw predictions over old classes = near-zero expected
    ynt,ynp = evaluate(model, new_loader, device)
    ys_new = score(ynt,ynp)

    with (dirs['metrics'] / 'b0_old_class_metrics.json').open('w', encoding='utf-8') as f:
        json.dump(ys_old, f, indent=2)
    with (dirs['metrics'] / 'b0_new_class_metrics.json').open('w', encoding='utf-8') as f:
        json.dump(ys_new, f, indent=2)


if __name__ == '__main__':
    main()
