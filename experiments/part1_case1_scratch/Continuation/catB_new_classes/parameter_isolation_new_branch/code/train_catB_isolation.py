import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import models

import sys
sys.path.append(str(Path(__file__).resolve().parents[2] / '_shared' / 'code'))
from common_catb import (
    set_seed, ensure_dirs, build_transforms, build_global_classes, RemapImageFolder,
    build_eval_loaders, evaluate, score, load_base_model, save_logs, plot_training,
    save_confusion, dump_metrics, get_process_ram_mb, get_model_size_mb
)


class NewClassBranch(nn.Module):
    def __init__(self, n_new=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.Hardswish(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.Hardswish(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.Hardswish(),
            nn.Conv2d(64, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.Hardswish(),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Linear(128, 64), nn.Hardswish(), nn.Dropout(0.2), nn.Linear(64, n_new))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.head(x)


def train_branch(branch, loader, val_loader, device, cfg):
    opt = Adam(branch.parameters(), lr=float(cfg['train']['lr_branch']), weight_decay=float(cfg['train']['weight_decay']))
    sch = ReduceLROnPlateau(opt, mode='max', patience=3, factor=0.5, min_lr=1e-6)
    ce = nn.CrossEntropyLoss()

    best = {'f1': -1, 'state': None, 'epoch': 0}
    bad = 0
    logs = []
    for ep in range(1, int(cfg['train']['max_epochs'])+1):
        t0 = time.perf_counter()
        branch.train()
        run = 0.0
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y_local = (y - int(cfg['labels']['new_start_idx'])).to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = branch(x)
            loss = ce(logits, y_local)
            loss.backward()
            opt.step()
            run += loss.item()
        train_loss = run / max(len(loader), 1)

        # val on new classes only
        branch.eval()
        y_t, y_p = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                p = torch.argmax(branch(x), dim=1).cpu()
                y_p.extend(p.tolist())
                y_t.extend((y - int(cfg['labels']['new_start_idx'])).tolist())
        from sklearn.metrics import f1_score
        f1 = float(f1_score(y_t, y_p, average='macro'))
        sch.step(f1)

        logs.append({'epoch': ep, 'epoch_time_sec': round(time.perf_counter()-t0,4), 'train_loss': train_loss, 'val_new_macro_f1': f1, 'lr': opt.param_groups[0]['lr']})
        if f1 > best['f1']:
            best = {'f1': f1, 'state': {k:v.cpu().clone() for k,v in branch.state_dict().items()}, 'epoch': ep}
            bad = 0
        else:
            bad += 1
        if ep >= int(cfg['train']['min_epochs']) and bad >= int(cfg['train']['patience']):
            break

    branch.load_state_dict(best['state'])
    return branch, logs, best


def eval_combined(old_model, branch, loader_old, loader_new, device, old_n, old_logit_scale=1.0, new_logit_scale=1.0):
    old_model.eval(); branch.eval()
    y_old_t, y_old_p = [], []
    y_new_t, y_new_p = [], []
    with torch.no_grad():
        for x, y in loader_old:
            x = x.to(device, non_blocking=True)
            old_logits = old_model(x) * float(old_logit_scale)
            new_logits = branch(x) * float(new_logit_scale)
            comb = torch.cat([old_logits, new_logits], dim=1)
            p = torch.argmax(comb, dim=1).cpu().tolist()
            y_old_p.extend(p)
            y_old_t.extend(y.tolist())
        for x, y in loader_new:
            x = x.to(device, non_blocking=True)
            old_logits = old_model(x) * float(old_logit_scale)
            new_logits = branch(x) * float(new_logit_scale)
            comb = torch.cat([old_logits, new_logits], dim=1)
            p = torch.argmax(comb, dim=1).cpu().tolist()
            y_new_p.extend(p)
            y_new_t.extend(y.tolist())
    return y_old_t, y_old_p, y_new_t, y_new_p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))

    set_seed(int(cfg['seed']))
    device = torch.device('cuda' if cfg['device']=='cuda' and torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    dirs = ensure_dirs(Path(cfg['output']['root']))
    total_start = time.perf_counter()

    global_classes, old_classes, new_classes = build_global_classes(cfg)
    cls_to_idx = {c:i for i,c in enumerate(global_classes)}
    train_tfms, eval_tfms = build_transforms(cfg)

    old_model = load_base_model(cfg['model']['base_checkpoint'], old_num_classes=len(old_classes), device=device)
    for p in old_model.parameters():
        p.requires_grad = False

    # new-only train
    new_train_ds = RemapImageFolder(cfg['data']['new_train_dir'], train_tfms, cls_to_idx)
    new_val_ds = RemapImageFolder(cfg['data']['new_val_dir'], eval_tfms, cls_to_idx)
    nw = int(cfg['num_workers'])
    train_loader = DataLoader(new_train_ds, batch_size=int(cfg['train']['batch_size']), shuffle=True, num_workers=nw, pin_memory=True)
    new_val_loader = DataLoader(new_val_ds, batch_size=int(cfg['train']['batch_size']), shuffle=False, num_workers=nw, pin_memory=True)

    old_test_loader, new_test_loader, _, _ = build_eval_loaders(cfg, global_classes, eval_tfms)

    branch = NewClassBranch(n_new=len(new_classes)).to(device)
    prev_branch_ckpt = cfg.get('model', {}).get('prev_branch_checkpoint', '')
    prev_loaded = False
    if prev_branch_ckpt:
        prev_path = Path(prev_branch_ckpt)
        if not prev_path.is_absolute():
            prev_path = Path.cwd() / prev_path
        if prev_path.exists():
            branch.load_state_dict(torch.load(prev_path, map_location=device))
            prev_loaded = True
            print(f"[ISOLATION] loaded previous branch checkpoint: {prev_path}", flush=True)
        elif str(cfg.get('meta', {}).get('cycle_name', '')).lower() == 'cycle2':
            raise FileNotFoundError(
                f"Cycle2 requires previous branch checkpoint, but not found: {prev_path}"
            )

    cfg['labels'] = {'new_start_idx': len(old_classes)}
    branch, logs, best = train_branch(branch, train_loader, new_val_loader, device, cfg)

    base_out = dirs['checkpoints'] / 'base_frozen.pth'
    branch_out = dirs['checkpoints'] / 'new_branch.pth'
    torch.save(old_model.state_dict(), base_out)
    torch.save(branch.state_dict(), branch_out)
    print(f"[ISOLATION] saved base checkpoint: {base_out}", flush=True)
    print(f"[ISOLATION] saved branch checkpoint: {branch_out}", flush=True)
    with (dirs['metrics'] / 'model_size_mb.txt').open('w', encoding='utf-8') as f:
        f.write(str(round(get_model_size_mb(old_model) + get_model_size_mb(branch), 4)))

    save_logs(logs, dirs['logs'] / 'train_log.csv')
    plot_training(logs, dirs['figures'] / 'train_val_curves.png', 'CatB Isolation New Branch Training')

    inf_cfg = cfg.get('inference', {})
    old_logit_scale = float(inf_cfg.get('old_logit_scale', 1.0))
    new_logit_scale = float(inf_cfg.get('new_logit_scale', 1.0))
    y_old_t, y_old_p, y_new_t, y_new_p = eval_combined(
        old_model, branch, old_test_loader, new_test_loader, device, len(old_classes),
        old_logit_scale=old_logit_scale, new_logit_scale=new_logit_scale
    )
    old_stats = score(y_old_t, y_old_p)
    new_stats = score(y_new_t, y_new_p)

    save_confusion(y_old_t, y_old_p, old_classes, dirs['figures'] / 'confusion_old.png', 'CatB Isolation Old Classes')
    save_confusion([v-len(old_classes) for v in y_new_t], [v-len(old_classes) for v in y_new_p], new_classes, dirs['figures'] / 'confusion_new.png', 'CatB Isolation New Classes')

    runtime = {
        'total_wall_time_sec': round(time.perf_counter()-total_start,4),
        'train_wall_time_sec': round(sum(r['epoch_time_sec'] for r in logs),4),
        'peak_vram_mb': round(torch.cuda.max_memory_allocated()/(1024**2),2) if device.type=='cuda' else None,
        'process_ram_mb_end': get_process_ram_mb(),
    }
    extra = {
        'best_epoch': best['epoch'],
        'prev_branch_loaded': prev_loaded,
        'old_logit_scale': old_logit_scale,
        'new_logit_scale': new_logit_scale,
    }

    # dummy shell model object just for dump size/params context
    class _Wrap(nn.Module):
        def __init__(self, oldm, br):
            super().__init__(); self.oldm=oldm; self.br=br
    wrap = _Wrap(old_model, branch)
    dump_metrics(dirs, cfg, 'cat_b_isolation_new_branch', cfg['meta']['cycle_name'], global_classes, old_classes, new_classes, old_stats, new_stats, runtime, wrap, extra=extra)


if __name__ == '__main__':
    main()
