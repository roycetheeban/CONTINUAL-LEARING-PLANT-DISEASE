import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

import sys
sys.path.append(str(Path(__file__).resolve().parents[2] / '_shared' / 'code'))
from common_catb import (
    set_seed, ensure_dirs, build_transforms, build_global_classes, RemapImageFolder,
    build_eval_loaders, evaluate, score, load_base_model, expand_head_to_7, freeze_for_catb,
    train_epoch_mixed, save_logs, plot_training, save_confusion, dump_metrics, get_process_ram_mb
)


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

    old_train_ds = RemapImageFolder(cfg['data']['old_train_dir'], train_tfms, cls_to_idx)
    new_train_ds = RemapImageFolder(cfg['data']['new_train_dir'], train_tfms, cls_to_idx)

    bs_old = int(cfg['train']['old_batch_size'])
    bs_new = int(cfg['train']['new_batch_size'])
    nw = int(cfg['num_workers'])

    old_train_loader = DataLoader(old_train_ds, batch_size=bs_old, shuffle=True, num_workers=nw, pin_memory=True, drop_last=True)
    new_train_loader = DataLoader(new_train_ds, batch_size=bs_new, shuffle=True, num_workers=nw, pin_memory=True, drop_last=True)

    old_test_loader, new_test_loader, old_val_loader, new_val_loader = build_eval_loaders(cfg, global_classes, eval_tfms)

    model = load_base_model(cfg['model']['base_checkpoint'], old_num_classes=len(old_classes), device=device)
    expand_head_to_7(model, total_classes=len(global_classes))
    freeze_for_catb(model, unfreeze_g2=bool(cfg['train']['unfreeze_g2']))

    optimizer = Adam(
        [
            {'params': model.features[4:9].parameters(), 'lr': float(cfg['train']['lr_g2'])},
            {'params': model.features[9:].parameters(), 'lr': float(cfg['train']['lr_g3'])},
            {'params': model.classifier.parameters(), 'lr': float(cfg['train']['lr_head'])},
        ],
        weight_decay=float(cfg['train']['weight_decay'])
    )
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5, min_lr=1e-6)
    criterion = nn.CrossEntropyLoss()

    best = {'f1': -1, 'state': None, 'epoch': 0}
    bad = 0
    logs = []
    for ep in range(1, int(cfg['train']['max_epochs'])+1):
        t0 = time.perf_counter()
        tr_loss = train_epoch_mixed(model, old_train_loader, new_train_loader, device, criterion, optimizer)
        y_old_t, y_old_p = evaluate(model, old_val_loader, device)
        y_new_t, y_new_p = evaluate(model, new_val_loader, device)
        old_s = score(y_old_t, y_old_p)
        new_s = score(y_new_t, y_new_p)
        joint_f1 = 0.5 * (old_s['macro_f1'] + new_s['macro_f1'])
        scheduler.step(joint_f1)

        logs.append({
            'epoch': ep,
            'epoch_time_sec': round(time.perf_counter()-t0,4),
            'train_loss': tr_loss,
            'val_old_macro_f1': old_s['macro_f1'],
            'val_new_macro_f1': new_s['macro_f1'],
            'val_joint_macro_f1': joint_f1,
            'lr': optimizer.param_groups[-1]['lr']
        })
        if joint_f1 > best['f1']:
            best = {'f1': joint_f1, 'state': {k:v.cpu().clone() for k,v in model.state_dict().items()}, 'epoch': ep}
            bad = 0
        else:
            bad += 1
        if ep >= int(cfg['train']['min_epochs']) and bad >= int(cfg['train']['patience']):
            break

    if best['state'] is not None:
        model.load_state_dict(best['state'])

    torch.save(model.state_dict(), dirs['checkpoints'] / 'model.pth')
    save_logs(logs, dirs['logs'] / 'train_log.csv')
    plot_training(logs, dirs['figures'] / 'train_val_curves.png', 'CatB Naive Full Retrain Training')

    y_old_t, y_old_p = evaluate(model, old_test_loader, device)
    y_new_t, y_new_p = evaluate(model, new_test_loader, device)
    old_stats = score(y_old_t, y_old_p)
    new_stats = score(y_new_t, y_new_p)

    save_confusion(y_old_t, y_old_p, old_classes, dirs['figures'] / 'confusion_old.png', 'CatB Naive Full Retrain Old Classes')
    save_confusion([v-len(old_classes) for v in y_new_t], [v-len(old_classes) for v in y_new_p], new_classes, dirs['figures'] / 'confusion_new.png', 'CatB Naive Full Retrain New Classes')

    runtime = {
        'total_wall_time_sec': round(time.perf_counter()-total_start,4),
        'train_wall_time_sec': round(sum(r['epoch_time_sec'] for r in logs),4),
        'peak_vram_mb': round(torch.cuda.max_memory_allocated()/(1024**2),2) if device.type=='cuda' else None,
        'process_ram_mb_end': get_process_ram_mb(),
    }
    extra = {'best_epoch': best['epoch']}
    dump_metrics(dirs, cfg, 'cat_b_naive_full_retrain_head_expand', cfg['meta']['cycle_name'], global_classes, old_classes, new_classes, old_stats, new_stats, runtime, model, extra=extra)


if __name__ == '__main__':
    main()
