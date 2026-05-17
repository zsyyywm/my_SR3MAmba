#!/usr/bin/env python3
"""Train RS3Mamba on DataA/DataB/DataC wire binary segmentation."""
import argparse
import csv
import json
import math
import os
import os.path as osp
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from wire_paths import (checkpoints1_dir, ensure_wire_dataset,
                        image_size_for_scheme, project_root,
                        suffix_for_scheme)


MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32) / 255.0
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32) / 255.0


def c(text, color):
    colors = {
        'green': '\033[92m',
        'yellow': '\033[93m',
        'red': '\033[91m',
        'cyan': '\033[96m',
        'magenta': '\033[95m',
        'blue': '\033[94m',
        'white': '\033[97m',
    }
    return f"{colors.get(color, '')}{text}\033[0m"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_pairs(data_root, split):
    img_dir = osp.join(data_root, 'image', split)
    mask_dir = osp.join(data_root, 'mask', split)
    pairs = []
    for name in sorted(os.listdir(img_dir)):
        if not name.lower().endswith('.jpg'):
            continue
        base = osp.splitext(name)[0]
        mask = osp.join(mask_dir, base + '.png')
        if osp.isfile(mask):
            pairs.append((osp.join(img_dir, name), mask))
    if not pairs:
        raise FileNotFoundError(f'No image/mask pairs found in {img_dir}')
    return pairs


def oversample_foreground_pairs(pairs, repeat, min_pixels):
    """Repeat (repeat-1) extra copies for images with enough fg pixels (SegFormer wire)."""
    repeat = max(1, int(repeat))
    min_pixels = max(1, int(min_pixels))
    original = list(pairs)
    extra = []
    for img_path, mask_path in original:
        mask_raw = np.array(Image.open(mask_path), dtype=np.uint8)
        fg = int(np.count_nonzero(mask_raw > 0))
        if fg >= min_pixels:
            extra.extend([(img_path, mask_path)] * (repeat - 1))
    if extra:
        n0 = len(original)
        out = original + extra
        print(
            c('[过采样]', 'magenta') +
            f' 含前景样本 repeat={repeat}, min_px={min_pixels}: '
            f'{n0} -> {len(out)} 条训练对')
        return out
    return original


class WireDataset(Dataset):
    def __init__(self, data_root, split, image_size, mask_255_as_foreground=False,
                 augment=False, foreground_oversample=False,
                 foreground_repeat=3, foreground_min_pixels=1):
        self.pairs = list_pairs(data_root, split)
        if foreground_oversample and split == 'train':
            self.pairs = oversample_foreground_pairs(
                self.pairs, foreground_repeat, foreground_min_pixels)
        self.image_size = int(image_size)
        self.mask_255_as_foreground = bool(mask_255_as_foreground)
        self.augment = bool(augment)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        img = Image.open(img_path).convert('RGB').resize(
            (self.image_size, self.image_size), Image.BILINEAR)
        mask_raw = np.array(
            Image.open(mask_path).resize(
                (self.image_size, self.image_size), Image.NEAREST),
            dtype=np.uint8)
        if self.mask_255_as_foreground:
            # DataC train masks use 1 as foreground, while val/test use 255.
            # Normalize both encodings to class 1 for fair foreground metrics.
            mask = (mask_raw > 0).astype(np.int64)
        else:
            mask = (mask_raw > 0).astype(np.int64)

        img = np.asarray(img, dtype=np.float32) / 255.0
        if self.augment and random.random() < 0.5:
            img = img[:, ::-1, :].copy()
            mask = mask[:, ::-1].copy()
        img = (img - MEAN) / STD
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()
        mask = torch.from_numpy(mask).long()
        return img, mask


def build_model(args):
    rs3_root = osp.join(project_root(), 'RS3Mamba')
    if rs3_root not in sys.path:
        sys.path.insert(0, rs3_root)
    from model.RS3Mamba import RS3Mamba, load_pretrained_ckpt

    model = RS3Mamba(num_classes=2, pretrained=args.timm_pretrained)
    if args.vmamba_pretrained:
        ckpt = args.vmamba_pretrained
        if osp.isfile(ckpt):
            model = load_pretrained_ckpt(model, ckpt)
        elif args.require_pretrained:
            raise FileNotFoundError(ckpt)
        else:
            print(c('[预训练]', 'yellow') + f' 未找到 {ckpt}，跳过 VMamba 预训练。')
    return model


def compute_metrics_from_counts(tp, fp, fn, tn):
    eps = 1e-12
    iou = tp / max(tp + fp + fn, eps)
    precision = tp / max(tp + fp, eps)
    recall = tp / max(tp + fn, eps)
    f1 = 2 * precision * recall / max(precision + recall, eps)
    aacc = (tp + tn) / max(tp + fp + fn + tn, eps)
    return {
        'IoU': float(iou),
        'Precision': float(precision),
        'Recall': float(recall),
        'F1': float(f1),
        'aAcc': float(aacc),
    }


def _ascii_table(rows):
    """Same grid style as mmcv.utils.AsciiTable (SegFormer wire eval)."""
    str_rows = [[str(x) for x in row] for row in rows]
    if not str_rows:
        return ''
    n = len(str_rows[0])
    widths = [
        max(len(str_rows[r][c]) for r in range(len(str_rows))) for c in range(n)
    ]
    sep = '+' + '+'.join('-' * (w + 2) for w in widths) + '+'

    def line(row):
        return '|' + '|'.join(
            ' ' + row[c].ljust(widths[c]) + ' ' for c in range(n)) + '|'

    out = [sep, line(str_rows[0]), sep]
    for row in str_rows[1:]:
        out.append(line(row))
    out.append(sep)
    return '\n'.join(out)


def wire_binary_eval_tables(tp, fp, fn, tn):
    """Per-class + summary tables aligned with SegFormer WireBinaryDataset.evaluate."""
    eps = 1e-12
    iou_bg = tn / max(tn + fp + fn, eps)
    iou_fg = tp / max(tp + fp + fn, eps)
    acc_bg = tn / max(tn + fp, eps)
    acc_fg = tp / max(tp + fn, eps)
    prec_bg = tn / max(tn + fn, eps)
    prec_fg = tp / max(tp + fp, eps)
    rec_bg = tn / max(tn + fp, eps)
    rec_fg = tp / max(tp + fn, eps)

    def f1(p, r):
        return 2 * p * r / max(p + r, eps)

    f1_bg = f1(prec_bg, rec_bg)
    f1_fg = f1(prec_fg, rec_fg)
    aacc = (tp + tn) / max(tp + fp + fn + tn, eps)

    def pct(x):
        return round(float(x) * 100, 2)

    class_data = [
        ['Class', 'IoU', 'Acc', 'Fscore', 'Precision', 'Recall'],
        [
            'background',
            pct(iou_bg),
            pct(acc_bg),
            pct(f1_bg),
            pct(prec_bg),
            pct(rec_bg),
        ],
        [
            'foreground',
            pct(iou_fg),
            pct(acc_fg),
            pct(f1_fg),
            pct(prec_fg),
            pct(rec_fg),
        ],
    ]
    summary_data = [[
        'Scope',
        'aAcc',
        'IoU(fg)',
        'F1',
        'Precision',
        'Recall',
    ], [
        'global',
        pct(aacc),
        pct(iou_fg),
        pct(f1_fg),
        pct(prec_fg),
        pct(rec_fg),
    ]]
    class_table = _ascii_table(class_data)
    summary_table = _ascii_table(summary_data)
    per_class_payload = [{
        'class': 'background',
        'IoU': float(iou_bg),
        'Acc': float(acc_bg),
        'F1': float(f1_bg),
        'Precision': float(prec_bg),
        'Recall': float(rec_bg),
    }, {
        'class': 'foreground',
        'IoU': float(iou_fg),
        'Acc': float(acc_fg),
        'F1': float(f1_fg),
        'Precision': float(prec_fg),
        'Recall': float(rec_fg),
    }]
    return class_table, summary_table, per_class_payload


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    tp = fp = fn = tn = 0
    for imgs, masks in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(imgs)
        probs = F.softmax(logits, dim=1)[:, 1]
        preds = probs >= float(threshold)
        valid = masks != 255
        gt = masks == 1
        tp += int((preds & gt & valid).sum().item())
        fp += int((preds & (~gt) & valid).sum().item())
        fn += int(((~preds) & gt & valid).sum().item())
        tn += int(((~preds) & (~gt) & valid).sum().item())
    return compute_metrics_from_counts(tp, fp, fn, tn)


def write_json(path, payload):
    os.makedirs(osp.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def append_val_csv(path, epoch, metrics, loss):
    write_header = not osp.isfile(path)
    with open(path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                'epoch', 'IoU_fg', 'Precision', 'Recall', 'F1', 'aAcc',
                'train_loss'
            ])
        writer.writerow([
            epoch,
            metrics['IoU'],
            metrics['Precision'],
            metrics['Recall'],
            metrics['F1'],
            metrics['aAcc'],
            loss,
        ])


def save_curves(run_dir, train_losses, val_hist):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if train_losses:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(range(1, len(train_losses) + 1), train_losses)
        ax.set_xlabel('epoch')
        ax.set_title('train loss')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(osp.join(run_dir, 'train_curves.png'), dpi=150)
        plt.close(fig)
    if val_hist:
        fig, ax = plt.subplots(figsize=(7, 4))
        xs = [x['epoch'] for x in val_hist]
        for key, color in [('IoU', 'r'), ('Precision', 'b'),
                           ('Recall', 'g'), ('F1', 'm')]:
            ax.plot(xs, [x[key] for x in val_hist], color + '-o', label=key)
        ax.set_xlabel('epoch')
        ax.set_title('validation foreground metrics')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(osp.join(run_dir, 'val_foreground_trends.png'), dpi=150)
        plt.close(fig)


def lr_factor(epoch_1based, args):
    """LR multiplier vs ``args.lr`` (warmup then cosine to ``args.min_lr``)."""
    e = epoch_1based - 1
    E = max(int(args.epochs), 1)
    w = max(int(args.warmup_epochs), 0)
    wr = float(args.warmup_ratio)
    eta = float(args.min_lr) / float(args.lr) if float(args.lr) > 1e-15 else 0.0

    if w <= 0:
        denom = max(E - 1, 1)
        cos_t = min(max(float(e) / float(denom), 0.0), 1.0)
        cos = 0.5 * (1 + math.cos(math.pi * cos_t))
        return eta + (1.0 - eta) * cos

    if e < w:
        denom = max(w - 1, 1)
        return wr + (1.0 - wr) * float(e) / float(denom)

    denom = max(E - w - 1, 1)
    cos_t = min(max(float(e - w) / float(denom), 0.0), 1.0)
    cos = 0.5 * (1 + math.cos(math.pi * cos_t))
    return eta + (1.0 - eta) * cos


def parse_args():
    parser = argparse.ArgumentParser(description='RS3Mamba wire training')
    parser.add_argument('--wire-scheme', choices=('dataa', 'datab', 'datac'),
                        required=True)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--min-lr', type=float, default=0.0,
                        help='cosine floor (absolute LR); 0 matches legacy CosineAnnealingLR')
    parser.add_argument('--warmup-epochs', type=int, default=0,
                        help='linear warmup epochs; 0 = cosine decay only (legacy)')
    parser.add_argument('--warmup-ratio', type=float, default=0.1,
                        help='LR multiplier at warmup start vs --lr')
    parser.add_argument(
        '--ce-class-weight',
        nargs=2,
        type=float,
        default=None,
        metavar=('W_BG', 'W_FG'),
        help='CrossEntropy class weights [background, foreground], e.g. 1 5')
    parser.add_argument(
        '--foreground-oversample',
        action='store_true',
        help='repeat train pairs that contain foreground (SegFormer WireBinaryDataset)')
    parser.add_argument('--foreground-repeat', type=int, default=3)
    parser.add_argument('--foreground-min-pixels', type=int, default=1)
    parser.add_argument(
        '--dataa-boost',
        action='store_true',
        help='DataA preset: CE [1,5] + fg oversample + lr=3e-5 + warmup 5 (no model change)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--timm-pretrained', action='store_true',
                        help='let timm load the ResNet branch pretrained weights')
    parser.add_argument(
        '--vmamba-pretrained',
        default='RS3Mamba/pretrain/vmamba_tiny_e292.pth',
        help='optional VMamba checkpoint path; skipped if missing by default')
    parser.add_argument('--require-pretrained', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    os.chdir(project_root())
    set_seed(args.seed)

    if args.dataa_boost:
        if args.wire_scheme != 'dataa':
            raise SystemExit(
                c('[错误]', 'red') + ' --dataa-boost 仅适用于 --wire-scheme dataa')
        args.ce_class_weight = args.ce_class_weight or [1.0, 5.0]
        args.foreground_oversample = True
        args.lr = 3e-5
        if not args.warmup_epochs:
            args.warmup_epochs = 5
        if args.min_lr <= 0:
            args.min_lr = 1e-6
        print(
            c('[DataA boost]', 'yellow') +
            ' 与 SegFormer DataA 改进方案对齐：加权 CE [1,5]、前景过采样、'
            f'lr={args.lr}、warmup_epochs={args.warmup_epochs}、min_lr={args.min_lr}'
            '（未修改 RS3Mamba 结构）')

    data_root = ensure_wire_dataset(args.wire_scheme)
    suffix = suffix_for_scheme(args.wire_scheme)
    image_size = image_size_for_scheme(args.wire_scheme)
    run_name = time.strftime('%Y%m%d%H%M%S', time.localtime()) + suffix
    run_dir = osp.join(checkpoints1_dir(), run_name)
    os.makedirs(run_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    mask_255_as_foreground = args.wire_scheme == 'datac'
    train_ds = WireDataset(
        data_root,
        'train',
        image_size,
        mask_255_as_foreground,
        True,
        foreground_oversample=args.foreground_oversample,
        foreground_repeat=args.foreground_repeat,
        foreground_min_pixels=args.foreground_min_pixels)
    val_ds = WireDataset(data_root, 'val', image_size,
                         mask_255_as_foreground, False)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False)
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False)

    model = build_model(args).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ce_weight = None
    if args.ce_class_weight is not None:
        ce_weight = torch.tensor(
            args.ce_class_weight, dtype=torch.float32, device=device)
        print(c('[损失]', 'cyan') +
              f' CrossEntropy class_weight={args.ce_class_weight}')
    criterion = torch.nn.CrossEntropyLoss(
        weight=ce_weight, ignore_index=255)

    use_cosine_only = args.warmup_epochs <= 0
    scheduler = None
    if use_cosine_only:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(args.epochs, 1), eta_min=args.min_lr)

    meta = {
        'model': 'RS3Mamba',
        'wire_scheme': args.wire_scheme,
        'data_root': data_root,
        'image_size': image_size,
        'threshold': args.threshold,
        'epochs': args.epochs,
        'patience': args.patience,
        'batch_size': args.batch_size,
        'seed': args.seed,
        'run_dir': run_dir,
        'lr': args.lr,
        'min_lr': args.min_lr,
        'warmup_epochs': args.warmup_epochs,
        'warmup_ratio': args.warmup_ratio,
        'lr_scheduler': 'cosine_annealing' if use_cosine_only else 'warmup_cosine',
        'ce_class_weight': args.ce_class_weight,
        'foreground_oversample': args.foreground_oversample,
        'foreground_repeat': args.foreground_repeat,
        'foreground_min_pixels': args.foreground_min_pixels,
        'dataa_boost': bool(args.dataa_boost),
    }
    write_json(osp.join(run_dir, 'run_meta.json'), meta)
    print(c('[Run]', 'cyan') + f' {run_dir}')
    print(c('[Data]', 'cyan') + f' {data_root} | train={len(train_ds)} val={len(val_ds)}')

    best_iou = -1.0
    bad_epochs = 0
    train_losses = []
    val_hist = []
    last_path = osp.join(run_dir, 'last_model.pth')
    best_path = None

    for epoch in range(1, args.epochs + 1):
        if not use_cosine_only:
            mult = lr_factor(epoch, args)
            for pg in optimizer.param_groups:
                pg['lr'] = args.lr * mult

        model.train()
        total_loss = 0.0
        for step, (imgs, masks) in enumerate(train_loader, start=1):
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            if step % args.log_interval == 0 or step == len(train_loader):
                mem = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
                lr = optimizer.param_groups[0]['lr']
                print(
                    f'{c("[训练]", "green")} epoch={epoch}/{args.epochs} | '
                    f'batch={step}/{len(train_loader)} | '
                    f'{c("GPU Mem", "magenta")} {mem:.2f} MB | '
                    f'{c("Loss", "yellow")} {float(loss.item()):.6f} | '
                    f'{c("LR", "cyan")} {lr:.8f} | '
                    f'{c("Image_size", "blue")} {image_size}')
        if scheduler is not None:
            scheduler.step()
        avg_loss = total_loss / max(len(train_loader), 1)
        train_losses.append(avg_loss)

        metrics = evaluate(model, val_loader, device, args.threshold)
        metrics['epoch'] = epoch
        val_hist.append(metrics)
        append_val_csv(osp.join(run_dir, 'val_metrics.csv'), epoch, metrics,
                       avg_loss)
        print(
            f'{c("[验证]", "red")} epoch={epoch}/{args.epochs} | '
            f'IoU={metrics["IoU"] * 100:.2f} | '
            f'P={metrics["Precision"] * 100:.2f} | '
            f'R={metrics["Recall"] * 100:.2f} | '
            f'F1={metrics["F1"] * 100:.2f} | '
            f'aAcc={metrics["aAcc"] * 100:.2f}')

        torch.save({
            'model': model.state_dict(),
            'epoch': epoch,
            'metrics': metrics,
            'meta': meta,
        }, last_path)
        if metrics['IoU'] > best_iou:
            best_iou = metrics['IoU']
            bad_epochs = 0
            if best_path and osp.isfile(best_path):
                os.remove(best_path)
            best_path = osp.join(run_dir, f'best_IoU_epoch_{epoch}.pth')
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'metrics': metrics,
                'meta': meta,
            }, best_path)
            print(c('[Best]', 'green') + f' foreground IoU={best_iou:.6f} -> {best_path}')
        else:
            bad_epochs += 1
        save_curves(run_dir, train_losses, val_hist)
        if bad_epochs >= args.patience:
            print(c('[早停]', 'yellow') + f' {args.patience} epochs no IoU improvement.')
            break


if __name__ == '__main__':
    main()
