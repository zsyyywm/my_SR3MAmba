#!/usr/bin/env python3
"""Evaluate an RS3Mamba wire training run by folder name."""
import argparse
import glob
import os
import os.path as osp
import pickle
import time

import torch
from torch.utils.data import DataLoader

from train_wire import (WireDataset, build_model, c, compute_metrics_from_counts,
                        ensure_wire_dataset, image_size_for_scheme,
                        project_root, suffix_for_scheme, wire_binary_eval_tables,
                        write_json)
from wire_paths import checkpoints1_dir, checkpoints2_dir


@torch.no_grad()
def predict_and_eval(model, loader, device, threshold):
    model.eval()
    outputs = []
    tp = fp = fn = tn = 0
    for imgs, masks in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(imgs)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = probs >= float(threshold)
        valid = masks != 255
        gt = masks == 1
        tp += int((preds & gt & valid).sum().item())
        fp += int((preds & (~gt) & valid).sum().item())
        fn += int(((~preds) & gt & valid).sum().item())
        tn += int(((~preds) & (~gt) & valid).sum().item())
        outputs.extend([p.cpu().numpy().astype('uint8') for p in preds])
    metrics = compute_metrics_from_counts(tp, fp, fn, tn)
    return outputs, metrics, (tp, fp, fn, tn)


def resolve_run_dir(train_run):
    if osp.isdir(train_run):
        return osp.abspath(train_run)
    cand = osp.join(checkpoints1_dir(), train_run)
    if osp.isdir(cand):
        return osp.abspath(cand)
    raise FileNotFoundError(f'Cannot find training run: {train_run}')


def find_best_checkpoint(run_dir):
    hits = []
    for pattern in ('best_IoU*.pth', 'best_*.pth'):
        hits.extend(glob.glob(osp.join(run_dir, pattern)))
    hits = [p for p in hits if osp.isfile(p)]
    if not hits:
        raise FileNotFoundError(
            f'No best checkpoint found in {run_dir}; latest is intentionally not used.')
    hits.sort(key=lambda p: osp.getmtime(p), reverse=True)
    return hits[0]


def infer_suffix(run_dir):
    last = osp.basename(run_dir.rstrip(osp.sep))[-1:].upper()
    return last if last in ('A', 'B', 'C') else 'X'


def scheme_from_suffix(suffix):
    return {'A': 'dataa', 'B': 'datab', 'C': 'datac'}[suffix]


def dataset_name_from_suffix(suffix):
    return {'A': 'DataA', 'B': 'DataB', 'C': 'DataC'}.get(suffix, 'Unknown')


def note_from_suffix(suffix):
    if suffix == 'C':
        return 'RS3Mamba, 512, threshold=0.5, 1/255=foreground'
    if suffix in ('A', 'B'):
        return 'RS3Mamba, 256, threshold=0.5'
    return 'RS3Mamba, threshold=0.5'


def pct(v):
    return f'{float(v) * 100:.2f}'


def build_readme_result_row(eval_time, suffix, run_dir, out_dir, ckpt_path,
                            metrics):
    date = eval_time.split(' ')[0]
    return (
        f'| {date} | {dataset_name_from_suffix(suffix)} | '
        f'`{osp.basename(run_dir)}` | `{osp.basename(out_dir)}` | '
        f'`{osp.basename(ckpt_path)}` | {pct(metrics["IoU"])} | '
        f'{pct(metrics["Precision"])} | {pct(metrics["Recall"])} | '
        f'{pct(metrics["F1"])} | {note_from_suffix(suffix)} |')


def update_readme_row(suffix, row):
    readme_path = osp.join(project_root(), 'README.md')
    dataset_name = dataset_name_from_suffix(suffix)
    with open(readme_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    target = f'| 待填写 | {dataset_name} |'
    replaced = False
    for i, line in enumerate(lines):
        if line.startswith(target):
            lines[i] = row + '\n'
            replaced = True
            break
    if not replaced:
        marker = '|------|--------|----------|----------|------|---------|-----------|--------|----|------|'
        for i, line in enumerate(lines):
            if line.strip() == marker:
                lines.insert(i + 1, row + '\n')
                replaced = True
                break
    if not replaced:
        raise RuntimeError('Cannot find README result table.')
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    return readme_path


def parse_args():
    parser = argparse.ArgumentParser(description='RS3Mamba wire test')
    parser.add_argument('train_run')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--update-readme', action='store_true')
    parser.add_argument('--timm-pretrained', action='store_true')
    parser.add_argument(
        '--vmamba-pretrained',
        default='',
        help='not needed for eval; checkpoint weights are loaded from train run')
    parser.add_argument('--require-pretrained', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    os.chdir(project_root())
    run_dir = resolve_run_dir(args.train_run)
    ckpt_path = find_best_checkpoint(run_dir)
    suffix = infer_suffix(run_dir)
    scheme = scheme_from_suffix(suffix)
    data_root = ensure_wire_dataset(scheme)
    image_size = image_size_for_scheme(scheme)
    out_name = time.strftime('%Y%m%d%H%M%S', time.localtime()) + suffix
    out_dir = osp.join(checkpoints2_dir(), out_name)
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    test_ds = WireDataset(data_root, 'test', image_size, scheme == 'datac',
                          False)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)
    model = build_model(args).to(device)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt.get('model', ckpt)
    model.load_state_dict(state, strict=True)
    print(c('[测试]', 'cyan') + f' train_run={run_dir}')
    print(c('[测试]', 'cyan') + f' best={ckpt_path}')
    print(c('[测试]', 'cyan') + f' out_dir={out_dir}')
    outputs, metrics, counts = predict_and_eval(model, test_loader, device,
                                                args.threshold)
    tp, fp, fn, tn = counts
    class_tbl, summary_tbl, per_class = wire_binary_eval_tables(tp, fp, fn, tn)

    print('per class results:')
    print('\n' + class_tbl)
    print('Summary (wire binary):')
    print('\n' + summary_tbl)

    seg_pkl = osp.join(out_dir, 'seg_results.pkl')
    with open(seg_pkl, 'wb') as f:
        pickle.dump(outputs, f)
    payload = {
        'summary': {k: float(v) for k, v in metrics.items()},
        'summary_percent': {
            'IoU_fg': round(metrics['IoU'] * 100, 2),
            'Precision': round(metrics['Precision'] * 100, 2),
            'Recall': round(metrics['Recall'] * 100, 2),
            'F1': round(metrics['F1'] * 100, 2),
            'aAcc': round(metrics['aAcc'] * 100, 2),
        },
        'per_class': per_class,
    }
    write_json(osp.join(out_dir, 'eval_metrics.json'), payload)
    report = (
        'per class results:\n\n'
        f'{class_tbl}\n\n'
        'Summary (wire binary):\n\n'
        f'{summary_tbl}\n')
    with open(osp.join(out_dir, 'eval_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report)

    eval_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    row = build_readme_result_row(eval_time, suffix, run_dir, out_dir,
                                  ckpt_path, metrics)
    row_path = osp.join(out_dir, 'README_result_row.md')
    with open(row_path, 'w', encoding='utf-8') as f:
        f.write(row + '\n')
    write_json(
        osp.join(out_dir, 'test_meta.json'), {
            'eval_time': eval_time,
            'train_run': run_dir,
            'checkpoint': ckpt_path,
            'seg_results': seg_pkl,
            'metrics': {k: float(v) for k, v in metrics.items()},
            'readme_result_row': row,
        })
    print(c('[README结果行]', 'yellow'))
    print(row)
    if args.update_readme:
        print(c('[README已更新]', 'green') + f' {update_readme_row(suffix, row)}')
    print(c('[完成]', 'green') + f' 测试产物已写入 {out_dir}')


if __name__ == '__main__':
    main()
