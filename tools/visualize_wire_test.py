#!/usr/bin/env python3
"""
RS3Mamba wire 测试集逐张可视化与像素级汇总指标（对齐参考 test.py 的目录与报告形态）。

推理：双类 logits -> softmax 类 1 概率 -> 阈值（默认 0.5），与 train_wire / test_wire 一致。
预处理：与 WireDataset 一致（RGB resize BILINEAR，mask NEAREST，归一化 MEAN/STD）。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import os.path as osp
import time
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from PIL import Image

# 与 train_wire 相同，保证与训练/val 一致
from train_wire import MEAN, STD, build_model, list_pairs, project_root
from wire_paths import (checkpoints1_dir, checkpoints2_dir, ensure_wire_dataset,
                        image_size_for_scheme)

EPS = 1e-7


def add_title_bar(img: np.ndarray, text: str, height: int = 35) -> np.ndarray:
    """在原图上方叠黑色标题栏与白色文字（BGR）。"""
    h, w, _ = img.shape
    bar = np.zeros((height, w, 3), dtype=np.uint8)
    cv2.putText(
        bar,
        text,
        (10, height - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return np.vstack([bar, img])


def calculate_binary_metrics(pred: np.ndarray, label: np.ndarray):
    """pred/label 为 0/1，H×W。"""
    p = pred.astype(np.uint8).ravel()
    g = label.astype(np.uint8).ravel()
    tp = int(np.sum((p == 1) & (g == 1)))
    fp = int(np.sum((p == 1) & (g == 0)))
    fn = int(np.sum((p == 0) & (g == 1)))
    tn = int(np.sum((p == 0) & (g == 0)))
    return tp, fp, fn, tn


def calculate_final_metrics(total_tp: int, total_fp: int, total_fn: int,
                            total_tn: int):
    """全测试集像素汇总后再算指标（numpy，不用 sklearn）。"""
    iou = total_tp / (total_tp + total_fp + total_fn + EPS)
    precision = total_tp / (total_tp + total_fp + EPS)
    recall = total_tp / (total_tp + total_fn + EPS)
    f1 = 2 * precision * recall / (precision + recall + EPS)
    denom = total_tp + total_fp + total_fn + total_tn
    aacc = (total_tp + total_tn) / (denom + EPS)
    return iou, precision, recall, f1, aacc


def smart_load_weights(net: torch.nn.Module, weights_path: str,
                       device: torch.device) -> bool:
    """兼容 state_dict / checkpoint['model'] 等；去 module. 前缀；按 key+shape 部分加载。"""
    if not osp.isfile(weights_path):
        print(f'[错误] 未找到权重: {weights_path}')
        return False

    ckpt = torch.load(weights_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict):
        if 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
        elif 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        elif 'model' in ckpt and isinstance(ckpt['model'], dict):
            state_dict = ckpt['model']
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    clean = {}
    for k, v in state_dict.items():
        nk = k[7:] if k.startswith('module.') else k
        clean[nk] = v

    model_dict = net.state_dict()
    matched = {
        k: v for k, v in clean.items()
        if k in model_dict and model_dict[k].shape == v.shape
    }
    model_dict.update(matched)
    net.load_state_dict(model_dict, strict=False)
    print(f'loaded weights: {weights_path}')
    print(f'matched params: {len(matched)} / {len(model_dict)}')
    return True


def preprocess_tensor_from_pil_rgb(img_rgb: Image.Image, image_size: int,
                                   device: torch.device) -> torch.Tensor:
    """与 WireDataset 一致：resize BILINEAR -> float/255 -> MEAN/STD -> NCHW。"""
    img = img_rgb.resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    t = torch.from_numpy(arr.transpose(2, 0, 1)).float().unsqueeze(0)
    return t.to(device)


def load_display_bgr(img_path: str, image_size: int) -> np.ndarray:
    """resize 与训练相同尺寸，用于可视化（BGR）。"""
    rgb = Image.open(img_path).convert('RGB').resize(
        (image_size, image_size), Image.BILINEAR)
    rgb_np = np.asarray(rgb, dtype=np.uint8)
    return cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)


def load_label_binary(mask_path: str, image_size: int,
                      mask_255_as_foreground: bool) -> np.ndarray:
    """与 WireDataset mask 分支一致：NEAREST resize，>0 -> 前景 1。"""
    mask_raw = np.array(
        Image.open(mask_path).resize(
            (image_size, image_size), Image.NEAREST),
        dtype=np.uint8)
    if mask_255_as_foreground:
        return (mask_raw > 0).astype(np.uint8)
    return (mask_raw > 0).astype(np.uint8)


def predict_foreground_prob(
        model: torch.nn.Module,
        img_tensor: torch.Tensor,
        use_tta: bool,
        device: torch.device,
) -> np.ndarray:
    """
    返回 H×W 的前景概率 float32 [0,1]。
    TTA：四向翻转 logits 平均后再 softmax（与参考单通道 TTA 对 logits 平均一致）。
    """
    model.eval()
    if not use_tta:
        with torch.no_grad():
            logits = model(img_tensor)
        prob = torch.softmax(logits, dim=1)[0, 1]
        return prob.cpu().numpy().astype(np.float32)

    aug_list = [
        (img_tensor, lambda x: x),
        (torch.flip(img_tensor, dims=[3]), lambda x: torch.flip(x, dims=[3])),
        (torch.flip(img_tensor, dims=[2]), lambda x: torch.flip(x, dims=[2])),
        (torch.flip(img_tensor, dims=[2, 3]),
         lambda x: torch.flip(x, dims=[2, 3])),
    ]
    logits_sum = None
    with torch.no_grad():
        for aug_t, inv_fn in aug_list:
            lg = model(aug_t)
            lg = inv_fn(lg)
            logits_sum = lg if logits_sum is None else logits_sum + lg
    logits_avg = logits_sum / len(aug_list)
    prob = torch.softmax(logits_avg, dim=1)[0, 1]
    return prob.cpu().numpy().astype(np.float32)


def save_overlap_red(img_bgr: np.ndarray, pred_map: np.ndarray,
                     label_arr: np.ndarray, save_path: str, alpha: float = 0.65):
    """TP 区域红色半透明叠加（BGR 红 = 0,0,255）。"""
    pred_map = pred_map.astype(np.uint8)
    label_arr = label_arr.astype(np.uint8)

    if pred_map.shape != label_arr.shape:
        pred_map = cv2.resize(
            pred_map,
            (label_arr.shape[1], label_arr.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    if img_bgr.shape[:2] != label_arr.shape:
        img_bgr = cv2.resize(
            img_bgr,
            (label_arr.shape[1], label_arr.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    overlap = (pred_map == 1) & (label_arr == 1)
    vis = img_bgr.copy()
    red_layer = np.zeros_like(img_bgr, dtype=np.uint8)
    red_layer[:, :] = (0, 0, 255)
    blended = cv2.addWeighted(img_bgr, 1 - alpha, red_layer, alpha, 0)
    vis[overlap] = blended[overlap]
    cv2.imwrite(save_path, vis)


def save_tp_fp_fn_tn_pixels(pred_map: np.ndarray, label_arr: np.ndarray,
                            save_path: str):
    """BGR：TP 绿、FP 红、FN 蓝、TN 黑。"""
    pred_map = pred_map.astype(np.uint8)
    label_arr = label_arr.astype(np.uint8)
    if pred_map.shape != label_arr.shape:
        pred_map = cv2.resize(
            pred_map,
            (label_arr.shape[1], label_arr.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    tp = (pred_map == 1) & (label_arr == 1)
    fp = (pred_map == 1) & (label_arr == 0)
    fn = (pred_map == 0) & (label_arr == 1)
    tn = (pred_map == 0) & (label_arr == 0)
    h, w = label_arr.shape
    color_map = np.zeros((h, w, 3), dtype=np.uint8)
    color_map[tn] = (0, 0, 0)
    color_map[tp] = (0, 255, 0)
    color_map[fp] = (0, 0, 255)
    color_map[fn] = (255, 0, 0)
    cv2.imwrite(save_path, color_map)


def save_color_legend(save_dir: str):
    legend = np.ones((160, 430, 3), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    items = [
        ('TP: pred=1, gt=1', (0, 255, 0), 30),
        ('FP: pred=1, gt=0', (0, 0, 255), 65),
        ('FN: pred=0, gt=1', (255, 0, 0), 100),
        ('TN: pred=0, gt=0', (0, 0, 0), 135),
    ]
    for text, color, y in items:
        cv2.rectangle(legend, (20, y - 20), (55, y + 5), color, -1)
        cv2.putText(legend, text, (70, y), font, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.imwrite(osp.join(save_dir, 'color_legend.png'), legend)


def resolve_run_dir(train_run: str) -> str:
    if osp.isdir(train_run):
        return osp.abspath(train_run)
    cand = osp.join(checkpoints1_dir(), train_run)
    if osp.isdir(cand):
        return osp.abspath(cand)
    raise FileNotFoundError(f'Cannot find training run: {train_run}')


def find_best_checkpoint(run_dir: str) -> str:
    hits = []
    for pattern in ('best_IoU*.pth', 'best_*.pth'):
        hits.extend(glob.glob(osp.join(run_dir, pattern)))
    hits = [p for p in hits if osp.isfile(p)]
    if not hits:
        raise FileNotFoundError(
            f'No best checkpoint in {run_dir}; use --checkpoint to specify path.')
    hits.sort(key=lambda p: osp.getmtime(p), reverse=True)
    return hits[0]


def infer_suffix_from_dir(run_dir: str | None) -> str:
    if not run_dir:
        return 'X'
    last = osp.basename(run_dir.rstrip(osp.sep))[-1:].upper()
    return last if last in ('A', 'B', 'C') else 'X'


def parse_args():
    p = argparse.ArgumentParser(
        description='RS3Mamba wire test visualization + aggregated metrics')
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        '--train-run',
        help='training folder name under data/checkpoints1/ or absolute path')
    g.add_argument(
        '--checkpoint',
        help='explicit .pth path (requires --wire-scheme)')
    p.add_argument(
        '--wire-scheme',
        choices=('dataa', 'datab', 'datac'),
        default=None,
        help='required with --checkpoint; auto from train-run folder suffix A/B/C')
    p.add_argument(
        '--output-dir',
        default=None,
        help='root save dir (default: data/checkpoints2/vis_<time><A|B|C>)')
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--flip-tta', action='store_true', help='4-way flip TTA on logits')
    p.add_argument('--device', default='cuda')
    p.add_argument('--timm-pretrained', action='store_true')
    p.add_argument('--vmamba-pretrained', default='')
    p.add_argument('--require-pretrained', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    os.chdir(project_root())

    _suf = {'dataa': 'A', 'datab': 'B', 'datac': 'C'}

    if args.checkpoint:
        if not args.wire_scheme:
            raise SystemExit('--wire-scheme is required when using --checkpoint')
        ckpt_path = osp.abspath(args.checkpoint)
        scheme = args.wire_scheme
        run_dir = ''
        suffix = _suf[scheme]
    else:
        run_dir = resolve_run_dir(args.train_run)
        ckpt_path = find_best_checkpoint(run_dir)
        suffix = infer_suffix_from_dir(run_dir)
        map_suf = {'A': 'dataa', 'B': 'datab', 'C': 'datac'}
        scheme = map_suf.get(suffix)
        if scheme is None:
            raise SystemExit(
                'Cannot infer --wire-scheme from run folder name; '
                'use names ending in A/B/C or pass --checkpoint + --wire-scheme')

    data_root = ensure_wire_dataset(scheme)
    image_size = image_size_for_scheme(scheme)
    mask_255_as_foreground = scheme == 'datac'

    if args.output_dir:
        predict_save_dir = osp.abspath(args.output_dir)
    else:
        vis_name = time.strftime('vis_%Y%m%d%H%M%S', time.localtime()) + suffix
        predict_save_dir = osp.join(checkpoints2_dir(), vis_name)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    os.makedirs(predict_save_dir, exist_ok=True)
    pred_mask_dir = osp.join(predict_save_dir, '01_pred_mask')
    gt_mask_dir = osp.join(predict_save_dir, '02_gt_mask')
    overlap_red_dir = osp.join(predict_save_dir, '03_overlap_red')
    tp_fp_fn_tn_dir = osp.join(predict_save_dir, '04_tp_fp_fn_tn_pixels')
    for d in (pred_mask_dir, gt_mask_dir, overlap_red_dir, tp_fp_fn_tn_dir):
        os.makedirs(d, exist_ok=True)

    model_args = SimpleNamespace(
        timm_pretrained=args.timm_pretrained,
        vmamba_pretrained=args.vmamba_pretrained or
        'RS3Mamba/pretrain/vmamba_tiny_e292.pth',
        require_pretrained=args.require_pretrained)
    net = build_model(model_args).to(device)

    if not smart_load_weights(net, ckpt_path, device):
        return

    net.eval()
    save_color_legend(tp_fp_fn_tn_dir)

    pairs = list_pairs(data_root, 'test')
    img_exts = ('.jpg', '.png', '.jpeg', '.bmp', '.tif', '.tiff')

    print('=' * 60)
    print(f'test pairs:    {len(pairs)} (image/mask from list_pairs)')
    print(f'data_root:     {data_root}')
    print(f'save dir:      {predict_save_dir}')
    print(f'scheme:        {scheme} (input {image_size})')
    print(f'checkpoint:    {ckpt_path}')
    print(f'threshold:     {args.threshold:.2f}')
    print(f'TTA:           {"4-flip logits avg" if args.flip_tta else "single forward"}')
    print(f'metric:        aggregate all pixels then IoU/P/R/F1/aAcc')
    print('=' * 60)

    total_tp = total_fp = total_fn = total_tn = 0
    valid_img = 0
    valid_label = 0

    for img_path, mask_path in pairs:
        base_name = osp.splitext(osp.basename(img_path))[0]

        img_rgb = Image.open(img_path).convert('RGB')
        img_tensor = preprocess_tensor_from_pil_rgb(img_rgb, image_size, device)
        pred_prob = predict_foreground_prob(net, img_tensor, args.flip_tta,
                                            device)

        if pred_prob.shape != (image_size, image_size):
            pred_prob = cv2.resize(
                pred_prob,
                (image_size, image_size),
                interpolation=cv2.INTER_LINEAR,
            )

        pred_map = (pred_prob > args.threshold).astype(np.uint8)
        img_bgr = load_display_bgr(img_path, image_size)

        mask_gray = (pred_map * 255).astype(np.uint8)
        mask_bgr = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2BGR)
        overlay = np.zeros_like(img_bgr)
        overlay[pred_map == 1] = [0, 255, 0]
        vis_overlay = cv2.addWeighted(img_bgr, 0.6, overlay, 0.4, 0)

        hstack = np.hstack([
            add_title_bar(img_bgr, 'Original Image'),
            add_title_bar(mask_bgr, 'Predicted Mask'),
            add_title_bar(vis_overlay, 'Segmentation Result'),
        ])
        cv2.imwrite(
            osp.join(predict_save_dir, f'{base_name}_comparison.png'), hstack)
        cv2.imwrite(osp.join(predict_save_dir, f'{base_name}.png'), mask_gray)

        label_arr = load_label_binary(mask_path, image_size,
                                      mask_255_as_foreground)
        valid_img += 1
        valid_label += 1

        pred_for_m = pred_map
        if pred_for_m.shape != label_arr.shape:
            pred_for_m = cv2.resize(
                pred_map,
                (label_arr.shape[1], label_arr.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        tp, fp, fn, tn = calculate_binary_metrics(pred_for_m, label_arr)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

        cv2.imwrite(
            osp.join(pred_mask_dir, f'{base_name}_pred.png'),
            (pred_for_m * 255).astype(np.uint8),
        )
        cv2.imwrite(
            osp.join(gt_mask_dir, f'{base_name}_gt.png'),
            (label_arr * 255).astype(np.uint8),
        )
        save_overlap_red(
            img_bgr,
            pred_for_m,
            label_arr,
            osp.join(overlap_red_dir, f'{base_name}_overlap_red.png'),
        )
        save_tp_fp_fn_tn_pixels(
            pred_for_m,
            label_arr,
            osp.join(tp_fp_fn_tn_dir, f'{base_name}_tp_fp_fn_tn.png'),
        )

        print(
            f'{osp.basename(img_path)} | TP={tp} FP={fp} FN={fn} TN={tn} | '
            f'pred_pixels={int(pred_for_m.sum())} gt_pixels={int(label_arr.sum())} | '
            f'prob_min={pred_prob.min():.4f} prob_max={pred_prob.max():.4f} '
            f'prob_mean={pred_prob.mean():.4f}')

    print(f'\nprediction finished -> {predict_save_dir}')

    iou, precision, recall, f1, aacc = calculate_final_metrics(
        total_tp, total_fp, total_fn, total_tn)

    print('\n' + '=' * 60)
    print('Final Test Report')
    print(f'Valid images:  {valid_img}')
    print(f'Valid labels:  {valid_label}')
    print(f'Threshold:     {args.threshold:.2f}')
    print('-' * 60)
    print('Aggregated Pixel Counts:')
    print(f'TP: {total_tp}')
    print(f'FP: {total_fp}')
    print(f'FN: {total_fn}')
    print(f'TN: {total_tn}')
    print('-' * 60)
    print(f'IoU:       {iou:.6f}')
    print(f'Precision: {precision:.6f}')
    print(f'Recall:    {recall:.6f}')
    print(f'F1 Score:  {f1:.6f}')
    print(f'aAcc:      {aacc:.6f}')
    print('=' * 60)

    meta_path = osp.join(predict_save_dir, 'vis_meta.json')
    try:
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump({
                'checkpoint': ckpt_path,
                'train_run': run_dir or None,
                'wire_scheme': scheme,
                'data_root': data_root,
                'image_size': image_size,
                'threshold': args.threshold,
                'flip_tta': bool(args.flip_tta),
                'aggregated': {
                    'tp': total_tp,
                    'fp': total_fp,
                    'fn': total_fn,
                    'tn': total_tn,
                    'IoU': iou,
                    'Precision': precision,
                    'Recall': recall,
                    'F1': f1,
                    'aAcc': aacc,
                },
            }, f, indent=2, ensure_ascii=False)
        print(f'vis_meta.json written -> {meta_path}')
    except OSError:
        pass


if __name__ == '__main__':
    main()
