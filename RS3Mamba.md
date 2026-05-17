# RS3Mamba 电线课题 — 训练 / 测试命令速查

> 本文件只放可直接复制的命令。环境见 [`SETUP.md`](SETUP.md)，项目介绍与结果表见 [`README.md`](README.md)。  
> **GitHub：** [https://github.com/zsyyywm/my_SR3MAmba](https://github.com/zsyyywm/my_SR3MAmba)  
> **克隆后**：先按 [`SETUP.md`](SETUP.md) **§0** 安装环境、预训练与数据，再执行下文训练/测试。

---

## 0. 进入工程

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate ssrs_rs3mamba
conda  activate /root/autodl-tmp/envs/ssrs

export SSRSROOT=/root/my_TransNext/SSRS-main/SSRS-main
cd "$SSRSROOT"
```

数据在 AutoDL 数据盘时：

```bash
export WIRE_SEG_DATA_ROOT=/root/autodl-tmp/DataA-B
export WIRE_SEG_DATAC_ROOT=/root/autodl-tmp/DataC/DataC
```

---

## 1. 训练

DataA：输入 256，阈值 0.5。

```bash
python tools/train_wire.py --wire-scheme dataa --seed 0 --batch-size 4
```

DataA（改进策略，与 SegFormer DataA 方案对齐：**不改 RS3Mamba 结构**，仍为 256 / 阈值 0.5）：

- **`--dataa-boost`**：快捷组合 — 加权 CE `[1, 5]`、前景图过采样（repeat=3）、`lr=3e-5`、warmup 5 epoch + 余弦降至 `--min-lr`。
- 亦可分项：`--ce-class-weight 1 5`、`--foreground-oversample`、`--foreground-repeat 3`、`--lr 3e-5`、`--warmup-epochs 5`、`--warmup-ratio 0.1`。

一键组合示例：

```bash
python tools/train_wire.py --wire-scheme dataa --seed 0 --batch-size 4 --dataa-boost
```

分项消融示例（仅加权 CE）：

```bash
python tools/train_wire.py --wire-scheme dataa --seed 0 --batch-size 4 \
  --ce-class-weight 1 5
```

仅稳定 LR（小学习率 + warmup + 余弦，`warmup_epochs=0` 时与旧版纯余弦一致）：

```bash
python tools/train_wire.py --wire-scheme dataa --seed 0 --batch-size 4 \
  --lr 3e-5 --warmup-epochs 5 --warmup-ratio 0.1 --min-lr 1e-6
```

说明：`--dataa-boost` 会固定写上组合；若要完全自定义学习率请不要用 `--dataa-boost`，改用分项参数。

DataB：输入 256，阈值 0.5。

```bash
python tools/train_wire.py --wire-scheme datab --seed 0 --batch-size 4
```

DataC：输入 512，阈值 0.5，兼容 `1` / `255` 两种前景编码。

```bash
python tools/train_wire.py --wire-scheme datac --seed 0 --batch-size 4
```

如 DataC 显存不足：

```bash
python tools/train_wire.py --wire-scheme datac --seed 0 --batch-size 2
```

每次训练输出：

```bash
data/checkpoints1/<YYYYMMDDHHMMSS+A或B或C>/
```

包含：

```bash
best_IoU_epoch_*.pth
last_model.pth
run_meta.json
val_metrics.csv
train_curves.png
val_foreground_trends.png
```

---

## 2. 测试

测试只传训练文件夹名，自动找 `best_IoU*.pth`，不会使用 latest。终端会打印与 SegFormer wire 测试一致的 ASCII 表格：`per class results:`（background / foreground 的 IoU、Acc、Fscore、Precision、Recall）以及 `Summary (wire binary):`（global 的 aAcc、IoU(fg)、F1 等）。

```bash
python tools/test_wire.py 20260515153000A
python tools/test_wire.py 20260515160000B
python tools/test_wire.py 20260515163000C
```

也可以传绝对路径：

```bash
python tools/test_wire.py /root/my_TransNext/SSRS-main/SSRS-main/data/checkpoints1/20260515153000A
```

每次测试输出：

```bash
data/checkpoints2/<YYYYMMDDHHMMSS+A或B或C>/
```

包含：

```bash
eval_metrics.json
eval_report.txt
seg_results.pkl
test_meta.json
README_result_row.md
```

自动更新 README 中对应 DataA/DataB/DataC 的待填写结果行：

```bash
python tools/test_wire.py 20260515153000A --update-readme
```

### 测试集逐张可视化（参考 test.py 目录结构）

对 `test` 分割下每张图导出：`{base}.png`、`{base}_comparison.png`、`01_pred_mask/`、`02_gt_mask/`、`03_overlap_red/`、`04_tp_fp_fn_tn_pixels/`（含 `color_legend.png`）。  
推理为 **softmax 类 1 概率 + 阈值**，预处理与 `WireDataset` 一致；指标为**全测试集像素**汇总后再算 IoU / P / R / F1 / aAcc。

```bash
python tools/visualize_wire_test.py --train-run 20260515153000A
```

指定权重与数据集（不依赖训练目录名后缀）：

```bash
python tools/visualize_wire_test.py \
  --checkpoint data/checkpoints1/20260515153000A/best_IoU_epoch_10.pth \
  --wire-scheme dataa \
  --output-dir /tmp/wire_vis_dataa
```

可选 **`--flip-tta`**（四向翻转 logits 平均）、`--threshold`、`--output-dir`（默认写入 `data/checkpoints2/vis_<时间><A|B|C>/`）。

---

## 3. 常用参数

```bash
python tools/train_wire.py --wire-scheme dataa \
  --epochs 300 \
  --patience 50 \
  --threshold 0.5 \
  --batch-size 4 \
  --lr 1e-4 \
  --seed 0
```

如果要强制使用 VMamba 预训练：

```bash
python tools/train_wire.py --wire-scheme dataa \
  --vmamba-pretrained RS3Mamba/pretrain/vmamba_tiny_e292.pth \
  --require-pretrained
```

如果要让 `timm` 加载 ResNet 分支预训练：

```bash
python tools/train_wire.py --wire-scheme dataa --timm-pretrained
```

---

## 5. 实验结果记录（wire / visualize_wire_test）

以下为 **2026-05-17** 使用 `visualize_wire_test.py`（**test** 集、**全像素汇总**、`threshold=0.5`、无 `--flip-tta`）终端 **Final Test Report** 中的数值；与 `README.md` 结果表一致。DataA 训练行对应示例为 **`--dataa-boost`** 的一次 run（见 §1）。

| 数据集 | Test 张数 | IoU(fg) | Precision | Recall | F1 | aAcc | 可视化目录（项目内相对路径） |
|--------|-----------|---------|-----------|--------|-----|------|------------------------------|
| DataA | 20 | 0.503 | 0.538 | 0.887 | 0.670 | 0.992 | `data/checkpoints2/vis_20260517143940A/` |
| DataB | 125 | 0.632 | 0.828 | 0.728 | 0.775 | 0.991 | `data/checkpoints2/vis_20260517144718B/` |
| DataC | 86 | 0.767 | 0.862 | 0.875 | 0.868 | 0.989 | `data/checkpoints2/vis_20260517144855C/` |

复现可视化（将 `<RUN>` 换成你的 `data/checkpoints1/` 下训练文件夹名）：

```bash
python tools/visualize_wire_test.py --train-run <RUN>
```

---
