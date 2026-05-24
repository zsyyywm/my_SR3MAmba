# RS3Mamba 电线二分类语义分割（个人课题仓库）

本仓库是**个人实验项目**：在论文模型 **RS³Mamba**（[RS³Mamba](https://ieeexplore.ieee.org/abstract/document/10556777)，IEEE GRSL 2024）基础上，针对 **DataA / DataB / DataC** 输电线路/线缆的 **二分类语义分割**（背景 vs 前景）。实现入口集中在 `tools/`，核心网络与配置放在 `RS3Mamba/`，不包含上游 SSRS 大仓库里的其它子课题代码。

**GitHub（本项目唯一克隆地址）：** [https://github.com/zsyyywm/my_SR3MAmba](https://github.com/zsyyywm/my_SR3MAmba)

---

## 从 GitHub 克隆后怎么做（按顺序）

1. **克隆**

   ```bash
   git clone https://github.com/zsyyywm/my_SR3MAmba.git
   cd my_SR3MAmba
   ```

2. **环境与依赖** — 打开 [`SETUP.md`](SETUP.md)，从 **§0「从 GitHub 克隆后的操作顺序」** 开始，依次完成 Conda、PyTorch、`mamba_ssm` 等安装（**§1**）。

3. **数据** — 按 **§2** 准备或指向 DataA / DataB / DataC 目录（`image/train|val|test` 与 `mask/train|val|test`）。可用环境变量 `WIRE_SEG_DATA_ROOT` / `WIRE_SEG_DATAC_ROOT` 覆盖默认路径。

4. **（可选）预训练** — **§3**：若将 `vmamba_tiny_e292.pth` 放到 `RS3Mamba/pretrain/`，会加载 VSSM 分支初始化；不放则从当前任务随机初始化。

5. **自检** — **§4** 确认路径与 `import mamba_ssm` 正常。

6. **训练、测试、可视化** — 命令在下方「常用命令」与 [`RS3Mamba.md`](RS3Mamba.md)（可直接复制）。

> 推送到自己的 GitHub、或从训练机**排除大文件**，见 [`SETUP.md`](SETUP.md) **§5**（含 `.gitignore` 说明，勿提交 `data/`、`* .pth`）。

---

## 文档分工

| 文档 | 内容 |
|------|------|
| **本 `README.md`** | 项目定位、克隆后流程总览、实验设定、**结果表**。 |
| **[`SETUP.md`](SETUP.md)** | 环境、依赖、数据路径、可选预训练、自检、**克隆/推送清单**。 |
| **[`RS3Mamba.md`](RS3Mamba.md)** | 训练 / 测试 / 可视化等**可复制命令**（路径、参数、续训等）。 |

---

## 仓库里有什么

| 路径 | 说明 |
|------|------|
| `tools/train_wire.py` | 训练入口（`--wire-scheme dataa|datab|datac`） |
| `tools/test_wire.py` | 测试入口（传入训练产物的文件夹名） |
| `tools/visualize_wire_test.py` | 测试集逐张 TP/FN/FP 可视化（`tp_fn_fp_replace/`）与像素级指标汇总 |
| `tools/wire_paths.py` | 数据集根目录解析 |
| `RS3Mamba/` | RS³Mamba 模型与相关脚本（本课题所用结构） |

训练与测试产物默认写在本地（**不随 Git 上传**）：

```text
data/checkpoints1/<时间戳+A|B|C>/   # 训练：权重、日志、验证曲线等
data/checkpoints2/<时间戳+A|B|C>/   # test_wire.py 测试输出
data/checkpoints2/vis_<时间><A|B|C>/ # visualize_wire_test.py：tp_fn_fp_replace/ + vis_meta.json
```

---

## 常用命令（详细见 RS3Mamba.md）

```bash
# 在仓库根目录，且已按 SETUP 激活环境
python tools/train_wire.py --wire-scheme dataa --seed 0 --batch-size 4
python tools/train_wire.py --wire-scheme datab --seed 0 --batch-size 4
python tools/train_wire.py --wire-scheme datac --seed 0 --batch-size 4

python tools/test_wire.py <训练文件夹名>
python tools/visualize_wire_test.py --train-run <训练文件夹名>
```

---

## 实验设定摘要

- 模型：`RS3Mamba(num_classes=2)`
- DataA / DataB 输入边长：256；DataC：512
- 前景阈值：0.5；最大 epoch：300；早停：验证集前景 IoU 连续 50 轮不提升
- 最佳权重：按**验证集前景 IoU** 保存 `best_IoU_epoch_*.pth`
- DataC：mask 中 **1 与 255 均视为前景类别**（与训练脚本约定一致）

---

## 实验结果（参考）

下文 **IoU / P / R / F1 / aAcc 为 0–1 比例**（乘 100 为百分比）。指标来自 `visualize_wire_test.py` 在 **test** 集上**全图素累计**后的 Final Test Report，`threshold=0.5`，单次前向、无 TTA。训练目录以你本机 `data/checkpoints1/` 为准。

| 日期 | 数据集 | 训练目录（示例） | 测试 / 可视化输出 | 权重 | IoU(fg) | Precision | Recall | F1 | 备注 |
|------|--------|------------------|-------------------|------|---------|-----------|--------|----|------|
| 2026-05-17 | DataA | `20260516135907A` 等 | `data/checkpoints2/vis_20260517143940A/` | `best_IoU_epoch_*.pth` | 0.503 | 0.538 | 0.887 | 0.670 | aAcc 0.992；20 张 test；`--dataa-boost` 见 RS3Mamba.md |
| 2026-05-17 | DataB | 与 `--train-run` 一致 | `data/checkpoints2/vis_20260517144718B/` | `best_IoU_epoch_*.pth` | 0.632 | 0.828 | 0.728 | 0.775 | aAcc 0.991；125 张 test |
| 2026-05-17 | DataC | 与 `--train-run` 一致 | `data/checkpoints2/vis_20260517144855C/` | `best_IoU_epoch_*.pth` | 0.767 | 0.862 | 0.875 | 0.868 | aAcc 0.989；86 张 test |

---

## 引用（RS³Mamba）

使用本仓库中的 RS³Mamba 结构做研究时，请引用原作者论文：

```bibtex
@ARTICLE{ma2024rs3mamba,
  author={Ma, Xianping and Zhang, Xiaokang and Pun, Man-On},
  journal={IEEE Geoscience and Remote Sensing Letters},
  title={RS$^3$Mamba: Visual State Space Model for Remote Sensing Image Semantic Segmentation},
  year={2024}
}
```

---

## 附录：与上游 SSRS 的关系

本 Git 仓库是从大型遥感分割合集 **[sstary/SSRS](https://github.com/sstary/SSRS)** 中**仅抽取**「RS3Mamba + 本课题 `tools`」所需文件构成的**精简个人仓库**，便于复现电线实验。原 SSRS 中还有 MFNet、SAM\_RS、GLGAN 等其它方法，**不在本仓库中**；若需阅读原文列表与更多引用，请访问上游 [SSRS](https://github.com/sstary/SSRS)。
