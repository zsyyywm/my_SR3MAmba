# SSRS / RS3Mamba 电线实验环境配置

> 本文件只放环境、依赖、数据路径与自检。训练/测试命令见 [`RS3Mamba.md`](RS3Mamba.md)，项目介绍与结果记录见 [`README.md`](README.md)。  
> **GitHub：** [https://github.com/zsyyywm/my_SR3MAmba](https://github.com/zsyyywm/my_SR3MAmba)

---

## 0. 从 GitHub 克隆后的操作顺序

1. `git clone https://github.com/zsyyywm/my_SR3MAmba.git`（目录名以远端为准），`cd` 到**工程根**（含 `tools/`、`RS3Mamba/`、`README.md` 的目录）。
2. 按 **§1** 创建并激活 Conda 环境，安装 PyTorch 与 `mamba_ssm` 等依赖。
3.（可选）按 **§3** 将 `vmamba_tiny_e292.pth` 放到 `RS3Mamba/pretrain/`，不做则从零训练。
4. 按 **§2** 放置或指向 DataA / DataB / DataC 数据集。
5. 按 **§4** 自检通过后，打开 [`RS3Mamba.md`](RS3Mamba.md) 执行训练、`test_wire.py`、`visualize_wire_test.py`。

维护者上传代码时务必阅读 **§5**，确保 **`tools/` + `RS3Mamba/` 全量源码与三份 Markdown、`.gitignore` 齐备**，且不要把 `data/`、权重 `.pth` 推上去。

---

## 工程根

```bash
export SSRSROOT=/root/my_TransNext/SSRS-main/SSRS-main
cd "$SSRSROOT"
```

---

## 1. Conda 环境

`RS3Mamba` 依赖 `mamba_ssm`、`causal_conv1d`、`monai`、`timm`、`einops` 等包，建议单独建环境。

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda create -n ssrs_rs3mamba python=3.8 -y
conda activate ssrs_rs3mamba
```

按你的 CUDA/PyTorch 版本安装 PyTorch。若复用已有可跑 Mamba 的环境，可跳过这一步。

```bash
pip install torch torchvision
pip install timm einops monai scikit-image scikit-learn matplotlib pillow tqdm
pip install causal-conv1d mamba-ssm
```

如果 `mamba-ssm` 安装失败，优先参考 `RS3Mamba/README.md` 中提到的 VM-UNet / VMamba 安装方式，保证 `import mamba_ssm` 可用。

---

## 2. 数据路径

默认自动识别当前工作区：

```bash
/root/my_TransNext/DataA-B/DataA
/root/my_TransNext/DataA-B/DataB
/root/my_TransNext/DataC/DataC
```

结构必须为：

```bash
image/train  image/val  image/test
mask/train   mask/val   mask/test
```

如果数据在 AutoDL 数据盘，设置：

```bash
export WIRE_SEG_DATA_ROOT=/root/autodl-tmp/DataA-B
export WIRE_SEG_DATAC_ROOT=/root/autodl-tmp/DataC/DataC
```

或分别指定：

```bash
export WIRE_SEG_DATAA_ROOT=/root/autodl-tmp/DataA-B/DataA
export WIRE_SEG_DATAB_ROOT=/root/autodl-tmp/DataA-B/DataB
export WIRE_SEG_DATAC_ROOT=/root/autodl-tmp/DataC/DataC
```

---

## 3. 可选预训练

原 `RS3Mamba` 论文代码使用 `RS3Mamba/pretrain/vmamba_tiny_e292.pth` 初始化 VSSM 分支。本项目的电线入口默认：如果该文件存在就加载，不存在则跳过并继续训练。

放置路径：

```bash
mkdir -p RS3Mamba/pretrain
# 将 vmamba_tiny_e292.pth 放到 RS3Mamba/pretrain/vmamba_tiny_e292.pth
```

如果你希望没有预训练时直接报错：

```bash
python tools/train_wire.py --wire-scheme dataa --require-pretrained
```

---

## 4. 自检

```bash
cd "$SSRSROOT"
python - <<'PY'
import os, sys
sys.path.insert(0, os.path.join(os.getcwd(), 'tools'))
from wire_paths import dataa_root, datab_root, datac_root
print('DataA:', dataa_root())
print('DataB:', datab_root())
print('DataC:', datac_root())
PY
```

检查模型依赖：

```bash
python - <<'PY'
import timm, monai, einops
import mamba_ssm
print('RS3Mamba deps ok')
PY
```

---

## 5. 克隆、推送与「必含清单」

仓库根目录应包含：`.gitignore`、`README.md`、`RS3Mamba.md`、`SETUP.md`、`tools/`（电线入口）、`RS3Mamba/`（`model/RS3Mamba.py`、`model/SwinUMamba.py`、`utils_Mamba.py` 等**完整子目录**）。  

**不要提交**：`data/`（训练/测试输出）、`RS3Mamba/pretrain/*.pth`、`*.pkl`、数据集目录；这些已由根目录 **`.gitignore`** 忽略（推送前仍建议 `git status` 过目）。

### 5.1 他人克隆运行

```bash
git clone https://github.com/zsyyywm/my_SR3MAmba.git
cd my_SR3MAmba
# 然后按本文 §1–§4 与 RS3Mamba.md
```

### 5.2 推送前推荐 `git add`（避免漏文件）

在**工程根**执行（路径随你本机修改）：

```bash
cd /path/to/SSRS-main/SSRS-main

git add \
  .gitignore \
  README.md RS3Mamba.md SETUP.md \
  tools/ \
  RS3Mamba/

git status
# 若暂存区出现 data/、*.pth，应去掉：git reset HEAD data RS3Mamba/pretrain 等
```

说明：电线实验**仅依赖**上述目录；上游 SSRS 仓库中其它子项目（如 `MFNet/`）不必为复现本课题而提交，除非你希望镜像完整上游。

### 5.3 首次建库并推送（无历史时）

```bash
cd /path/to/SSRS-main/SSRS-main
git init
git checkout -b main
git remote add origin https://github.com/zsyyywm/my_SR3MAmba.git
git add .gitignore README.md RS3Mamba.md SETUP.md tools/ RS3Mamba/
git commit -m "Wire binary segmentation workflow for RS3Mamba (DataA/B/C)"
git push -u origin main
```

### 5.4 远端已有提交时

先 `git pull origin main --allow-unrelated-histories`（或 rebase）再合并推送；也可用新分支发 PR。已安装 **GitHub CLI** 时：`gh auth login` 后按提示操作。
