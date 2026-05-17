# SSRS / RS3Mamba 电线实验环境配置

> 本文件只放环境、依赖、数据路径与自检。训练/测试命令见 [`RS3Mamba.md`](RS3Mamba.md)，项目介绍与结果记录见 [`README.md`](README.md)。  
> **GitHub：** [https://github.com/zsyyywm/my_SR3MAmba](https://github.com/zsyyywm/my_SR3MAmba)

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

## 5. 克隆本仓库与推送到 GitHub

在**本地或服务器**克隆（若仓库已初始化并有内容）：

```bash
git clone https://github.com/zsyyywm/my_SR3MAmba.git
cd my_SR3MAmba
```

若你希望把当前 `SSRS-main` 工程树作为**首次**推送内容，在工程根目录（含 `tools/`、`RS3Mamba/`、`README.md` 等）执行：

```bash
cd /path/to/SSRS-main/SSRS-main   # 以你的实际路径为准
git init
git checkout -b main
git remote add origin https://github.com/zsyyywm/my_SR3MAmba.git
# 大文件：可选添加 .gitignore 排除 data/、*.pth、__pycache__/ 等后再提交
git add README.md RS3Mamba.md SETUP.md tools/ RS3Mamba/
git commit -m "Wire binary segmentation workflow for RS3Mamba (DataA/B/C)"
git push -u origin main
```

若 GitHub 上**已有 README/提交**，先 `git pull origin main --allow-unrelated-histories` 再合并，或改用新分支发 PR。已安装 **GitHub CLI** 时亦可用：`gh auth login` 后按提示操作。
