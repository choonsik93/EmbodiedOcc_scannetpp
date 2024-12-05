# Installation
Our code is based on the following environment.

## 1. Create conda environment
```bash
conda create -n embodiedocc python=3.8.19
conda activate embodiedocc
```

## 2. Install PyTorch
```bash
pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 --index-url https://download.pytorch.org/whl/cu113
```

## 3. Install packages following [GaussianFormer](https://github.com/huang-yh/GaussianFormer)

### 1. Install packages from MMLab
```bash
pip install openmim==0.3.9
mim install mmcv==2.1.0
mim install mmdet==3.3.0
mim install mmsegmentation==1.2.2
mim install mmdet3d==1.1.1
```

### 2. Install other packages
```bash
pip install spconv-cu114==2.3.6
pip install timm
```

### 3. Install custom CUDA ops
```bash
cd model/encoder/gaussianformer/ops && pip install -e .
cd model/head/gaussian_occ_head/ops/localagg && pip install -e .