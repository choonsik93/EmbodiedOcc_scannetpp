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

## 3. Install some packages following [GaussianFormer](https://github.com/huang-yh/GaussianFormer)

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
git clone --recursive https://github.com/YkiWu/EmbodiedOcc.git
cd EmbodiedOcc
cd model/encoder/gaussianformer/ops && pip install -e .
cd model/head/gaussian_occ_head/ops/localagg && pip install -e .
```

## 4. Install the additional dependencies
```bash
cd EmbodiedOcc
pip install -r requirements.txt
```

## 5. Download Depth-Anything-V2 and make some slight changes
```bash
cd EmbodiedOcc
git clone https://github.com/DepthAnything/Depth-Anything-V2.git
```

**Folder structure**
```
EmbodiedOcc
├── ...
├── Depth-Anything-V2
```

Go to **Depth-Anything-V2/metric_depth/depth_anything_v2/dpt.py** and change the function **infer_image** in the class **DepthAnythingV2** as follows:
```Python
def infer_image(self, image, h_, w_, input_size=518):
    depth = self.forward(image)
    depth = F.interpolate(depth[:, None], (h_, w_), mode="bilinear", align_corners=True)[0, 0]
    return depth
```

## 6. Download EfficientNet-Pytorch
```bash
cd EmbodiedOcc
git clone https://github.com/lukemelas/EfficientNet-PyTorch.git
```

**Folder structure**
```
EmbodiedOcc
├── ...
├── Depth-Anything-V2
├── EfficientNet-Pytorch
```

## 7. Download our [finetuned checkpoint](https://huggingface.co/YkiWu/EmbodiedOcc) of Depth-Anything-V2 on Occ-ScanNet and put it under the **checkpoints**

**Folder structure**
```
EmbodiedOcc
├── ...
├── checkpoints/
│   ├── occscannet/
```