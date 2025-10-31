FROM nvidia/cuda:11.8.0-devel-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get -y upgrade

# install python3.9
RUN apt-get install -y keyboard-configuration
RUN apt install -y python3.9 python3.9-dev
RUN apt-get update && apt-get -y install wget python3-pip build-essential git curl lsb-release

# symlink
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1
RUN ln -s -f /usr/bin/python /usr/bin/python3

# Dependency
RUN echo "export CUDA_HOME=/usr/local/cuda && export PATH=/usr/local/cuda-11.8/bin:$PATH && LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:/lib64:$LD_LIBRARY_PATH" >> ~/.bashrc
RUN pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 qt5-default libsparsehash-dev libopenblas-dev

RUN python -m pip install --upgrade pip

RUN printf "numpy==1.26.4\n" > /etc/pip-constraints.txt
ENV PIP_CONSTRAINT=/etc/pip-constraints.txt
ENV PIP_UPGRADE_STRATEGY=only-if-needed

RUN pip install open3d==0.18.0
RUN pip install spconv-cu118

RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
RUN pip install torch-geometric -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

# OpenMMLab
RUN pip install mmengine
RUN pip install -U --no-deps "mmcv==2.1.0" -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
RUN pip install "mmdet==3.3.0"
RUN pip install "mmsegmentation>=1.0.0,<1.3.0"
RUN pip install "mmdet3d==1.4.0"

# Source build
RUN pip install --no-cache-dir --no-build-isolation git+https://github.com/mit-han-lab/torchsparse@v1.4.0
RUN pip install --no-cache-dir --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"
RUN pip install --no-cache-dir --no-build-isolation -U git+https://github.com/NVIDIA/MinkowskiEngine --no-deps

RUN pip install ftfy regex numba

RUN pip install timm vtk einops

RUN apt-get install -y python3-tk