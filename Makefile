# Get version of CUDA and enable it for compilation if CUDA > 11.0
# This solves https://github.com/IDEA-Research/Grounded-Segment-Anything/issues/53
# and https://github.com/IDEA-Research/Grounded-Segment-Anything/issues/84
# when running in Docker
# Check if nvcc is installed
NVCC := $(shell which nvcc)
ifeq ($(NVCC),)
	# NVCC not found
	USE_CUDA := 0
	NVCC_VERSION := "not installed"
else
	NVCC_VERSION := $(shell nvcc --version | grep -oP 'release \K[0-9.]+')
	USE_CUDA := $(shell echo "$(NVCC_VERSION) > 11" | bc -l)
endif

# Add the list of supported ARCHs
ifeq ($(USE_CUDA), 1)
	TORCH_CUDA_ARCH_LIST := "3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"
	BUILD_MESSAGE := "I will try to build the image with CUDA support"
else
	TORCH_CUDA_ARCH_LIST :=
	BUILD_MESSAGE := "CUDA $(NVCC_VERSION) is not supported"
endif

build-image:
	@echo $(BUILD_MESSAGE)
	docker build --build-arg USE_CUDA=$(USE_CUDA) \
	--build-arg TORCH_ARCH=$(TORCH_CUDA_ARCH_LIST) \
	-t embodiedocc:latest .
	docker run -d --gpus all -it --rm --net=host \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-v "${PWD}":/home/appuser/EmbodiedOcc \
	-w /home/appuser/EmbodiedOcc \
	-e DISPLAY=$(DISPLAY) \
	--name=embodiedocc \
	--ipc=host embodiedocc:latest
	docker exec -it embodiedocc sh -c "cd EfficientNet-PyTorch && pip install -e ."
	docker commit embodiedocc embodiedocc:latest
	docker stop embodiedocc

run:
	docker run -it --rm -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$(DISPLAY) -e USER=$(USER) \
	-e runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=all -e NVIDIA_VISIBLE_DEVICES=all \
	-e PYTHONPATH=/home/appuser/EmbodiedOcc \
	-v "${PWD}":/home/appuser/EmbodiedOcc \
	-e PYTHONPATH=/home/appuser/EmbodiedOcc/Depth-Anything-V2:$${PYTHONPATH} \
	-w /home/appuser/EmbodiedOcc \
	-v "${SCANNET_PATH}":/data \
	--shm-size 128G \
	--net host --gpus all --privileged --name embodiedocc embodiedocc:latest /bin/bash

# docker exec -it embodiedocc sh -c "cd model/encoder/gaussianformer/ops && pip install -e ."
# docker exec -it embodiedocc sh -c "cd model/head/gaussian_occ_head/ops/localagg && pip install -e ."
# git clone https://github.com/DepthAnything/Depth-Anything-V2.git
# git clone https://github.com/lukemelas/EfficientNet-PyTorch.git
# docker exec -it embodiedocc sh -c "cd model/encoder/gaussianformer/ops && pip install -e ."
# docker exec -it embodiedocc sh -c "cd model/head/gaussian_occ_head/ops/localagg && pip install -e ."

# export SCANNET_PATH=/media/sequor/PortableSSD/scannetpp && make run
# torchrun --nproc_per_node=1 train_mono_scannetpp.py --py-config config/train_mono_config.py

# export SCANNET_PATH=/data/scannetpp && make run
# torchrun --nproc_per_node=8 train_mono_scannetpp.py --py-config config/train_mono_config.py

# userful commands
# xhost +Local:*  && xhost
# sudo chown -R $USER: $HOME

# models/detectors/sliceformer_occ.py 의 visualizer 주석 제거
