from nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04

run apt-get update && apt-get install -y \
    python3\
    python3-pip \
    && rm -rf /var/lib/apt/lists/*
run pip3 install --upgrade pip
run pip3 install --upgrade cython
run pip3 install \
    torch==1.2\
    torchvision==0.4.0\
    scipy\
    pillow==6.2.1\
    sklearn\
    tqdm\
    torchsummary\
    matplotlib\
    opencv-python-headless\
    pandas\
    scikit-image

workdir /home/zhengwei
cmd ["/bin/bash"]
