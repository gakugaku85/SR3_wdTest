FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    python3-wheel \
    python3-setuptools \
    git \
    cifs-utils

# Pythonのデフォルトバージョンを3.12に設定
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# pipのインストール
# RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
#     && python3 get-pip.py \
#     && rm get-pip.py

RUN apt-get update --fix-missing && \
    apt-get install -y apt-utils && \
    apt-get install -y software-properties-common vim curl unzip htop openssh-server wget less procps cmake libboost-all-dev && \
    apt-get clean && apt-get autoremove -y  && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN pip3 install --upgrade pip setuptools
RUN pip3 install --no-cache-dir joblib numpy tqdm pillow scipy==1.13.0 joblib matplotlib scikit-image argparse SimpleITK pyyaml pandas pydicom scikit-learn natsort opencv-python-headless wandb lmdb gudhi tensorboardX pytest
# RUN pip3 install torch==1.13.1+cu117 torchvision==0.14.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
RUN pip3 install -U https://github.com/PythonOT/POT/archive/master.zip
