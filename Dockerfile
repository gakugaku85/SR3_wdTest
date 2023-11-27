FROM nvidia/cuda:10.1-cudnn8-devel-ubuntu18.04

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-pip \
    python3-wheel \
    python3-setuptools \
    git \
    cifs-utils \
    && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN apt-get update --fix-missing && \
    apt-get install -y apt-utils && \
    apt-get install -y software-properties-common vim curl unzip htop openssh-server wget procps

RUN pip3 install --upgrade pip setuptools
RUN pip3 install --no-cache-dir joblib numpy tqdm pillow scipy joblib matplotlib scikit-image argparse SimpleITK pyyaml pandas pydicom scikit-learn
RUN pip3 install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html


RUN mkdir /var/run/sshd
RUN echo 'root:gakugaku' | chpasswd
# RUN echo '\nPermitRootLogin yes' >> /etc/ssh/sshd_config
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

# SSH接続のための設定です。
ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

# runしたときにSSH接続を始められるようにします。
EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]