# for Jetson devices(need CUDA, ubuntu20.04)
FROM nvcr.io/nvidia/l4t-pytorch:r35.1.0-pth1.13-py3

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    build-essential \
    software-properties-common \
    libopenblas-dev \
    libpython3.8-dev \
    python3-pip \
    python3-dev \
    python3-setuptools \
    python3-wheel \
    libcurl4-openssl-dev \
    libssl-dev \
    sudo \
    wget \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# CMake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.25.1/cmake-3.25.1-linux-aarch64.sh \
 && chmod +x ./cmake-*.sh \
 && bash ./cmake-*.sh --prefix=/usr/local --exclude-subdir --skip-license \
 && rm ./cmake-*.sh

# onnxruntime-gpu, onnx
RUN git clone -b v1.13.1 --recursive https://github.com/microsoft/onnxruntime \
 && cd onnxruntime \
 && CUDACXX=/usr/local/cuda/bin/nvcc ./build.sh --config Release --update --build --parallel --build_wheel \
    --use_tensorrt --cuda_home /usr/local/cuda --cudnn_home /usr/lib/aarch64-linux-gnu \
    --tensorrt_home /usr/lib/aarch64-linux-gnu \
 && python3 -m pip install build/Linux/Release/dist/onnxruntime_gpu-*.whl \
 && cd .. \
 && rm -rf onnxruntime \
 && python3 -m pip install onnx \
 && rm -rf ~/.cache/pip

# for Jetson environment only
ENV LANG=en_US.UTF-8
ARG uid=1000
ARG gid=1000
RUN groupadd -r -f -g ${gid} nvidia \
 && useradd -o -r -l -u ${uid} -g ${gid} -ms /bin/bash nvidia \
 && usermod -aG sudo nvidia \
 && usermod -aG video nvidia \
 && echo 'nvidia:nvidia' | chpasswd \
 && touch /home/nvidia/.bash_profile \
 && chown nvidia:nvidia /home/nvidia/.bash_profile \
 && echo 'export LANG=en_US.UTF-8' >> /home/nvidia/.bash_profile \
 && echo 'export PATH=$PATH:/usr/local/cuda/bin' >> /home/nvidia/.bash_profile \
 && echo 'export PATH=$PATH:/home/nvidia/.local/bin' >> /home/nvidia/.bash_profile \
 && echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64' >> /home/nvidia/.bash_profile \
 && echo 'export TERM=xterm-256color' >> /home/nvidia/.bash_profile \
 && echo 'export PS1="\[\e[31m\]container\[\e[0m\]:\u@\W\$ "' >> /home/nvidia/.bash_profile
