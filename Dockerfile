FROM nvidia/cuda:13.0.0-runtime-ubuntu22.04

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 libgomp1 \
    libzbar0 ffmpeg poppler-utils \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python 依赖：先装 PaddlePaddle GPU (cu130)，再装其余依赖
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
    && pip install paddlepaddle-gpu==3.3.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu130/ \
    && pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 \
    && pip install -r requirements.txt

COPY . .

# 屏蔽 TensorFlow CPU/oneDNN INFO 日志（TF 仅用于 VGG16 轻量推理）
ENV TF_CPP_MIN_LOG_LEVEL=2

# onnxruntime-gpu 需要 CUDA 12 运行时库（镜像 CUDA 13，需单独装 cu12 库）
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/cublas/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}

CMD ["python3", "-u", "index.py"]