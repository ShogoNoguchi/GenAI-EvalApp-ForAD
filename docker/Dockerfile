#/home/shogo/coding/eval/ucn_eval/docker/Dockerfile
# CUDA 12.8 ランタイム（NVIDIA公式）
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# 基本ツール + Tesseract（OCR）
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3 python3-pip python3-dev git wget curl ca-certificates \
      libgl1 libglib2.0-0 tesseract-ocr && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ベース依存（torch以外）— 最低限のみ。上位依存は起動時オーバーレイに委譲。
COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install -U pip && \
    python3 -m pip install -r /app/requirements.txt

# PyTorch (cu128固定) — RTX 5090 / CUDA 12.8 に厳密整合
RUN python3 -m pip install \
      torch==2.7.0 \
      torchvision==0.22.0 \
      torchaudio==2.7.0 \
      --index-url https://download.pytorch.org/whl/cu128

# スクリプト本体 & 既定エントリポイント
COPY eval_unicontrol_waymo.py /app/eval_unicontrol_waymo.py
COPY docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
