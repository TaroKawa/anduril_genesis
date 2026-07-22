# Genesis + PyTorch(cu128) 学習用イメージ。
# ドライバはCUDA 13.1(前方互換)なので cu128 ランタイムで動く。
# レンダリングはヘッドレスEGL(NVIDIA GLVND)。mesa-vulkan-drivers は入れない
# (CPUのlavapipeがNVIDIA Vulkanを覆い隠す事故を防ぐ)。
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates \
        libegl1 libgl1 libglvnd0 libgles2 libglib2.0-0 libx11-6 libxext6 \
        libvulkan1 ffmpeg xvfb \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.2 /uv /uvx /usr/local/bin/

ENV UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    PYTHONUNBUFFERED=1 \
    PYOPENGL_PLATFORM=egl \
    NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility,video

WORKDIR /app

# 依存を先に解決してレイヤキャッシュを効かせる(コードは実行時bind mount)
COPY pyproject.toml uv.lock ./
RUN uv python install 3.12 && uv sync --frozen --no-install-project --no-dev

COPY . .
RUN uv sync --frozen --no-dev

CMD ["sleep", "infinity"]
