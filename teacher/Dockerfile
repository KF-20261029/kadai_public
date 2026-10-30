FROM ghcr.io/prefix-dev/pixi:0.81.0@sha256:788ae451641666e2d1f79d3dbe35392dfc7e9b394b16a3acb75c347f3badb2ab

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates git build-essential libegl1 libgl1 libgles2 \
    && rm -rf /var/lib/apt/lists/*

# Register the NVIDIA EGL provider; its library is supplied by Container Toolkit.
RUN mkdir -p /usr/share/glvnd/egl_vendor.d \
    && printf '%s\n' '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' \
      > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

RUN mkdir -p /workspace/teacher/.pixi /cache/huggingface /cache/pixi \
    && chmod 1777 /workspace /workspace/teacher/.pixi /cache /cache/huggingface /cache/pixi

ENV HOME=/cache \
    HF_HOME=/cache/huggingface \
    PIXI_CACHE_DIR=/cache/pixi \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=8

WORKDIR /workspace
USER 1000:1000
CMD ["bash"]
