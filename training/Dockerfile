FROM ghcr.io/prefix-dev/pixi:0.81.0@sha256:788ae451641666e2d1f79d3dbe35392dfc7e9b394b16a3acb75c347f3badb2ab

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates git build-essential libegl1 libgl1 libgles2 \
    && rm -rf /var/lib/apt/lists/*

ARG VLASH_REVISION=22cbabfee0f57874987c75a35a7dac129e695db0
ARG RELEASE_REVISION=73167f4dba88633aeec95ba7a3b3881916fa9aaa
RUN git clone https://github.com/mit-han-lab/vlash.git /opt/vlash \
    && git -C /opt/vlash checkout "${VLASH_REVISION}" \
    && git clone https://github.com/jinnymo/so101-pi05-base.git /opt/so101-pi05-base \
    && git -C /opt/so101-pi05-base checkout "${RELEASE_REVISION}" \
    && cp /opt/so101-pi05-base/training-docker/patched/modeling_pi05.py /opt/vlash/vlash/policies/pi05/modeling_pi05.py \
    && cp /opt/so101-pi05-base/training-docker/patched/run.py /opt/vlash/vlash/run.py \
    && cp /opt/so101-pi05-base/training-docker/patched/train.py /opt/vlash/vlash/train.py

RUN mkdir -p /usr/share/glvnd/egl_vendor.d \
    && printf '%s\n' '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' \
    > /usr/share/glvnd/egl_vendor.d/10_nvidia.json \
    && chown -R 1000:1000 /opt/vlash /opt/so101-pi05-base \
    && mkdir -p /workspace/training/.pixi /cache/huggingface /cache/pixi \
    && chmod 1777 /workspace /workspace/training/.pixi /cache /cache/huggingface /cache/pixi

ENV HOME=/cache HF_HOME=/cache/huggingface PIXI_CACHE_DIR=/cache/pixi \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=4 VLASH_PALIGEMMA_PATH=/cache/tokenizer

WORKDIR /workspace
USER 1000:1000
CMD ["bash"]
