FROM ghcr.io/astral-sh/uv:0.9.5@sha256:f459f6f73a8c4ef5d69f4e6fbbdb8af751d6fa40ec34b39a1ab469acd6e289b7 AS uv
FROM ros:jazzy-ros-base-noble@sha256:066420e07f60aa18262f2479981def87ebcfcec42eefb0c0c57c4a46098348ca

ENV DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_CONCURRENT_DOWNLOADS=4 \
    UV_CONCURRENT_INSTALLS=4 \
    UV_CONCURRENT_BUILDS=2

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates git build-essential python3-venv ffmpeg \
      libegl1 libgl1 libgles2 \
      ros-jazzy-sensor-msgs ros-jazzy-trajectory-msgs ros-jazzy-control-msgs \
      ros-jazzy-cv-bridge ros-jazzy-tf2-ros ros-jazzy-rosbag2 \
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
      > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /validation
RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-jazzy-mujoco-ros2-control ros-jazzy-mujoco-ros2-control-demos ros-jazzy-joint-trajectory-controller \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock /validation/
RUN --mount=type=cache,target=/root/.cache/uv \
    python3 -m venv --system-site-packages /opt/venv \
    && uv sync --locked --no-install-project --group validation --python /usr/bin/python3.12 \
    && cp uv.lock /opt/dependencies.lock \
    && cp /opt/so101-pi05-base/training-docker/patched/video_utils.py \
      /opt/venv/lib/python3.12/site-packages/lerobot/datasets/video_utils.py

ENV PATH=/opt/venv/bin:$PATH \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    VLASH_PALIGEMMA_PATH=/cache/tokenizer \
    ROS_DOMAIN_ID=85 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 \
    HOME=/tmp/ros2-pi05-home

RUN mkdir -p "$HOME" && chmod 1777 "$HOME"
WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
      xfce4 xfce4-terminal terminator mousepad dbus-x11 xvfb x11vnc \
      novnc websockify supervisor x11-utils scrot fonts-noto-cjk \
      ros-jazzy-rqt-image-view ros-jazzy-rqt-graph \
    && rm -rf /var/lib/apt/lists/*

ENV HOME=/home/ubuntu \
    XDG_RUNTIME_DIR=/home/ubuntu/.runtime \
    XDG_SESSION_TYPE=x11 XDG_CURRENT_DESKTOP=XFCE \
    QT_X11_NO_MITSHM=1

RUN mkdir -p /tmp/.X11-unix "$HOME/.runtime" "$HOME/.config/autostart" \
    && chmod 1777 /tmp/.X11-unix \
    && chmod 700 "$HOME/.runtime" \
    && printf '%s\n' 'source /opt/ros/jazzy/setup.bash' 'export PATH=/opt/venv/bin:$PATH' >> "$HOME/.bashrc" \
    && chown -R 1000:1000 "$HOME"

COPY desktop/supervisord.conf /etc/supervisor/desktop.conf
COPY --chown=1000:1000 desktop/terminal.desktop /home/ubuntu/.config/autostart/terminal.desktop
EXPOSE 6080
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/desktop.conf"]

USER root
RUN apt-get update && apt-get install -y --no-install-recommends patchelf \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch 0.0.7 https://github.com/pal-robotics/mujoco_vendor.git /tmp/mujoco_vendor \
    && test "$(git -C /tmp/mujoco_vendor rev-parse HEAD)" = e4c679d922fdf977be8625fd770020b955d8cc62 \
    && sed -i 's/set(MUJOCO_VERSION "3.4.0")/set(MUJOCO_VERSION "3.3.7")/' /tmp/mujoco_vendor/CMakeLists.txt \
    && grep -q 'set(MUJOCO_VERSION "3.3.7")' /tmp/mujoco_vendor/CMakeLists.txt \
    && git clone --depth 1 --branch 0.1.1 https://github.com/ros-controls/mujoco_ros2_control.git /tmp/mujoco_ros2_control \
    && test "$(git -C /tmp/mujoco_ros2_control rev-parse HEAD)" = d5b1f019dc3b33f3d0270450506bd6e72bea1913
RUN bash -c 'source /opt/ros/jazzy/setup.bash && MAKEFLAGS=-j2 python /usr/bin/colcon --log-base /tmp/vendor-log build --base-paths /tmp/mujoco_vendor --build-base /tmp/vendor-build --install-base /opt/mujoco337/vendor --packages-select mujoco_vendor --parallel-workers 1 --cmake-args -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release'
RUN bash -c 'source /opt/ros/jazzy/setup.bash && source /opt/mujoco337/vendor/setup.bash && MAKEFLAGS=-j2 python /usr/bin/colcon --log-base /tmp/control-log build --base-paths /tmp/mujoco_ros2_control --build-base /tmp/control-build --install-base /opt/mujoco337/control --packages-select mujoco_3d_lidar mujoco_ros2_control_plugins mujoco_ros2_control --parallel-workers 1 --cmake-args -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release -DMUJOCO_DEP_VERSION_lodepng=e584a8490db1b76b81bf298c11c09d3b55d9c270'
RUN printf '%s\n' 'source /opt/mujoco337/vendor/setup.bash' 'source /opt/mujoco337/control/setup.bash' >> /home/ubuntu/.bashrc
RUN apt-get update && apt-get install -y --no-install-recommends ros-jazzy-joint-trajectory-controller \
    && rm -rf /var/lib/apt/lists/*
RUN printf '%s\n' '#!/bin/bash' 'set -e' \
      'source /opt/ros/jazzy/setup.bash' \
      'source /opt/mujoco337/vendor/setup.bash' \
      'source /opt/mujoco337/control/setup.bash' \
      'exec "$@"' > /ros_entrypoint.sh

# Keep scoring physics separate from the inference metadata dependency.
RUN uv pip install --target /opt/evaluation_deps --no-deps mujoco==3.3.7
ENV EVAL_PYTHONPATH=/opt/evaluation_deps

# A new Compose cache must be writable by the non-root runtime user.
RUN mkdir -p /cache && chmod 1777 /cache
