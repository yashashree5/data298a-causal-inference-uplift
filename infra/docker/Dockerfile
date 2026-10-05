# syntax=docker/dockerfile:1.7
FROM --platform=linux/amd64 python:3.9.23-slim-bookworm@sha256:7bffea15bcc3d7fb87cf10a027986203e4281e078fa2f5b234c30fca291f0834

ARG NUPLAN_DEVKIT_COMMIT=ce3c323af01c0d7ec5672f7832ef53f9c679aab0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NUPLAN_DEVKIT_ROOT=/opt/nuplan-devkit \
    NUPLAN_PYTHON=/usr/local/bin/python \
    NUPLAN_DATA_ROOT=/data \
    NUPLAN_MAPS_ROOT=/data/maps \
    NUPLAN_EXP_ROOT=/artifacts/nuplan

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
        libspatialindex-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY infra/docker/requirements.txt /tmp/requirements.txt

RUN python -m pip install --no-cache-dir \
        pip==23.2.1 \
        setuptools==59.5.0 \
        wheel==0.41.2 \
    && python -m pip install --no-cache-dir -r /tmp/requirements.txt \
    && python -m pip install --no-cache-dir pyinstrument==4.3.0 pytest==7.1.3 sympy==1.11.1

RUN git clone --filter=blob:none https://github.com/motional/nuplan-devkit.git "${NUPLAN_DEVKIT_ROOT}" \
    && git -C "${NUPLAN_DEVKIT_ROOT}" checkout "${NUPLAN_DEVKIT_COMMIT}" \
    && test "$(git -C "${NUPLAN_DEVKIT_ROOT}" rev-parse HEAD)" = "${NUPLAN_DEVKIT_COMMIT}" \
    && python -m pip install --no-deps -e "${NUPLAN_DEVKIT_ROOT}"

RUN useradd --create-home --uid 1000 --shell /bin/bash nuplan \
    && mkdir -p /workspace /data /artifacts \
    && chown -R nuplan:nuplan /workspace /artifacts

COPY infra/docker/entrypoint.sh /usr/local/bin/nuplan-entrypoint
RUN chmod 0755 /usr/local/bin/nuplan-entrypoint

WORKDIR /workspace
USER nuplan

ENTRYPOINT ["nuplan-entrypoint"]
CMD ["bash"]
