# syntax=docker/dockerfile:1.7
FROM python:3.12-slim

# ---- Env ----
ENV TZ=Etc/UTC \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ---- OS deps for Nextflow shells and basic tooling ----
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip python3-venv python-is-python3 bash coreutils findutils grep sed gawk curl \
        ca-certificates tini \
        && rm -rf /var/lib/apt/lists/*

# ---- Non-root user and writable workspace ----
RUN useradd -r -u 10001 -m appuser
WORKDIR /workspace
RUN chown -R appuser:appuser /workspace
VOLUME ["/workspace"]

# ---- Pin versions via build args ----
ARG PGSUI_VERSION=1.6.4

# ---- Security ----
USER appuser

SHELL ["/bin/bash", "-c"]

RUN python -m venv /home/appuser/venv \
    && source /home/appuser/venv/bin/activate \
    && python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install pg-sui \
    && pg-sui --version

CMD []
