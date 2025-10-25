# syntax=docker/dockerfile:1.6
FROM node:20-bullseye

# Electron + Python + noVNC deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    build-essential ca-certificates git curl tini \
    libgtk-3-0 libnss3 libasound2 libx11-xcb1 libxcomposite1 \
    libxrandr2 libxdamage1 libxfixes3 libdrm2 libgbm1 libxrender1 \
    libxi6 libxtst6 \
    xvfb x11vnc novnc websockify openbox \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1 \
    ELECTRON_DISABLE_SANDBOX=1 ELECTRON_ENABLE_LOGGING=1

# PG-SUI from PyPI with GUI extra
RUN python3 -m pip install --upgrade pip wheel setuptools && \
    python3 -m pip install "pg-sui[gui]==1.6.3"

# Electron app
WORKDIR /app
COPY pgsui/electron/app/ /app/
RUN npm ci || npm install

# Start script
COPY --chmod=755 <<'SH' /usr/local/bin/start.sh
#!/usr/bin/env bash
set -euo pipefail
MODE="${MODE:-novnc}"          # novnc | x11
export DISPLAY=${DISPLAY:-:0}
if [[ "$MODE" == "novnc" ]]; then
    Xvfb "$DISPLAY" -screen 0 1280x800x24 -nolisten tcp &
    xvfb_pid=$!
    openbox >/dev/null 2>&1 &
    x11vnc -display "$DISPLAY" -rfbport 5900 -forever -shared -nopw -quiet &
    websockify --web=/usr/share/novnc/ 6080 localhost:5900 &
    (cd /app && npx electron .)
    kill $xvfb_pid || true
else
    # Linux native X11. Run with: -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix
    (cd /app && npx electron .)
fi
SH

EXPOSE 6080
ENTRYPOINT ["tini","--","/usr/local/bin/start.sh"]
