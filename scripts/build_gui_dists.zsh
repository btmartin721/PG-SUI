#!/usr/bin/env zsh
# Build GUI installers: DMG on macOS, .deb on Linux (run on the respective OS).

set -euo pipefail

ROOT="$(cd "$(dirname "${0:A}")/.." && pwd)"
APP_DIR="$ROOT/pgsui/electron/app"
EXTRA_RES="$APP_DIR/extra-resources"

echo "[pgsui] Building GUI from: $APP_DIR"

if [[ ! -d "$APP_DIR" ]]; then
  echo "Electron app directory not found at $APP_DIR" >&2
  exit 2
fi

mkdir -p "$EXTRA_RES"

pushd "$APP_DIR" >/dev/null

if [[ ! -d node_modules ]]; then
  echo "[pgsui] Installing node dependencies (npm ci)..."
  npm ci
fi

# Optionally bundle a Python virtual environment for offline deps.
if [[ -n "${PGSUI_BUNDLE_VENV:-}" ]]; then
  if [[ -d "$PGSUI_BUNDLE_VENV" ]]; then
    echo "[pgsui] Bundling virtualenv from $PGSUI_BUNDLE_VENV"
    rm -rf "$EXTRA_RES/venv"
    mkdir -p "$EXTRA_RES"
    rsync -a "$PGSUI_BUNDLE_VENV"/ "$EXTRA_RES/venv"/
  else
    echo "[pgsui] WARNING: PGSUI_BUNDLE_VENV is set but not a directory: $PGSUI_BUNDLE_VENV" >&2
  fi
else
  if [[ -d "$EXTRA_RES/venv" ]]; then
    echo "[pgsui] Removing stale bundled venv (no PGSUI_BUNDLE_VENV set)."
    rm -rf "$EXTRA_RES/venv"
  fi
  echo "[pgsui] Skipping bundled virtualenv (set PGSUI_BUNDLE_VENV to include one)."
fi

platform_args=()
case "$OSTYPE" in
  darwin*) platform_args=(--mac dmg) ;;
  linux*)  platform_args=(--linux deb) ;;
  *) echo "Unsupported platform: $OSTYPE" >&2; exit 3 ;;
esac

echo "[pgsui] Running electron-builder ${platform_args[*]:-(default target)}"
npx electron-builder --projectDir "$APP_DIR" "${platform_args[@]}"

echo "[pgsui] Done. Artifacts are under $APP_DIR/dist"

popd >/dev/null
