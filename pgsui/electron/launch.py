from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent / "app"


def _bin(path: Path, name: str) -> Path:
    # node_modules/.bin on POSIX, node_modules\\.bin on Windows
    d = path / ("node_modules/.bin" if os.name != "nt" else r"node_modules\.bin")
    exe = name + (".cmd" if os.name == "nt" else "")
    return d / exe


def main() -> int:
    if not APP_DIR.exists():
        print(f"[pgsui-gui] Missing Electron app at {APP_DIR}", file=sys.stderr)
        return 2

    # Resolve electron binary: local install first, else global npx fallback
    local_electron = _bin(APP_DIR, "electron")
    npx = shutil.which("npx")

    env = os.environ.copy()
    env.setdefault("PGSUI_PYTHON", sys.executable)
    env.setdefault("PGSUI_CLI_DEFAULT", str(Path(__file__).resolve().parents[1] / "cli.py"))

    try:
        if local_electron.exists():
            cmd = [str(local_electron), "."]
            proc = subprocess.Popen(cmd, cwd=str(APP_DIR), env=env)
        elif npx:
            # Uses Electron from registry on demand
            cmd = ["npx", "electron", "."]
            proc = subprocess.Popen(cmd, cwd=str(APP_DIR), env=env)
        else:
            print(
                "[pgsui-gui] Electron is not installed. Run: pgsui-gui-setup",
                file=sys.stderr,
            )
            return 2

        proc.wait()
        return proc.returncode or 0
    except KeyboardInterrupt:
        return 130
    except FileNotFoundError as e:
        print(f"[pgsui-gui] Failed to start Electron: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
