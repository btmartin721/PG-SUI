from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent / "app"


def _which(cmd: str) -> str | None:
    return shutil.which(cmd)


def main() -> int:
    if not APP_DIR.exists():
        print(f"[pgsui-gui-setup] Missing Electron app at {APP_DIR}", file=sys.stderr)
        return 2

    node = _which("node")
    npm = _which("npm")
    if not node or not npm:
        print(
            "[pgsui-gui-setup] Node.js and npm are required. Install from https://nodejs.org/",
            file=sys.stderr,
        )
        return 2

    # Prefer deterministic installs if package-lock.json exists
    lock = APP_DIR / "package-lock.json"
    cmd = ["npm", "ci"] if lock.exists() else ["npm", "install"]
    try:
        print(f"[pgsui-gui-setup] Running: {' '.join(cmd)} in {APP_DIR}")
        subprocess.check_call(cmd, cwd=str(APP_DIR))
        print("[pgsui-gui-setup] Done.")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"[pgsui-gui-setup] Failed: {e}", file=sys.stderr)
        return e.returncode


if __name__ == "__main__":
    raise SystemExit(main())
