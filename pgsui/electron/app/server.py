from __future__ import annotations

import asyncio
import os
import signal
import shutil
from pathlib import Path

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent / "ui"
WORK = Path("/work")
app = FastAPI()
_proc = None
_log_queue: asyncio.Queue[str] = asyncio.Queue()


def _build_args(p: dict[str, str | int | bool | list[str] | None]) -> list[str]:
    a: list[str] = []

    def kv(flag, v):
        if v not in (None, "", []):
            a.extend([flag, str(v)])

    def fl(flag, cond):
        if cond:
            a.append(flag)

    kv("--input", p.get("inputPath"))
    kv("--format", p.get("format"))
    kv("--popmap", p.get("popmapPath"))
    kv("--prefix", p.get("prefix"))
    kv("--config", p.get("yamlPath"))
    kv("--dump-config", p.get("dumpConfigPath"))
    kv("--preset", p.get("preset") or "fast")
    kv("--sim-strategy", p.get("simStrategy"))
    m = p.get("models") or []
    if m:
        a.extend(["--models", *m])
    inc = p.get("includePops") or []
    if inc:
        a.extend(["--include-pops", *inc])
    fl("--tune", bool(p.get("tune")))
    if p.get("tuneNTrials"):
        kv("--tune-n-trials", p["tuneNTrials"])
    kv("--batch-size", p.get("batchSize") or 64)
    dev = p.get("device")
    if dev in ("cpu", "cuda", "mps"):
        kv("--device", dev)
    if p.get("nJobs"):
        kv("--n-jobs", p["nJobs"])
    pf = p.get("plotFormat")
    if pf in ("png", "pdf", "svg"):
        kv("--plot-format", pf)
    if p.get("seed"):
        kv("--seed", p["seed"])
    fl("--verbose", bool(p.get("verbose")))
    if p.get("logFile"):
        kv("--log-file", p["logFile"])
    fl("--force-popmap", bool(p.get("forcePopmap")))
    fl("--dry-run", bool(p.get("dryRun")))
    for kvp in p.get("setPairs") or []:
        if isinstance(kvp, str) and "=" in kvp:
            a.extend(["--set", kvp])
    return a


@app.get("/api/status")
def status():
    return {"running": _proc is not None}


@app.get("/api/ls")
def ls(path: str = Query("/work")):
    p = Path(path).resolve()
    if not str(p).startswith(str(WORK)):
        return {"ok": False, "error": "out_of_root"}
    items = [
        {"name": e.name, "path": str(e), "dir": e.is_dir()} for e in sorted(p.iterdir())
    ]
    return {"ok": True, "cwd": str(p), "items": items}


@app.post("/api/start")
async def start(payload: dict):
    global _proc
    if _proc:
        return {"ok": False, "error": "already_running"}
    cwd = Path(payload.get("cwd") or "/work").resolve()
    if not str(cwd).startswith(str(WORK)):
        return {"ok": False, "error": "cwd_must_be_under_/work"}
    args = _build_args(payload)
    use_pg = bool(payload.get("usePgSui", True))
    if use_pg:
        cmd = ["pg-sui", *args]
    else:
        py_candidates = [
            payload.get("pythonPath"),
            os.environ.get("PGSUI_PYTHON"),
            "/opt/homebrew/bin/python3.12",
            "/usr/local/bin/python3.12",
            "python3.12",
            "python3",
            "python",
        ]
        py = next((c for c in py_candidates if c and shutil.which(c)), "python3")
        cli = payload.get("cliPath")
        if not cli:
            return {"ok": False, "error": "cli_path_required"}
        cmd = [py, cli, *args]
    _proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=os.environ.copy(),
    )

    async def pump(stream, tag):
        async for line in stream:
            await _log_queue.put(f"{tag}|{line.decode(errors='ignore').rstrip()}")

    asyncio.create_task(pump(_proc.stdout, "stdout"))
    asyncio.create_task(pump(_proc.stderr, "stderr"))
    return {"ok": True, "argv": cmd, "cwd": str(cwd)}


@app.post("/api/stop")
async def stop():
    global _proc
    if not _proc:
        return {"ok": False, "error": "not_running"}
    try:
        _proc.send_signal(signal.SIGTERM)
        try:
            await asyncio.wait_for(_proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            _proc.kill()
    finally:
        _proc = None
    return {"ok": True}


@app.websocket("/api/logs")
async def logs(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            msg = await _log_queue.get()
            await ws.send_text(msg)
    except WebSocketDisconnect:
        return


app.mount("/", StaticFiles(directory=ROOT, html=True), name="ui")
