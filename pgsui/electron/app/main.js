const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');
const readline = require('readline');

app.disableHardwareAcceleration();

let win;
let currentChild = null;
let dialogBusy = false;

function createWindow() {
  win = new BrowserWindow({
    width: 1200,
    height: 880,
    backgroundColor: '#0b0f14',
    titleBarStyle: 'hiddenInset',
    webPreferences: { preload: path.join(__dirname, 'preload.js') }
  });
  win.loadFile(path.join(__dirname, 'renderer', 'index.html'));
}

app.whenReady().then(() => {
  createWindow();
  app.on('activate', () => { if (BrowserWindow.getAllWindows().length === 0) createWindow(); });
});
app.on('window-all-closed', () => { if (process.platform !== 'darwin') app.quit(); });

function buildArgs(payload) {
  const a = [];
  const pushKV = (flag, val) => { if (val !== undefined && val !== null && String(val).trim() !== '') a.push(flag, String(val)); };
  const pushFlag = (flag, cond) => { if (cond) a.push(flag); };

  pushKV('--input', payload.inputPath);
  pushKV('--format', payload.format);
  pushKV('--popmap', payload.popmapPath);
  pushKV('--prefix', payload.prefix);
  pushKV('--config', payload.yamlPath);
  pushKV('--dump-config', payload.dumpConfigPath);

  pushKV('--preset', payload.preset || 'fast');
  if (Array.isArray(payload.models) && payload.models.length) a.push('--models', ...payload.models);
  if (Array.isArray(payload.includePops) && payload.includePops.length) a.push('--include-pops', ...payload.includePops);

  pushFlag('--tune', !!payload.tune);
  if (payload.tuneNTrials) pushKV('--tune-n-trials', payload.tuneNTrials);
  pushKV('--batch-size', payload.batchSize ?? 64);

  if (payload.device && ['cpu','cuda','mps'].includes(payload.device)) pushKV('--device', payload.device);
  if (payload.nJobs) pushKV('--n-jobs', payload.nJobs);
  if (payload.plotFormat && ['png','pdf','svg'].includes(payload.plotFormat)) pushKV('--plot-format', payload.plotFormat);
  if (payload.seed) pushKV('--seed', payload.seed);
  pushFlag('--verbose', !!payload.verbose);
  if (payload.logFile) pushKV('--log-file', payload.logFile);
  pushFlag('--force-popmap', !!payload.forcePopmap);
  pushFlag('--dry-run', !!payload.dryRun);

  if (Array.isArray(payload.setPairs)) payload.setPairs.forEach(kv => { if (kv && kv.includes('=')) a.push('--set', kv); });
  return a;
}

function checkCmd(cmd, args = ['--help']) {
  return new Promise((resolve) => {
    try {
      const p = spawn(cmd, args, { env: process.env });
      let seen = false;
      p.stdout.on('data', () => { seen = true; });
      p.stderr.on('data', () => { seen = true; });
      p.on('error', () => resolve(false));
      p.on('close', () => resolve(seen));
    } catch { resolve(false); }
  });
}

// Prefer `pg-sui` console entry. Fallback: python + cli.py
ipcMain.handle('detect:pgsui', async () => ({ ok: await checkCmd('pg-sui') }));

ipcMain.handle('pgsui:start', async (_evt, payload) => {
  if (currentChild) return { ok: false, error: 'Process already running.' };
  if (!payload.cwd || !payload.cwd.trim()) return { ok: false, error: 'Working directory is required.' };

  const args = buildArgs(payload);
  const cwd = payload.cwd;

  let cmd;
  let fullArgv;

  if (payload.usePgSui !== false && await checkCmd('pg-sui')) {
    cmd = 'pg-sui';
    fullArgv = [cmd, ...args];
    currentChild = spawn(cmd, args, { cwd, env: process.env });
  } else {
    const pythonExe = payload.pythonPath || 'python3';
    const cliPath = payload.cliPath;
    if (!cliPath) return { ok: false, error: 'cli.py path is required when not using pg-sui.' };
    fullArgv = [pythonExe, cliPath, ...args];
    currentChild = spawn(pythonExe, [cliPath, ...args], { cwd, env: process.env });
  }

  const rlOut = readline.createInterface({ input: currentChild.stdout });
  rlOut.on('line', line => win.webContents.send('pgsui:log', { stream: 'stdout', line }));
  const rlErr = readline.createInterface({ input: currentChild.stderr });
  rlErr.on('line', line => win.webContents.send('pgsui:log', { stream: 'stderr', line }));

  currentChild.on('close', code => { win.webContents.send('pgsui:exit', { code }); currentChild = null; });
  currentChild.on('error', err => { win.webContents.send('pgsui:error', { message: String(err) }); currentChild = null; });

  win.webContents.send('pgsui:started', { argv: fullArgv, cwd });
  return { ok: true };
});

// --------- Dialogs: native first, AppleScript fallback on macOS, single-flight ---------
async function withDialogMutex(fn) {
  if (dialogBusy) return null;
  dialogBusy = true;
  try {
    await new Promise(r => setTimeout(r, 0));
    return await fn();
  } finally { dialogBusy = false; }
}

function runOsa(script, timeoutMs = 45000) {
  return new Promise((resolve) => {
    const p = spawn('osascript', ['-e', script]);
    let out = '';
    const timer = setTimeout(() => { try { p.kill('SIGKILL'); } catch {} resolve(null); }, timeoutMs);
    p.stdout.on('data', d => out += d.toString());
    p.on('error', () => { clearTimeout(timer); resolve(null); });
    p.on('close', code => { clearTimeout(timer); resolve(code === 0 && out.trim() ? out.trim() : null); });
  });
}

async function pickFileNative(filters) {
  try {
    const res = await dialog.showOpenDialog({ properties: ['openFile'], filters: filters || [{ name: 'All Files', extensions: ['*'] }] });
    if (!res.canceled && res.filePaths.length) return res.filePaths[0];
  } catch {}
  return null;
}
async function pickDirNative() {
  try {
    const res = await dialog.showOpenDialog({ properties: ['openDirectory'] });
    if (!res.canceled && res.filePaths.length) return res.filePaths[0];
  } catch {}
  return null;
}

ipcMain.handle('pick:file', async (_evt, { filters }) => withDialogMutex(async () => {
  const nat = await pickFileNative(filters);
  if (nat) return nat;
  if (process.platform === 'darwin') return await runOsa('POSIX path of (choose file with prompt "Select a file for PG-SUI")');
  return null;
}));
ipcMain.handle('pick:dir', async () => withDialogMutex(async () => {
  const nat = await pickDirNative();
  if (nat) return nat;
  if (process.platform === 'darwin') return await runOsa('POSIX path of (choose folder with prompt "Select working directory for PG-SUI")');
  return null;
}));
ipcMain.handle('pick:save', async (_evt, { defaultPath }) => withDialogMutex(async () => {
  try {
    const res = await dialog.showSaveDialog({ defaultPath });
    if (!res.canceled && res.filePath) return res.filePath;
  } catch {}
  if (process.platform === 'darwin') {
    const name = defaultPath && typeof defaultPath === 'string' ? path.basename(defaultPath) : 'pgsui-output.txt';
    return await runOsa(`POSIX path of (choose file name with prompt "Choose save path for PG-SUI" default name "${name}")`);
  }
  return null;
}));

// Default CLI path helper kept for advanced users
ipcMain.handle('default:cli', async () => {
  const envp = process.env.PGSUI_CLI_DEFAULT;
  if (envp && fs.existsSync(envp)) return { ok: true, path: envp };
  const candidate = path.resolve(__dirname, '..', '..', 'cli.py');
  if (fs.existsSync(candidate)) return { ok: true, path: candidate };
  const candidate2 = path.resolve(process.cwd(), 'pgsui', 'cli.py');
  if (fs.existsSync(candidate2)) return { ok: true, path: candidate2 };
  return { ok: false };
});
