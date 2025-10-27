const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');
const readline = require('readline');

app.disableHardwareAcceleration();
app.commandLine.appendSwitch('disable-gpu');

let win = null;
let currentChild = null;

function createWindow() {
  win = new BrowserWindow({
    width: 1200,
    height: 880,
    backgroundColor: '#0b0f14',
    titleBarStyle: process.platform === 'darwin' ? 'hiddenInset' : 'default',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      sandbox: true,
      nodeIntegration: false,
      webSecurity: true,
      backgroundThrottling: false
    }
  });
  win.loadFile(path.join(__dirname, 'ui', 'index.html'));
}

const gotLock = app.requestSingleInstanceLock();
if (!gotLock) app.quit();
else {
  app.whenReady().then(() => {
    createWindow();
    app.on('activate', () => {
      if (BrowserWindow.getAllWindows().length === 0) createWindow();
    });
  });
}
app.on('window-all-closed', () => { if (process.platform !== 'darwin') app.quit(); });

function buildArgs(p) {
  const a = [];
  const kv = (f, v) => { if (v !== undefined && v !== null && String(v).trim() !== '') a.push(f, String(v)); };
  const fl = (f, c) => { if (c) a.push(f); };
  kv('--input', p.inputPath);
  kv('--format', p.format);
  kv('--popmap', p.popmapPath);
  kv('--prefix', p.prefix);
  kv('--config', p.yamlPath);
  kv('--dump-config', p.dumpConfigPath);
  kv('--preset', p.preset || 'fast');
  if (Array.isArray(p.models) && p.models.length) a.push('--models', ...p.models);
  if (Array.isArray(p.includePops) && p.includePops.length) a.push('--include-pops', ...p.includePops);
  fl('--tune', !!p.tune);
  if (p.tuneNTrials) kv('--tune-n-trials', p.tuneNTrials);
  kv('--batch-size', p.batchSize ?? 64);
  if (['cpu','cuda','mps'].includes(p.device || '')) kv('--device', p.device);
  if (p.nJobs) kv('--n-jobs', p.nJobs);
  if (['png','pdf','svg'].includes(p.plotFormat || '')) kv('--plot-format', p.plotFormat);
  if (p.seed) kv('--seed', p.seed);
  fl('--verbose', !!p.verbose);
  if (p.logFile) kv('--log-file', p.logFile);
  fl('--force-popmap', !!p.forcePopmap);
  fl('--dry-run', !!p.dryRun);
  (p.setPairs || []).forEach(s => { if (s && s.includes('=')) a.push('--set', s); });
  return a;
}

function checkCmd(cmd, args = ['--version']) {
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

// ---- IPC: always reply using event.sender; never use win.webContents ----
ipcMain.handle('detect:pgsui', async () => ({ ok: await checkCmd('pg-sui') }));

ipcMain.handle('pgsui:start', async (evt, payload) => {
  const send = (ch, m) => { try { evt?.sender?.send(ch, m); } catch {} };

  if (currentChild) return { ok: false, error: 'Process already running.' };
  if (!payload?.cwd || !String(payload.cwd).trim()) return { ok: false, error: 'Working directory is required.' };

  const args = buildArgs(payload);
  const cwd = payload.cwd;

  let fullArgv;
  if (payload.usePgSui === true) { // you likely removed this path; keeping for safety
    const cmd = 'pg-sui';
    fullArgv = [cmd, ...args];
    currentChild = spawn(cmd, args, { cwd, env: process.env });
  } else {
    const pythonExe = payload.pythonPath || 'python3';
    let cliPath = payload.cliPath;
    if (!cliPath) {
      const r = await resolveDefaultCli();
      if (r.ok) cliPath = r.path;
    }
    if (!cliPath || !fs.existsSync(cliPath)) return { ok: false, error: `cli.py not found at ${cliPath || '<auto>'}` };
    fullArgv = [pythonExe, cliPath, ...args];
    currentChild = spawn(pythonExe, [cliPath, ...args], { cwd, env: process.env });
  }

  // wire streams BEFORE returning
  const rlOut = readline.createInterface({ input: currentChild.stdout });
  rlOut.on('line', line => send('pgsui:log', { stream: 'stdout', line }));
  const rlErr = readline.createInterface({ input: currentChild.stderr });
  rlErr.on('line', line => send('pgsui:log', { stream: 'stderr', line }));

  currentChild.on('close', code => { send('pgsui:exit', { code }); currentChild = null; });
  currentChild.on('error', err => { send('pgsui:error', { message: String(err) }); currentChild = null; });

  send('pgsui:started', { argv: fullArgv, cwd });
  return { ok: true };
});

ipcMain.handle('pgsui:stop', async () => {
  if (!currentChild) return { ok: false, error: 'No process running.' };
  return new Promise((resolve) => {
    const child = currentChild; currentChild = null;
    const done = (ok, error) => resolve(ok ? { ok: true } : { ok: false, error });
    try {
      if (process.platform === 'win32') {
        const killer = spawn('taskkill', ['/PID', String(child.pid), '/T', '/F']);
        killer.on('close', (code) => done(code === 0));
      } else {
        child.once('close', () => done(true));
        child.kill('SIGTERM');
        setTimeout(() => { try { child.kill('SIGKILL'); } catch {} }, 5000);
      }
    } catch (e) { done(false, String(e)); }
  });
});

function resolveDefaultCliPathCandidates() {
  return [
    path.resolve(process.cwd(), '..', 'pgsui', 'cli.py'),
    path.resolve(__dirname, '..', '..', '..', 'pgsui', 'cli.py'),
    path.join(process.resourcesPath, 'pgsui', 'cli.py'),
  ];
}

async function resolveDefaultCli() {
  const envp = process.env.PGSUI_CLI_DEFAULT;
  if (envp && fs.existsSync(envp)) return { ok: true, path: envp };
  for (const pth of resolveDefaultCliPathCandidates()) {
    if (fs.existsSync(pth)) return { ok: true, path: pth };
  }
  return { ok: false };
}


// simple single-flight guard
let dialogBusy = false;
async function withDialogMutex(fn) {
  if (dialogBusy) return null;
  dialogBusy = true;
  try { return await fn(); }
  finally { dialogBusy = false; }
}

ipcMain.handle('pick:dir', async () => withDialogMutex(async () => {
  const res = await dialog.showOpenDialog({ properties: ['openDirectory'] });
  return (!res.canceled && res.filePaths[0]) ? res.filePaths[0] : null;
}));

ipcMain.handle('pick:file', async (_evt, { filters }) => withDialogMutex(async () => {
  const res = await dialog.showOpenDialog({
    properties: ['openFile'],
    filters: Array.isArray(filters) && filters.length ? filters : [{ name: 'All Files', extensions: ['*'] }],
  });
  return (!res.canceled && res.filePaths[0]) ? res.filePaths[0] : null;
}));

ipcMain.handle('pick:save', async (_evt, { defaultPath }) => withDialogMutex(async () => {
  const res = await dialog.showSaveDialog({ defaultPath });
  return (!res.canceled && res.filePath) ? res.filePath : null;
}));

ipcMain.handle('default:cli', async () => resolveDefaultCli());
