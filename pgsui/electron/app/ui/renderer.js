/* ---- helpers (define first) ---- */
const $ = (id) => document.getElementById(id);
const logEl = $('log');

function appendLog({ stream, line }) {
  const span = document.createElement('span');
  if (stream === 'stderr') span.className = 'err';
  span.textContent = line + '\n';
  logEl.appendChild(span);
  if (logEl.textContent.length > 2_000_000) logEl.textContent = logEl.textContent.slice(-1_000_000);
  logEl.scrollTop = logEl.scrollHeight;
}

/* ---- payload ---- */
function collectPayload() {
  const modelsSel = Array.from($('models').selectedOptions).map(o => o.value);
  const includePops = $('includePops').value.trim().split(/\s+/).filter(Boolean);
  const setPairs = $('setPairs').value.split('\n').map(s => s.trim()).filter(Boolean);
  const bs = $('batchSize').value ? Number($('batchSize').value) : 64;
  return {
    pythonPath: undefined,
    cliPath: undefined,
    cwd: $('cwd').value.trim(),
    inputPath: $('inputPath').value.trim(),
    format: $('format').value,
    popmapPath: $('popmapPath').value.trim() || undefined,
    prefix: $('prefix').value.trim() || undefined,
    yamlPath: $('yamlPath').value.trim() || undefined,
    dumpConfigPath: $('dumpConfigPath').value.trim() || undefined,
    preset: $('preset').value || 'fast',
    models: modelsSel,
    includePops,
    device: $('device').value || undefined,
    batchSize: bs,
    nJobs: $('nJobs').value ? Number($('nJobs').value) : undefined,
    plotFormat: $('plotFormat').value || undefined,
    seed: $('seed').value.trim() || undefined,
    verbose: $('verbose').checked,
    forcePopmap: $('forcePopmap').checked,
    dryRun: $('dryRun').checked,
    setPairs,
    logFile: $('logFile').value.trim() || undefined,
    tune: $('tune').checked,
    tuneNTrials: $('tune').checked ? (Number($('tuneNTrials').value) || 50) : undefined
  };
}

function setRunningUI(isRunning) {
  $('start').disabled = isRunning;
  $('stop').disabled = !isRunning;
  document.querySelectorAll('button').forEach(b => { if (b.id !== 'stop') b.disabled = isRunning; });
}

/* ---- init small UI bits ---- */
(() => {
  const tune = $('tune'), tuneOpts = $('tuneOpts');
  if (tune && tuneOpts) tune.addEventListener('change', () => { tuneOpts.style.display = tune.checked ? '' : 'none'; });
  const logoEl = $('logo');
  if (logoEl) logoEl.addEventListener('error', () => { logoEl.style.display = 'none'; });
})();

/* ---- events (wire exactly once) ---- */
const on = (id, ev, fn) => { const el = $(id); if (el) el.addEventListener(ev, fn); };

on('pickCwd', 'click', async () => {
  const d = await window.pgsui.pickDir();
  if (d) $('cwd').value = d;
});

on('start', 'click', async () => {
  try {
    if (!window.pgsui) { appendLog({ stream:'stderr', line:'Bridge missing (preload).' }); return; }
    logEl.textContent = '';
    const payload = collectPayload();
    if (!payload.cwd) { appendLog({ stream:'stderr', line:'Working directory is required.' }); return; }
    if (!payload.inputPath) { appendLog({ stream:'stderr', line:'Input file required.' }); return; }
    if (!payload.cliPath) {
      const r = await window.pgsui.defaultCli?.();
      if (r?.ok && r.path) payload.cliPath = r.path;
    }
    if (!payload.cliPath) { appendLog({ stream:'stderr', line:'Could not find <project_root>/pgsui/cli.py. Set PGSUI_CLI_DEFAULT or adjust repo layout.' }); return; }
    const res = await window.pgsui.start(payload);
    if (!res?.ok) appendLog({ stream:'stderr', line:`Start failed: ${res?.error || 'unknown error'}` });
  } catch (e) {
    appendLog({ stream:'stderr', line:`Start exception: ${e?.message || String(e)}` });
  }
});

on('stop', 'click', async () => {
  const res = await window.pgsui.stop();
  if (!res.ok) appendLog({ stream:'stderr', line: res.error });
});

on('pickInput', 'click', async () => {
  const fmt = $('format').value;
  const filters = {
    vcf: [{ name: 'VCF', extensions: ['vcf','gz'] }],
    phylip: [{ name: 'PHYLIP', extensions: ['phy','phylip'] }],
    structure: [{ name: 'STRUCTURE', extensions: ['str','stru'] }],
    genepop: [{ name: 'GENEPOP', extensions: ['gen','genepop'] }]
  }[fmt] || [{ name: 'All', extensions: ['*'] }];
  const f = await window.pgsui.pickFile(filters);
  if (f) $('inputPath').value = f;
});
on('pickPopmap', 'click', async () => {
  const f = await window.pgsui.pickFile();
  if (f) $('popmapPath').value = f;
});
on('pickYaml', 'click', async () => {
  const f = await window.pgsui.pickFile([{ name: 'YAML', extensions: ['yml','yaml'] }]);
  if (f) $('yamlPath').value = f;
});
on('pickDump', 'click', async () => {
  const f = await window.pgsui.pickSave('effective.yaml');
  if (f) $('dumpConfigPath').value = f;
});
on('pickLogFile', 'click', async () => {
  const f = await window.pgsui.pickSave('pgsui-run.log');
  if (f) $('logFile').value = f;
});

/* ---- streams ---- */
window.pgsui.onLog(appendLog);
window.pgsui.onError((e) => appendLog({ stream:'stderr', line: e.message || String(e) }));
window.pgsui.onExit(({ code }) => { appendLog({ stream:'stdout', line:`Process exited with code ${code}` }); setRunningUI(false); });
window.pgsui.onStarted(({ argv, cwd }) => {
  appendLog({ stream:'stdout', line:`Started: ${argv?.join(' ') || ''}` });
  appendLog({ stream:'stdout', line:`CWD: ${cwd}` });
  setRunningUI(true);
});
