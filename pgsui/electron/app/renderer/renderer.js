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

function collectPayload() {
  const modelsSel = Array.from($('models').selectedOptions).map(o => o.value);
  const includePops = $('includePops').value.trim().split(/\s+/).filter(Boolean);
  const setPairs = $('setPairs').value.split('\n').map(s => s.trim()).filter(Boolean);
  const preset = $('preset').value || 'fast';
  const bs = $('batchSize').value ? Number($('batchSize').value) : 64;

  return {
    usePgSui: $('runnerPgSui').checked,
    pythonPath: $('pythonPath').value.trim() || undefined,
    cliPath: $('cliPath').value.trim() || undefined,
    cwd: $('cwd').value.trim(),
    inputPath: $('inputPath').value.trim(),
    format: $('format').value,
    popmapPath: $('popmapPath').value.trim() || undefined,
    prefix: $('prefix').value.trim() || undefined,
    yamlPath: $('yamlPath').value.trim() || undefined,
    dumpConfigPath: $('dumpConfigPath').value.trim() || undefined,
    preset,
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

// init
(async () => {
  // default tune UI
  const tune = $('tune'), tuneOpts = $('tuneOpts');
  tune.addEventListener('change', () => { tuneOpts.style.display = tune.checked ? '' : 'none'; });

  // runner toggle UI
  const block = $('pythonBlock');
  function syncRunner() { block.style.display = $('runnerPython').checked ? '' : 'none'; }
  $('runnerPgSui').addEventListener('change', syncRunner);
  $('runnerPython').addEventListener('change', syncRunner);

  // prefer pg-sui if available
  const hasPgSui = await window.pgsui.detectPgSui();
  if (!hasPgSui.ok) { $('runnerPython').checked = true; $('runnerPgSui').checked = false; }
  syncRunner();
})();

// actions
$('pickPython').addEventListener('click', async () => {
  const f = await window.pgsui.pickFile([{ name: 'All', extensions: ['*'] }]);
  if (f) $('pythonPath').value = f;
});

$('start').addEventListener('click', async () => {
  logEl.textContent = '';
  const payload = collectPayload();
  if (!payload.cwd) { appendLog({ stream: 'stderr', line: 'Working directory is required.' }); return; }
  if (!payload.inputPath) { appendLog({ stream: 'stderr', line: 'Input file required.' }); return; }
  if (!payload.usePgSui && !payload.cliPath) { appendLog({ stream: 'stderr', line: 'cli.py path required when not using pg-sui.' }); return; }
  const res = await window.pgsui.start(payload);
  if (!res.ok) appendLog({ stream: 'stderr', line: res.error });
});

$('stop').addEventListener('click', async () => {
  const res = await window.pgsui.stop();
  if (!res.ok) appendLog({ stream: 'stderr', line: res.error });
});

$('pickCli').addEventListener('click', async () => {
  const f = await window.pgsui.pickFile([{ name: 'Python', extensions: ['py'] }]);
  if (f) $('cliPath').value = f;
});
$('pickInput').addEventListener('click', async () => {
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
$('pickPopmap').addEventListener('click', async () => {
  const f = await window.pgsui.pickFile();
  if (f) $('popmapPath').value = f;
});
$('pickYaml').addEventListener('click', async () => {
  const f = await window.pgsui.pickFile([{ name: 'YAML', extensions: ['yml','yaml'] }]);
  if (f) $('yamlPath').value = f;
});
$('pickDump').addEventListener('click', async () => {
  const f = await window.pgsui.pickSave('effective.yaml');
  if (f) $('dumpConfigPath').value = f;
});
$('pickLogFile').addEventListener('click', async () => {
  const f = await window.pgsui.pickSave('pgsui-run.log');
  if (f) $('logFile').value = f;
});

// stream hooks
window.pgsui.onLog(appendLog);
window.pgsui.onError((e) => appendLog({ stream: 'stderr', line: e.message || String(e) }));
window.pgsui.onExit(({ code }) => { appendLog({ stream: 'stdout', line: `Process exited with code ${code}` }); setRunningUI(false); });
window.pgsui.onStarted(({ argv, cwd }) => {
  appendLog({ stream: 'stdout', line: `Started: ${argv.join(' ')}` });
  appendLog({ stream: 'stdout', line: `CWD: ${cwd}` });
  setRunningUI(true);
});
