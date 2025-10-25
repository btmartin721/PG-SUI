const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('pgsui', {
  start: (payload) => ipcRenderer.invoke('pgsui:start', payload),
  stop: () => ipcRenderer.invoke('pgsui:stop'),
  onLog: (cb) => ipcRenderer.on('pgsui:log', (_e, m) => cb(m)),
  onExit: (cb) => ipcRenderer.on('pgsui:exit', (_e, m) => cb(m)),
  onStarted: (cb) => ipcRenderer.on('pgsui:started', (_e, m) => cb(m)),
  onError: (cb) => ipcRenderer.on('pgsui:error', (_e, m) => cb(m)),
  pickFile: (filters) => ipcRenderer.invoke('pick:file', { filters }),
  pickDir: () => ipcRenderer.invoke('pick:dir'),
  pickSave: (defaultPath) => ipcRenderer.invoke('pick:save', { defaultPath }),
  defaultCli: () => ipcRenderer.invoke('default:cli'),
  detectPgSui: () => ipcRenderer.invoke('detect:pgsui'),
});
