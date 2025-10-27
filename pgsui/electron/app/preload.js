const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('pgsui', {
  detectPgSui: () => ipcRenderer.invoke('detect:pgsui'),
  start: (payload) => ipcRenderer.invoke('pgsui:start', payload),
  stop: () => ipcRenderer.invoke('pgsui:stop'),
  pickFile: (filters) => ipcRenderer.invoke('pick:file', { filters }),
  pickDir: () => ipcRenderer.invoke('pick:dir'),
  pickSave: (defaultPath) => ipcRenderer.invoke('pick:save', { defaultPath }),
  onLog: (fn) => ipcRenderer.on('pgsui:log', (_e, d) => fn(d)),
  onError: (fn) => ipcRenderer.on('pgsui:error', (_e, d) => fn(d)),
  onExit: (fn) => ipcRenderer.on('pgsui:exit', (_e, d) => fn(d)),
  onStarted: (fn) => ipcRenderer.on('pgsui:started', (_e, d) => fn(d)),
  defaultCli: () => ipcRenderer.invoke('default:cli'),
});
