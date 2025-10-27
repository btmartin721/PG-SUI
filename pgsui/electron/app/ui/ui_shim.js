// ui-shim.js: Browser shim for Electron's window.pgsui API

(function () {
    function wsConnect(onLog) {
        const ws = new WebSocket(`ws://${location.host}/api/logs`);
        ws.onmessage = (e) => {
        const s = e.data.indexOf('|');
            if (s > 0) {
                const stream = e.data.slice(0, s);
                const line = e.data.slice(s + 1);
                onLog({ stream, line });
        }
    };
        ws.onerror = () => {/* no-op */};
        return ws;
    }

    const listeners = { log: [], error: [], exit: [], started: [] };
    function emit(type, payload) { listeners[type].forEach(fn => fn(payload)); }

    let ws = null;

    const api = {
        // Electron feature parity stubs
        pickFile: async () => null,   // Browser cannot provide host filesystem paths
        pickDir: async () => null,
        pickSave: async () => null,

        // Detect pg-sui presence by calling the API once we start; return optimistic true
        detectPgSui: async () => ({ ok: true }),

        // Start/stop map to REST
        start: async (payload) => {
            try {
                const r = await fetch('/api/start', {
                method: 'POST', headers: { 'content-type': 'application/json' },
                body: JSON.stringify(payload)
            });
                const j = await r.json();
                if (!j.ok) return j;
                emit('started', { argv: j.argv || [], cwd: j.cwd || '' });
                if (!ws) ws = wsConnect((m) => emit('log', m));
                return { ok: true };
            } catch (e) {
                emit('error', { message: String(e) });
                return { ok: false, error: String(e) };
            }
        },
        stop: async () => {
            try {
                const r = await fetch('/api/stop', { method: 'POST' });
                const j = await r.json();
                if (!j.ok) return j;
                emit('exit', { code: 0 });
                return { ok: true };
            } catch (e) {
                emit('error', { message: String(e) });
                return { ok: false, error: String(e) };
            }
        },

    // Event wiring
    onLog:   (fn) => { listeners.log.push(fn); },
    onError: (fn) => { listeners.error.push(fn); },
    onExit:  (fn) => { listeners.exit.push(fn); },
    onStarted: (fn) => { listeners.started.push(fn); },
    };

    // Expose only if Electron preload didn't define it
    if (!window.pgsui) window.pgsui = api;

})();
