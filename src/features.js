// ===================================================================================================
// Probes, recording, playback and data export
// ===================================================================================================

// ---------- probes ----------
const PROBE_MAX = 400;
const PB_REF = 16; // hidden probes along the pitot-static station, used as the pressure reference while recording
S.probes = [];     // {id, x, y}: tunnel coordinates in chords (x from the inlet, y from the floor); probes stay put if the model moves
S.probeTap = false;
let probeSeq = 1;
// probe coordinates: tunnel axes, origin at the model's leading edge (x downstream, y up), in chords
const leX = () => S.pos.x - 0.25 * Math.cos(S.aoa * Math.PI / 180), leY = () => S.pos.y + 0.25 * Math.sin(S.aoa * Math.PI / 180);
const relX = (X) => X - leX(), relY = (Y) => Y - leY();
const absX = (x) => x + leX(), absY = (y) => y + leY();
S.probeRate = 200; // samples per flow-through
const probeEvery = () => Math.max(1, Math.round(FT() / S.probeRate));

function probeLatPts() { // fine-grid cell coordinates for the user probes, then the reference probes
  const Nc = G.F.Nc, pts = S.probes.map(p => [p.x * Nc - 0.5, p.y * Nc - 0.5]);
  for (let j = 0; j < PB_REF; j++) { const y = 0.25 + (S.H - 0.5) * (j + 0.5) / PB_REF; pts.push([0.7 * Nc - 0.5, y * Nc - 0.5]); }
  return pts;
}
function newProbeBuffer() { return { rows: [], np: S.probes.length }; }
function sendProbes() {
  if (!G) return;
  if (MODE === 'gpu' && gpu) gpu.setProbes(probeLatPts());
  else if (engine) { const pts = probeLatPts(); engine.post({ type: 'probes', pts: Float32Array.from(pts.flat()), every: probeEvery() }); }
  G.liveNext = undefined;
}
function probesChanged() {
  sendProbes();
  if (G) G.pbuf = newProbeBuffer();
  renderProbeUI(); dirty = true;
}
function probeInside(X, Y) { // inside the model or outside the test section?
  const x = X * G.Nc - 0.5, y = Y * G.Nc - 0.5;
  if (x < 0.5 * G.Nc || x > G.nx - 1 || y < 0 || y > G.ny - 1) return true;
  return isSolid(x, y);
}
function addProbe(X, Y) {
  if (S.probes.length >= PROBE_MAX) { toast(`Up to ${PROBE_MAX} probes can be placed.`); return false; }
  if (probeInside(X, Y)) return false;
  S.probes.push({ id: probeSeq++, x: X, y: Y }); return true;
}
function probeNear(X, Y) { // nearest probe within a finger's reach
  let best = -1, bd = (14 * dpr / scale / G.Nc) ** 2;
  S.probes.forEach((p, i) => { const d = (p.x - X) ** 2 + (p.y - Y) ** 2; if (d < bd) { bd = d; best = i; } });
  return best;
}
// one sample row: [step, u/U, v/U, Cp for each probe]; NaN where a probe sits in a solid
function probeRow(t, vals, np, rhoRef, uRef) {
  const row = new Float32Array(1 + 3 * np); row[0] = t;
  const q = 2 / (3 * uRef * uRef);
  for (let i = 0; i < np; i++) {
    const ok = vals[4 * i + 3] > 0.5;
    row[1 + 3 * i] = ok ? vals[4 * i + 1] / uRef : NaN;
    row[2 + 3 * i] = ok ? vals[4 * i + 2] / uRef : NaN;
    row[3 + 3 * i] = ok ? (vals[4 * i] - rhoRef) * q : NaN;
  }
  return row;
}
function cpuProbeVals() { // CPU path: sample the solver arrays directly
  const pts = probeLatPts(), v = new Float32Array(pts.length * 4);
  pts.forEach(([x, y], i) => {
    if (isSolid(x, y)) { v[4 * i] = 1; return; }
    v[4 * i] = sampleField(G.rho, x, y, G.nx, G.ny); v[4 * i + 1] = sampleField(G.ux, x, y, G.nx, G.ny); v[4 * i + 2] = sampleField(G.uy, x, y, G.nx, G.ny); v[4 * i + 3] = 1;
  });
  return v;
}
// live tunnel, GPU: advance n steps, sampling the probes at an exact step cadence
function liveStep(n) {
  const np = S.probes.length;
  if (!np || !gpu.PRB || !G.pbuf) { gpu.step(n); return; }
  const every = probeEvery(); let rows = 0; const times = [];
  const drain = () => {
    if (!rows) return;
    const vals = gpu.readProbes(rows), w = gpu.probeW;
    for (let r = 0; r < rows; r++) G.pbuf.rows.push(probeRow(times[r], vals.subarray(r * w * 4, r * w * 4 + np * 4), np, G.rhoRef, G.uRef));
    rows = 0; times.length = 0;
  };
  let left = n;
  while (left > 0) {
    if (G.liveNext === undefined || G.liveNext <= gpu.t || G.liveNext > gpu.t + every) G.liveNext = gpu.t + every;
    const m = Math.min(left, G.liveNext - gpu.t);
    gpu.step(m); left -= m;
    if (gpu.t === G.liveNext) { if (rows >= gpu.PROWS) drain(); gpu.sampleProbes(rows); times.push(gpu.t); rows++; G.liveNext += every; }
  }
  drain(); trimLive();
}
function trimLive() { const keep = 4 * FT(); while (G.pbuf.rows.length && G.pbuf.rows[0][0] < G.t - keep) G.pbuf.rows.shift(); }
// worker samples (CPU path): rows of [t, Fx, Fy, Mz, then rho, u, v, ok per probe point]
function workerRows(m) {
  if (!m.samples || !m.ns) return [];
  const w = m.samples.length / m.ns, out = [];
  for (let r = 0; r < m.ns; r++) out.push(m.samples.subarray(r * w, (r + 1) * w));
  return out;
}
function liveProbeSampleCPU(rows) {
  const np = S.probes.length; if (!np || !G.pbuf || G.play) return;
  for (const r of rows) G.pbuf.rows.push(probeRow(r[0], r.subarray(4, 4 + 4 * np), np, G.rhoRef, G.uRef));
  trimLive();
}
function drawProbes() {
  if (!S.probes.length) return;
  const many = S.probes.length > 40, r = (many ? 2.2 : 3.4) * dpr;
  const light = (S.field === 'vort' || S.field === 'press' || S.field === 'schl');
  vctx.save();
  vctx.lineWidth = 1.2 * dpr; vctx.strokeStyle = light ? 'rgba(20,24,28,.9)' : 'rgba(255,255,255,.92)';
  vctx.fillStyle = 'rgba(238,132,66,.95)';
  vctx.font = `500 ${9 * dpr}px "JetBrains Mono", monospace`; vctx.textAlign = 'left'; vctx.textBaseline = 'bottom';
  const sel = probeChartSel();
  for (const p of S.probes) {
    const [X, Y] = toScr(p.x * G.Nc - 0.5, p.y * G.Nc - 0.5);
    vctx.beginPath(); vctx.arc(X, Y, p.id === sel ? r * 1.8 : r, 0, 6.283); vctx.fill(); vctx.stroke();
    if (!many || p.id === sel) { vctx.fillStyle = light ? 'rgba(20,24,28,.95)' : 'rgba(255,255,255,.95)'; vctx.fillText(String(p.id), X + r + 2 * dpr, Y - 1 * dpr); vctx.fillStyle = 'rgba(238,132,66,.95)'; }
  }
  vctx.restore();
}

// ---------- storage: IndexedDB, compressed ----------
const Store = (() => {
  let dbp = null;
  const open = () => dbp || (dbp = new Promise((res, rej) => {
    try {
      const r = indexedDB.open('aerofoil-wind-tunnel', 1);
      r.onupgradeneeded = () => { const d = r.result; d.createObjectStore('runs', { keyPath: 'id' }); d.createObjectStore('chunks'); };
      r.onsuccess = () => res(r.result); r.onerror = () => rej(r.error);
    } catch (e) { rej(e); }
  }));
  const req = (store, mode, fn) => open().then(d => new Promise((res, rej) => {
    const t = d.transaction(store, mode), r = fn(t.objectStore(store));
    t.oncomplete = () => res(r && 'result' in r ? r.result : undefined); t.onerror = () => rej(t.error); t.onabort = () => rej(t.error);
  }));
  return {
    putRun: (run) => req('runs', 'readwrite', s => s.put(run)),
    getRun: (id) => req('runs', 'readonly', s => s.get(id)),
    runs: () => req('runs', 'readonly', s => s.getAll()),
    put: (key, v) => req('chunks', 'readwrite', s => s.put(v, key)),
    get: (key) => req('chunks', 'readonly', s => s.get(key)),
    del: (key) => req('chunks', 'readwrite', s => s.delete(key)),
    async deleteRun(id) {
      await req('chunks', 'readwrite', s => s.delete(IDBKeyRange.bound(id + '/', id + '/￿')));
      await req('runs', 'readwrite', s => s.delete(id));
    },
    ok: () => open().then(() => true, () => false),
  };
})();
const canZip = typeof CompressionStream !== 'undefined' && typeof DecompressionStream !== 'undefined';
async function zip(buf) {
  if (!canZip) return { z: 0, b: buf };
  try { return { z: 1, b: await new Response(new Blob([buf]).stream().pipeThrough(new CompressionStream('gzip'))).arrayBuffer() }; } catch (e) { return { z: 0, b: buf }; }
}
async function unzip(o) { return o.z ? new Response(new Blob([o.b]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer() : o.b; }
function concatF32(rows, width) { const out = new Float32Array(rows.length * width); rows.forEach((r, i) => out.set(r, i * width)); return out; }

// frame layout: Int32 header [t, nx, ny, cw, ch], then Int16 ux, uy, cp (nx*ny), Int16 vort (cw*ch), Uint8 smoke+heat (cw*ch*2)
const QU = 4000, QC = 4000, QV = 100; // fixed-point scales: u/U0, Cp, vorticity (omega c / U)
function packFrame(f) {
  const n = f.nx * f.ny, m = f.cw * f.ch, size = 20 + 6 * n + 2 * m + 2 * m;
  const buf = new ArrayBuffer(size); new Int32Array(buf, 0, 5).set([f.t, f.nx, f.ny, f.cw, f.ch]);
  let o = 20;
  for (const a of [f.ux, f.uy, f.cp]) { new Int16Array(buf, o, n).set(a); o += 2 * n; }
  new Int16Array(buf, o, m).set(f.vort); o += 2 * m;
  new Uint8Array(buf, o, 2 * m).set(f.sm);
  return buf;
}
function unpackFrame(buf) {
  const [t, nx, ny, cw, ch] = new Int32Array(buf, 0, 5), n = nx * ny, m = cw * ch;
  let o = 20; const take16 = (k) => { const a = new Int16Array(buf, o, k); o += 2 * k; return a; };
  const ux = take16(n), uy = take16(n), cp = take16(n), vort = take16(m);
  return { t, nx, ny, cw, ch, ux, uy, cp, vort, sm: new Uint8Array(buf, o, 2 * m) };
}

// ---------- CSV export ----------
const dlP = (async () => { try { return window.claude && window.claude.use ? await window.claude.use('downloads') : null; } catch (e) { return null; } })();
const STANDALONE = !(window.claude && window.claude.use); // opened as a plain web page or app, outside Claude: save with ordinary downloads
function plainDownload(filename, blob) {
  const url = URL.createObjectURL(blob), a = document.createElement('a');
  a.href = url; a.download = filename; a.rel = 'noopener'; document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 60000);
}
async function deliverCSV(filename, text) {
  if (STANDALONE) { plainDownload(filename, new Blob([text], { type: 'text/csv' })); toast(`Saved ${filename}.`); return; }
  const dl = await Promise.race([dlP, new Promise(r => setTimeout(() => r(null), 1500))]);
  if (dl) {
    try { await dl.save({ filename, data: new Blob([text], { type: 'text/csv' }) }); return; }
    catch (e) {
      const c = e && e.code;
      if (c === 'declined') return;
      if (c === 'rate_limited') { toast('Another save is already waiting for your answer.'); return; }
      if (c === 'too_large') { toast('That file is too large for this destination. Try exporting fewer probes.'); return; }
    }
  }
  try { await navigator.clipboard.writeText(text); toast(`Copied ${filename} to the clipboard (${(text.length / 1024).toFixed(0)} KB). Paste it into a spreadsheet or text file.`); return; } catch (e) { /* fall through */ }
  showCSVDialog(filename, text);
}
function showCSVDialog(filename, text) {
  const d = $('#csvDialog'); $('#csvName').textContent = filename;
  const ta = $('#csvText'); ta.value = text.length > 8e6 ? text.slice(0, 8e6) + '\n…' : text; d.hidden = false; ta.focus(); ta.select();
}
function csvMeta(o) { // '# key=value' header lines, as a DAQ file would carry
  const L = ['# Aerofoil Wind Tunnel export (2-D lattice-Boltzmann simulation)'];
  for (const [k, v] of Object.entries(o)) L.push(`# ${k}=${v}`);
  return L.join('\n') + '\n';
}
function metaFor(st, extra) {
  const P = PRESETS, model = st.preset === 'cyl' ? 'cylinder D=0.5c' : st.preset === 'plate' ? 'flat plate' : `NACA ${st.m}${st.p}${String(st.t).padStart(2, '0')}`;
  void P;
  return Object.assign({ model, alpha_deg: st.aoa, chord_m: st.chord, U_inf_m_s: st.speed, Re_rig: Math.round(st.speed * st.chord / NU_AIR), Mach: (st.speed / 343).toFixed(4),
    freestream_turbulence_pct: st.turb, tunnel_length_c: st.L, tunnel_height_c: st.H, walls: st.walls === 'solid' ? 'closed no-slip' : 'slip' }, extra);
}
function slug() { return `${profileName().replace(/\s+/g, '')}_a${S.aoa}_Re${flowParams().Re}`; }
function probeCSV(rows, probes, dtStep, chord, speed, meta) {
  const hdr = ['time_ms', 't_U_over_c'];
  for (const p of probes) { const loc = `(x/c=${p.rx.toFixed(3)} y/c=${p.ry.toFixed(3)})`; hdr.push(`P${p.id} u/U ${loc}`, `P${p.id} v/U ${loc}`, `P${p.id} Cp ${loc}`); }
  const out = [hdr.map(h => `"${h}"`).join(',')];
  const t0 = rows.length ? rows[0][0] : 0;
  for (const r of rows) {
    const ts = (r[0] - t0) * dtStep, line = [(ts * 1000).toFixed(4), (ts * speed / chord).toFixed(5)];
    for (let k = 1; k < r.length; k++) line.push(isFinite(r[k]) ? r[k].toFixed(5) : '');
    out.push(line.join(','));
  }
  return (meta ? csvMeta(meta) : '') + out.join('\n');
}
function probeMetaList() { return S.probes.map(p => ({ id: p.id, rx: relX(p.x), ry: relY(p.y) })); }
function exportLiveProbes() {
  if (!S.probes.length || !G.pbuf || !G.pbuf.rows.length) { toast('Add probes and let the flow run for a moment first.'); return; }
  const every = probeEvery(), dt = flowTime(1);
  deliverCSV(`probes_live_${slug()}.csv`, probeCSV(G.pbuf.rows, probeMetaList(), dt, S.chord, S.speed, metaFor(S, {
    Re_simulated: flowParams().Re, cells_per_chord: G.F.Nc, sample_rate_Hz: (1 / (every * dt)).toFixed(3), sample_interval_steps: every, samples: G.pbuf.rows.length,
    coordinates: 'x/c and y/c in tunnel axes from the leading edge', velocity: 'u/U and v/U with U from the pitot-static probe (time-averaged)',
    Cp: 'solver static pressure at the probe, referenced to the pitot-static probe', probe_model: 'ideal point sensor (bilinear sample of the solver grid)' })));
}

// ---------- recording ----------
const REC_FT_DEFAULT = 5, REC_FPF_DEFAULT = 40;
S.recFT = REC_FT_DEFAULT; S.recFPF = REC_FPF_DEFAULT;
let wakeLock = null;
async function wake(on) {
  try {
    if (on && !wakeLock && navigator.wakeLock && !document.hidden) { wakeLock = await navigator.wakeLock.request('screen'); wakeLock.addEventListener('release', () => { wakeLock = null; }); }
    else if (!on && wakeLock) { const w = wakeLock; wakeLock = null; await w.release(); }
  } catch (e) { wakeLock = null; }
}
const SETTING_KEYS = ['preset', 'm', 'p', 't', 'aoa', 'speed', 'chord', 'turb', 'L', 'H', 'walls', 'res', 'smokeSrc'];
function settingsSnapshot() { const o = {}; for (const k of SETTING_KEYS) o[k] = S[k]; o.pos = { ...S.pos }; return o; }
function applySettings(o) {
  for (const k of SETTING_KEYS) if (k in o) S[k] = o[k];
  S.pos = { ...o.pos };
  for (const sec of CTRL) for (const c of sec.items) if (c.sync) c.sync();
  syncTags();
}
function captureDims() { const cw = Math.min(G.F.nx, 900); return [cw, Math.round(cw * G.F.ny / G.F.nx)]; }
function recEstimate() {
  const ft = FT(), steps = S.recFT * ft, frames = Math.round(S.recFT * S.recFPF);
  const [cw, ch] = MODE === 'gpu' ? captureDims() : [G.nx, G.ny];
  const bytes = frames * (G.nx * G.ny * 6 + cw * ch * 4) * 0.2 + steps / Math.max(4, Math.round(ft / 200)) * (S.probes.length * 3 + 4) * 4;
  const sps = G.sps ? G.sps * recBoost() : 0, settle = G.since >= ft ? 0 : 1;
  const per = MODE === 'gpu' ? flowSecs(gridPlan(S.res)) : NaN; // same pace as the grid estimates, when it is known
  return { steps, frames, bytes, secs: isFinite(per) ? (S.recFT + settle) * per : sps > 0 ? (steps + settle * ft) / sps : NaN, realMs: steps * flowTime(1) * 1000 };
}
function recBoost() { // how much faster recording runs than the live view, learned on this device
  let b = MODE === 'gpu' ? 1.7 : 1.1;
  try { const v = +localStorage.getItem('wt.recBoost.' + MODE); if (v > 0.5 && v < 20) b = v; } catch (e) { /* storage blocked */ }
  return b;
}
function learnBoost(rec, live) { if (!(rec > 0 && live > 0)) return; try { localStorage.setItem('wt.recBoost.' + MODE, String(Math.min(20, Math.max(0.5, rec / live)))); } catch (e) { /* storage blocked */ } }
const fmtBytes = (b) => b > 1e9 ? (b / 1e9).toFixed(1) + ' GB' : b > 1e6 ? Math.round(b / 1e6) + ' MB' : Math.max(1, Math.round(b / 1e3)) + ' kB';
const fmtDur = (s) => !isFinite(s) ? 'unknown' : s < 90 ? `${Math.max(1, Math.round(s))} s` : s < 5400 ? `${Math.round(s / 60)} min` : `${(s / 3600).toFixed(1)} h`;

// field statistics: sums over samples taken at the probe cadence (GPU accumulator, or per snapshot on the CPU path).
// Velocity is stored as u - U0 and density as rho - 1, so float sums keep their precision.
function newStats(N, m) { return { n: 0, su: new Float64Array(N), sv: new Float64Array(N), suu: new Float64Array(N), svv: new Float64Array(N), suv: new Float64Array(N), sp: new Float64Array(N), spp: new Float64Array(N), sw: new Float64Array(N), sww: new Float64Array(N), nF: 0, ss: new Float32Array(m), sh: new Float32Array(m) }; }
const STAT_KEYS = ['su', 'sv', 'suu', 'svv', 'suv', 'sp', 'spp', 'sw', 'sww'];

async function startRecording() {
  if (G.rec || G.play) return;
  if (!(await Store.ok())) { toast('This browser is not letting the page store data, so a run cannot be recorded here.'); return; }
  try { if (navigator.storage && navigator.storage.persist) navigator.storage.persist(); } catch (e) { /* optional */ }
  const ft = FT(), id = 'run-' + Date.now().toString(36);
  const [cw, ch] = MODE === 'gpu' ? captureDims() : [G.nx, G.ny];
  const every = probeEvery();
  const meta = {
    id, created: Date.now(), status: 'recording', mode: MODE, title: `${profileName()} at ${S.aoa.toFixed(1)}°`,
    settings: settingsSnapshot(), Re: flowParams().Re, grid: { nx: G.nx, ny: G.ny, Nc: G.Nc, F: { ...G.F } }, cap: { cw, ch },
    plan: { ft: S.recFT, fpf: S.recFPF, probeRate: S.probeRate },
    steps: { total: Math.round(S.recFT * ft / every) * every, done: 0, ft, probeEvery: every, frameEvery: Math.max(1, Math.round(ft / S.recFPF)), t0: 0 },
    dtStep: flowTime(1), chord: S.chord, speed: S.speed,
    probes: probeMetaList(), probeAbs: S.probes.map(p => ({ ...p })), np: S.probes.length,
    frames: 0, probeChunks: 0, forceChunks: 0, bytes: 0, wallSecs: 0,
  };
  G.rec = { liveSps: G.sps, meta, phase: 'settle', rowsInTex: 0, rowTimes: [], probeRows: [], forceRows: [], advAcc: 0, nextProbe: 0, nextFrame: 0,
    stats: newStats(G.N, cw * ch), base: null, accN: 0, cpSum: null, cpSq: null, cpCnt: null, refRho: G.rhoRef, refU: G.uRef, refInit: G.refInit,
    io: Promise.resolve(), lastCkpt: performance.now(), lastUi: 0, lastRender: 0, wallStart: performance.now(), healthAt: 0 };
  if (MODE === 'gpu') gpu.setProbes(probeLatPts()); else sendProbes();
  S.running = true; syncRunButton();
  setLocked(true); wake(true);
  await Store.putRun(meta);
  renderRecordUI();
  if (MODE === 'cpu') engine.post({ type: 'run', on: true });
}
function settledEnough() { const need = (G.firstRun ? 2 : 1) * FT(); return G.since >= need; }
function beginCapture() {
  const R = G.rec, M = R.meta; R.phase = 'run'; M.steps.t0 = G.t; M.steps.done = 0;
  R.wallStart = performance.now(); R.etaT0 = performance.now(); R.etaS0 = 0; G.firstRun = false;
  if (MODE === 'gpu') { // first sample and first frame at t = 0 of the run
    gpu.accReset(); R.accN = 0;
    const gl = gpu.gl; let e, oom = false; while ((e = gl.getError()) !== gl.NO_ERROR && !gl.isContextLost()) if (e === gl.OUT_OF_MEMORY) oom = true;
    if (oom) { abortRecording(`There isn't enough graphics memory to record on the ${RES_NAME[S.res]} grid on this device. Choose a coarser grid and try again.`); return; }
    recSampleGPU(); R.nextProbe = M.steps.probeEvery;
    captureFrame(); R.nextFrame = M.steps.frameEvery;
  } else { R.nextProbe = 0; R.nextFrame = 0; }
  renderRecordUI();
}
function advanceScalars(n) { // smoke and heat, as in the live loop
  gpu.fields(0, G.rhoRef, G.F.Nc, G.uRef);
  const Rl = smokeRelease(), k = G.F.k;
  gpu.advect(n, { rake: { x0: Rl.x0 * k, x1: Rl.x1 * k, y0: Rl.y0 * k, y1: Rl.y1 * k, sp: Rl.sp * k, off: Rl.off * k, w: Rl.w * k }, kHeat: Math.max(gpu.nu / 0.71, 0.008), kSmoke: gpu.nu / 4 });
}
function refUpdate(rhoS, uS, n) { // pitot-static reference from the reference probes, time-averaged like the live one
  const R = G.rec, tau = 2 * G.F.ny / 0.577, a = R.refInit ? 1 - Math.exp(-Math.max(1, n) / tau) : 1;
  if (uS < 0.3 * U0) return;
  R.refRho += a * (rhoS - R.refRho); R.refU += a * (uS - R.refU); R.refInit = true;
}
function refFromVals(vals, base, np, n) {
  let rs = 0, us = 0, c = 0;
  for (let j = 0; j < PB_REF; j++) { const o = base + (np + j) * 4; if (vals[o + 3] > 0.5) { rs += vals[o]; us += vals[o + 1]; c++; } }
  if (c) refUpdate(rs / c, us / c, n);
  const R = G.rec; if (R && R.phase === 'run') { R.refSumRho = (R.refSumRho || 0) + R.refRho; R.refSumU = (R.refSumU || 0) + R.refU; R.refN = (R.refN || 0) + 1; } // run-mean reference
}
function flushProbeRows() { // GPU: read the probe history texture and convert to sample rows
  const R = G.rec; if (!R || MODE !== 'gpu' || !R.rowsInTex) return;
  const np = R.meta.np, w = gpu.probeW, vals = gpu.readProbes(R.rowsInTex);
  for (let r = 0; r < R.rowsInTex; r++) {
    const base = r * w * 4;
    refFromVals(vals, base, np, R.meta.steps.probeEvery);
    if (np) R.probeRows.push(probeRow(R.rowTimes[r], vals.subarray(base, base + np * 4), np, R.refRho, R.refU));
  }
  R.rowsInTex = 0; R.rowTimes = [];
}
function recSampleGPU() {
  const R = G.rec, t = R.meta.steps.done;
  if (R.rowsInTex >= gpu.PROWS) flushProbeRows();
  gpu.sampleProbes(R.rowsInTex); R.rowTimes.push(t); R.rowsInTex++;
  gpu.accumulate(G.F.Nc / U0); R.accN++;
  const [fx, fy, mz] = gpu.forces(), qn = 0.5 * U0 * U0 * G.F.Nc;
  R.forceRows.push(new Float32Array([t, fy / qn, fx / qn, -mz / (qn * G.F.Nc)]));
  pushForces(fx, fy, mz, R.meta.steps.probeEvery);
}
function quant(src, scale, mask) { const out = new Int16Array(src.length); for (let i = 0; i < src.length; i++) out[i] = mask && mask[i] ? 0 : Math.max(-32767, Math.min(32767, Math.round(src[i] * scale))); return out; }
function blendedCp() { // display pressure: Poisson field blended into the solver pressure next to the model
  updatePoisson(1);
  const q = 2 / (3 * G.uRef * G.uRef), cap = cpLimit(), out = new Float32Array(G.N);
  for (let i = 0; i < G.N; i++) { if (G.mask[i]) continue; const cpL = (G.rho[i] - G.rhoRef) * q; out[i] = Math.min(cap, G.pw[i] * cpL + (1 - G.pw[i]) * G.cpP[i]); }
  return out;
}
function cpuVorticity() {
  const { nx, ny, ux, uy, mask, Nc } = G, w = new Float32Array(G.N);
  for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
    const i = y * nx + x; if (mask[i]) continue;
    const xl = x > 0 && !mask[i - 1] ? i - 1 : i, xr = x < nx - 1 && !mask[i + 1] ? i + 1 : i, yd = y > 0 && !mask[i - nx] ? i - nx : i, yu = y < ny - 1 && !mask[i + nx] ? i + nx : i;
    w[i] = ((uy[xr] - uy[xl]) / ((xr - xl) || 1) - (ux[yu] - ux[yd]) / (((yu - yd) / nx) || 1)) * Nc / U0;
  }
  return w;
}
function captureFrame() {
  const R = G.rec, M = R.meta;
  let vort, sm;
  const { cw, ch } = M.cap, m = cw * ch;
  if (MODE === 'gpu') {
    const d = gpu.readDown();
    for (let i = 0, j = 0; i < G.N; i++, j += 4) { G.rho[i] = d[j]; G.ux[i] = d[j + 1]; G.uy[i] = d[j + 2]; }
    for (let k = 0; k < 32; k++) { const r = G.rho[(Math.random() * G.N) | 0]; if (!(r > 0.5 && r < 1.6)) { abortRecording('The flow diverged during the recording, so it stopped. Try a lower air speed or a finer grid.'); return; } }
    G.rhoRef = R.refRho; G.uRef = R.refU; G.refInit = true;
    gpu.fields(1, G.rhoRef, G.F.Nc, G.uRef);
    const raw = gpu.capture(cw, ch, QV);
    vort = new Int16Array(m); sm = new Uint8Array(2 * m);
    for (let i = 0; i < m; i++) { vort[i] = raw[4 * i] * 256 + raw[4 * i + 1] - 32768; sm[2 * i] = raw[4 * i + 2]; sm[2 * i + 1] = raw[4 * i + 3]; }
  } else {
    vort = quant(cpuVorticity(), QV, G.mask); sm = new Uint8Array(2 * m);
    for (let i = 0; i < m; i++) sm[2 * i] = Math.min(255, Math.round(Math.min(1, G.dye[i] * 0.32) * 255));
  }
  const cp = blendedCp();
  const A = R.stats; A.nF++; // smoke and heat averages come from the frames (they are pictures, not measurements)
  for (let i = 0; i < m; i++) { A.ss[i] += sm[2 * i]; A.sh[i] += sm[2 * i + 1]; }
  updateCp(M.steps.frameEvery);
  const fr = { t: M.steps.done, nx: G.nx, ny: G.ny, cw, ch, ux: quant(G.ux, QU / U0, G.mask), uy: quant(G.uy, QU / U0, G.mask), cp: quant(cp, QC, G.mask), vort, sm };
  const idx = M.frames++, key = `${M.id}/f/${idx}`, buf = packFrame(fr);
  R.io = R.io.then(() => zip(buf)).then(z => { M.bytes += z.b.byteLength; return Store.put(key, z); }).catch(e => console.error(e));
  dirty = true; R.lastRender = performance.now();
}
function cpuAccumulate() { // CPU path: field statistics once per solver snapshot
  const A = G.rec.stats, w = cpuVorticity();
  for (let i = 0; i < G.N; i++) {
    if (G.mask[i]) continue;
    const du = G.ux[i] - U0, v = G.uy[i], dr = G.rho[i] - 1;
    A.su[i] += du; A.sv[i] += v; A.suu[i] += du * du; A.svv[i] += v * v; A.suv[i] += du * v; A.sp[i] += dr; A.spp[i] += dr * dr; A.sw[i] += w[i]; A.sww[i] += w[i] * w[i];
  }
  A.n++;
}
function surfaceCpAccumulate(cur) { // called from updateCp while recording: run-averaged surface pressure
  const R = G.rec; if (!R || R.phase !== 'run') return;
  if (!R.cpSum || R.cpSum.length !== cur.length) { R.cpSum = new Float64Array(cur.length); R.cpSq = new Float64Array(cur.length); R.cpCnt = new Uint32Array(cur.length); }
  for (let k = 0; k < cur.length; k++) if (isFinite(cur[k])) { R.cpSum[k] += cur[k]; R.cpSq[k] += cur[k] * cur[k]; R.cpCnt[k]++; }
}
function flushSeries() { // probe and force rows -> stored chunks
  const R = G.rec, M = R.meta;
  if (MODE === 'gpu') flushProbeRows();
  if (R.probeRows.length && M.np) {
    const data = concatF32(R.probeRows, 1 + 3 * M.np), key = `${M.id}/p/${M.probeChunks++}`; R.probeRows = [];
    R.io = R.io.then(() => zip(data.buffer)).then(z => { M.bytes += z.b.byteLength; return Store.put(key, z); });
  }
  if (R.forceRows.length) {
    const data = concatF32(R.forceRows, 4), key = `${M.id}/c/${M.forceChunks++}`; R.forceRows = [];
    R.io = R.io.then(() => zip(data.buffer)).then(z => { M.bytes += z.b.byteLength; return Store.put(key, z); });
  }
}
// High-resolution statistics for the averaged pictures: the same per-cell means and variances, binned onto the
// capture grid (up to 900 pixels across) as equivalent sums. Layout: nine arrays of cw*ch, one after another:
// fluid-cell count, then sums of u, v, rho, omega, then sums of their squares.
function gpuStats() { // read the fine-grid accumulator and average it onto the readback grid over fluid cells
  const R = G.rec, out = newStats(G.N, 0);
  if (!R.accN || !gpu.ACC) return out;
  // per fine cell: means and (co)variances; block-average those, then write them back as equivalent sums
  const k = G.F.k, fx = G.F.nx, n = R.accN, cnt = new Uint16Array(G.N);
  const mean = { u: new Float64Array(G.N), v: new Float64Array(G.N), p: new Float64Array(G.N), w: new Float64Array(G.N) };
  const vr = { uu: new Float64Array(G.N), vv: new Float64Array(G.N), uv: new Float64Array(G.N), pp: new Float64Array(G.N), ww: new Float64Array(G.N) };
  const { cw, ch } = R.meta.cap, m = cw * ch, sx = cw / fx, sy = ch / G.F.ny, hb = new Float32Array(9 * m);
  const xb = new Int32Array(fx); for (let x = 0; x < fx; x++) xb[x] = Math.min(cw - 1, (x * sx) | 0);
  gpu.readAccRows((A0, A1, A2, y0, rows) => {
    for (let yy = 0; yy < rows; yy++) {
      const y = y0 + yy, jr = Math.min(G.ny - 1, (y / k) | 0) * G.nx, hr = Math.min(ch - 1, (y * sy) | 0) * cw;
      for (let x = 0; x < fx; x++) {
        if (gpu.solid[y * fx + x]) continue;
        const j = jr + Math.min(G.nx - 1, (x / k) | 0), o = 4 * (yy * fx + x); cnt[j]++;
        const mu = A0[o] / n, mv = A0[o + 1] / n, mp = A1[o] / n, mw = A1[o + 2] / n;
        const vu = Math.max(0, A0[o + 2] / n - mu * mu), vv = Math.max(0, A0[o + 3] / n - mv * mv), vp = Math.max(0, A1[o + 1] / n - mp * mp), vw = Math.max(0, A1[o + 3] / n - mw * mw);
        mean.u[j] += mu; mean.v[j] += mv; mean.p[j] += mp; mean.w[j] += mw;
        vr.uu[j] += vu; vr.vv[j] += vv; vr.uv[j] += A2[o] / n - mu * mv; vr.pp[j] += vp; vr.ww[j] += vw;
        const b = hr + xb[x];
        hb[b]++; hb[m + b] += mu; hb[2 * m + b] += mv; hb[3 * m + b] += mp; hb[4 * m + b] += mw;
        hb[5 * m + b] += vu; hb[6 * m + b] += vv; hb[7 * m + b] += vp; hb[8 * m + b] += vw;
      }
    }
  });
  for (let j = 0; j < G.N; j++) {
    const c = cnt[j]; if (!c) continue;
    const mu = mean.u[j] / c, mv = mean.v[j] / c, mp = mean.p[j] / c, mw = mean.w[j] / c;
    out.su[j] = n * mu; out.sv[j] = n * mv; out.sp[j] = n * mp; out.sw[j] = n * mw;
    out.suu[j] = n * (vr.uu[j] / c + mu * mu); out.svv[j] = n * (vr.vv[j] / c + mv * mv); out.suv[j] = n * (vr.uv[j] / c + mu * mv);
    out.spp[j] = n * (vr.pp[j] / c + mp * mp); out.sww[j] = n * (vr.ww[j] / c + mw * mw);
  }
  for (let b = 0; b < m; b++) { // bin means and variances -> equivalent sums, as above
    const c = hb[b]; if (!c) continue;
    for (let q = 1; q <= 4; q++) hb[q * m + b] /= c;
    for (let q = 5; q <= 8; q++) { const mu = hb[(q - 4) * m + b]; hb[q * m + b] = n * (hb[q * m + b] / c + mu * mu); }
    for (let q = 1; q <= 4; q++) hb[q * m + b] *= n;
  }
  out.n = n; out.hi = { cw, ch, d: hb };
  return out;
}
function statsPayload() {
  const R = G.rec, A = R.stats, B = R.base, F = MODE === 'gpu' ? gpuStats() : A;
  const rN = (R.refN || 0) + (R.baseRefN || 0);
  const refRho = rN ? ((R.refSumRho || 0) + (R.baseRefRho || 0)) / rN : R.refRho, refU = rN ? ((R.refSumU || 0) + (R.baseRefU || 0)) / rN : R.refU;
  const o = { n: F.n + (B ? B.n : 0), nF: A.nF, ss: A.ss, sh: A.sh, u0: U0, refRho, refU, refN: rN,
    cpSum: R.cpSum, cpSq: R.cpSq, cpCnt: R.cpCnt };
  for (const key of STAT_KEYS) { const a = new Float32Array(G.N); for (let i = 0; i < G.N; i++) a[i] = F[key][i] + (B ? B[key][i] : 0); o[key] = a; }
  if (F.hi) { // sums add across a resume; the fluid-cell counts do not
    const d = F.hi.d, Bh = B && B.hi && B.hi.d.length === d.length ? B.hi.d : null, m = F.hi.cw * F.hi.ch;
    if (Bh) for (let i = m; i < d.length; i++) d[i] += Bh[i];
    o.hi = F.hi;
  } else if (B && B.hi) o.hi = B.hi;
  return o;
}
async function storeStats(id, stats) { // the high-resolution block is large, so it is stored compressed
  const o = { ...stats }; delete o.hi;
  if (stats.hi) { o.hiDims = [stats.hi.cw, stats.hi.ch]; o.hiZ = await zip(stats.hi.d.buffer.slice(0)); }
  return Store.put(`${id}/m/0`, o);
}
async function loadStats(id) {
  const A = await Store.get(`${id}/m/0`); if (!A) return A;
  if (A.hiZ) { try { A.hi = { cw: A.hiDims[0], ch: A.hiDims[1], d: new Float32Array(await unzip(A.hiZ)) }; } catch (e) { console.error(e); } delete A.hiZ; }
  return A;
}
const ckptEvery = () => 45000 * Math.max(1, G.F.nx * G.F.ny / 1e6); // larger grids take longer to save, so save less often
let cpuStateWaiter = null;
function cpuState() { return new Promise((res) => { cpuStateWaiter = res; engine.post({ type: 'getState' }); setTimeout(() => { if (cpuStateWaiter === res) { cpuStateWaiter = null; res(null); } }, 4000); }); }
async function checkpoint(withState = true) {
  const R = G.rec; if (!R || R.ckptBusy || R.phase !== 'run') return;
  R.ckptBusy = true; R.lastCkpt = performance.now();
  try {
    const M = R.meta;
    flushSeries();
    M.wallSecs += (performance.now() - R.wallStart) / 1000; R.wallStart = performance.now();
    const state = { t: G.t, since: G.since, steps: { ...M.steps }, nextProbe: R.nextProbe, nextFrame: R.nextFrame, frames: M.frames, probeChunks: M.probeChunks, forceChunks: M.forceChunks, refRho: R.refRho, refU: R.refU, rhoRef: G.rhoRef, uRef: G.uRef };
    let f = null, dye = null;
    if (withState) {
      if (MODE === 'gpu') { f = gpu.readPopulations(); dye = gpu.readDye8(); }
      else { const st = await cpuState(); if (st) { f = st.f; state.t = st.t; } dye = Uint8Array.from(G.dye, v => Math.min(255, Math.round(v * 64))); }
    }
    const stats = statsPayload();
    R.io = R.io.then(() => storeStats(M.id, stats)).then(() => f ? Store.put(`${M.id}/k/0`, { state, f, dye }) : null).then(() => Store.putRun({ ...M })).catch(e => console.error(e));
    await R.io;
  } finally { R.ckptBusy = false; }
}
async function finishRecording(early) {
  const R = G.rec; if (!R) return;
  const M = R.meta;
  flushSeries();
  M.wallSecs += (performance.now() - R.wallStart) / 1000;
  if (early) { M.steps.total = M.steps.done; M.plan.ft = +(M.steps.done / M.steps.ft).toFixed(2); }
  M.status = 'done'; M.finished = Date.now(); M.refRho = R.refRho; M.refU = R.refU;
  if (M.wallSecs > 5) { learnBoost(M.steps.done / M.wallSecs, R.liveSps); if (MODE === 'gpu' && !early) learnRecPace(M.steps.done * G.F.nx * G.F.ny / M.wallSecs); }
  const stats = statsPayload();
  G.rec = null;
  await R.io;
  await storeStats(M.id, stats);
  await Store.del(`${M.id}/k/0`);
  await Store.putRun(M);
  setLocked(false); wake(false);
  sendProbes();
  toast(`Recording saved: ${M.frames} frames over ${M.plan.ft} flow-throughs.`);
  renderRecordUI(); refreshRuns();
}
async function abortRecording(msg) {
  const R = G.rec; if (!R) return; const id = R.meta.id;
  G.rec = null; setLocked(false); wake(false);
  try { await R.io; } catch (e) { /* ignore */ }
  await Store.deleteRun(id).catch(() => {});
  if (msg) toast(msg);
  sendProbes();
  renderRecordUI(); refreshRuns();
}

// one animation frame of recording on the GPU: compute-heavy, with the screen refreshed about once a second
function recordFrameGPU() {
  const R = G.rec, M = R.meta;
  if (!S.running || document.hidden) return;
  const budget = 42, t0 = performance.now();
  while (performance.now() - t0 < budget && G.rec) {
    if (R.phase === 'settle') {
      gpu.step(64); G.t = gpu.t; R.advAcc += 64;
      if (R.advAcc >= 48) { advanceScalars(R.advAcc); R.advAcc = 0; }
      const [fx, fy, mz] = gpu.forces(); pushForces(fx, fy, mz, 64); // also a sync point, so the GPU queue never runs far ahead
      if (G.t - R.healthAt > 2000) { R.healthAt = G.t; healthCheckGPU(); if (!G.rec) return; }
      if (settledEnough()) { if (R.advAcc) { advanceScalars(R.advAcc); R.advAcc = 0; } beginCapture(); }
      continue;
    }
    const left = M.steps.total - M.steps.done; if (left <= 0) break;
    const n = Math.max(1, Math.min(R.nextProbe - M.steps.done, R.nextFrame - M.steps.done, left, 64)); // land exactly on every sample and frame
    gpu.step(n); G.t = gpu.t; R.advAcc += n; M.steps.done += n;
    if (R.advAcc >= 48) { advanceScalars(R.advAcc); R.advAcc = 0; }
    if (M.steps.done === R.nextProbe) { recSampleGPU(); R.nextProbe += M.steps.probeEvery; }
    if (M.steps.done === R.nextFrame) { if (R.advAcc) { advanceScalars(R.advAcc); R.advAcc = 0; } captureFrame(); if (!G.rec) return; R.nextFrame += M.steps.frameEvery; }
    if (M.steps.done >= M.steps.total) { finishRecording(false); break; }
  }
  if (G.rec) flushProbeRows();
  const now = performance.now();
  if (G.rec && R.phase === 'run' && now - R.lastCkpt > ckptEvery()) checkpoint(true);
  if (R.phase === 'settle' && now - R.lastRender > 1000) { // keep the picture alive while settling
    const d = gpu.readDown(); for (let i = 0, j = 0; i < G.N; i++, j += 4) { G.rho[i] = d[j]; G.ux[i] = d[j + 1]; G.uy[i] = d[j + 2]; }
    updateRhoRef(1000); dirty = true; R.lastRender = now;
  }
  if (now - R.lastUi > 500) { R.lastUi = now; updateRecProgress(); }
}
function healthCheckGPU() {
  const d = gpu.readDown();
  for (let k = 0; k < 32; k++) { const r = d[4 * ((Math.random() * G.N) | 0)]; if (!(r > 0.5 && r < 1.6)) { abortRecording('The flow diverged before the recording began, so it stopped. Try a lower air speed or a finer grid.'); unstable(); return; } }
}
// CPU path: called for every solver snapshot; probe and force samples arrive from the worker at the exact cadence
function recordSnapshotCPU(m, rows) {
  const R = G.rec, M = R.meta;
  if (R.phase === 'settle') { if (settledEnough()) beginCapture(); updateRecProgress(); return; }
  if (m.n <= 0) return;
  if (R.cpuT0 === undefined) R.cpuT0 = M.steps.t0;
  const np = M.np, qn = 0.5 * U0 * U0 * G.F.Nc;
  for (const r of rows) {
    const t = r[0] - R.cpuT0; if (t < 0 || t > M.steps.total) continue;
    refFromVals(r, 4, np, M.steps.probeEvery);
    if (np) R.probeRows.push(probeRow(t, r.subarray(4, 4 + 4 * np), np, R.refRho, R.refU));
    R.forceRows.push(new Float32Array([t, r[2] / qn, r[1] / qn, -r[3] / (qn * G.F.Nc)]));
  }
  M.steps.done = Math.min(M.steps.total, m.t - R.cpuT0);
  cpuAccumulate();
  if (M.steps.done >= R.nextFrame) { G.rhoRef = R.refRho; G.uRef = R.refU; captureFrame(); if (!G.rec) return; while (R.nextFrame <= M.steps.done) R.nextFrame += M.steps.frameEvery; }
  const now = performance.now();
  if (now - R.lastCkpt > ckptEvery()) checkpoint(true);
  if (now - R.lastUi > 500) { R.lastUi = now; updateRecProgress(); }
  if (M.steps.done >= M.steps.total) finishRecording(false);
}
async function resumeRecording(id) {
  const M = await Store.getRun(id), ck = await Store.get(`${id}/k/0`), saved = await loadStats(id);
  if (!M || !ck || !ck.f) { toast('This recording has no saved state to resume from.'); return; }
  if (M.mode !== MODE) { toast(`This run was recorded with the ${M.mode.toUpperCase()} solver and can't be resumed on this device.`); return; }
  applySettings(M.settings); S.probeRate = (M.plan && M.plan.probeRate) || S.probeRate; initGrid();
  S.probes = (M.probeAbs || []).map(p => ({ ...p })); probeSeq = S.probes.reduce((a, p) => Math.max(a, p.id + 1), 1);
  const st = ck.state, cells = G.F.nx * G.F.ny;
  if (ctxLost || S.res !== M.settings.res || ck.f.length !== 9 * cells) {
    toast(`This recording can't be resumed here: it needs the ${RES_NAME[M.settings.res] || M.settings.res} grid, which this device couldn't build. Keep what was recorded or discard it.`);
    checkResumable(); return;
  }
  if (MODE === 'gpu') { gpu.loadState(ck.f, st.t); if (ck.dye && ck.dye.length === gpu.dw * gpu.dh * 2) gpu.writeDye8(ck.dye); gpu.setProbes(probeLatPts()); gpu.accReset(); }
  else { engine.post({ type: 'setState', f: ck.f, t: st.t }); sendProbes(); if (ck.dye) for (let i = 0; i < G.dye.length && i < ck.dye.length; i++) G.dye[i] = ck.dye[i] / 64; }
  G.t = st.t; G.since = st.since; G.firstRun = false; G.rhoRef = st.rhoRef; G.uRef = st.uRef; G.refInit = true; G.hist = [];
  Object.assign(M.steps, st.steps); M.frames = st.frames; M.probeChunks = st.probeChunks; M.forceChunks = st.forceChunks; M.status = 'recording';
  const A = newStats(G.N, M.cap.cw * M.cap.ch);
  let base = null;
  if (saved && saved.su && saved.su.length === G.N) { // statistics so far become the base the new samples add to
    base = { n: saved.n }; for (const key of STAT_KEYS) base[key] = saved[key];
    if (saved.hi) base.hi = saved.hi;
    if (saved.refN) { G.__baseRef = { n: saved.refN, rho: saved.refRho * saved.refN, u: saved.refU * saved.refN }; }
    A.nF = saved.nF || 0; if (saved.ss && saved.ss.length === A.ss.length) { A.ss.set(saved.ss); A.sh.set(saved.sh); }
    if (MODE === 'cpu') { for (const key of STAT_KEYS) A[key].set(saved[key]); A.n = saved.n; base = null; }
  }
  G.rec = { meta: M, phase: 'run', rowsInTex: 0, rowTimes: [], probeRows: [], forceRows: [], advAcc: 0, nextProbe: st.nextProbe, nextFrame: st.nextFrame,
    stats: A, base, accN: 0, cpSum: saved && saved.cpSum, cpSq: saved && saved.cpSq, cpCnt: saved && saved.cpCnt, refRho: st.refRho, refU: st.refU, refInit: true, cpuT0: st.t - st.steps.done,
    io: Promise.resolve(), lastCkpt: performance.now(), lastUi: 0, lastRender: 0, wallStart: performance.now(), etaT0: performance.now(), etaS0: st.steps.done, healthAt: G.t };
  if (G.__baseRef) { G.rec.baseRefN = G.__baseRef.n; G.rec.baseRefRho = G.__baseRef.rho; G.rec.baseRefU = G.__baseRef.u; G.__baseRef = null; }
  S.running = true; syncRunButton(); setLocked(true); wake(true);
  if (MODE === 'cpu') engine.post({ type: 'run', on: true });
  renderProbeUI(); renderRecordUI(); toast('Recording resumed from its last checkpoint.');
}
document.addEventListener('visibilitychange', () => {
  if (document.hidden && G && G.rec && G.rec.phase === 'run') checkpoint(true);
  if (!document.hidden && G && G.rec) wake(true);
});
window.addEventListener('pagehide', () => { if (G && G.rec && G.rec.phase === 'run') checkpoint(true); });

// ---------- spectra ----------
function fftPower(sig) { // Hann-windowed power spectrum, zero-padded to a power of two (at least 4x)
  const n0 = sig.length; let n = 1; while (n < 4 * n0) n <<= 1;
  const re = new Float64Array(n), im = new Float64Array(n);
  let mu = 0, c = 0; for (const v of sig) if (isFinite(v)) { mu += v; c++; } mu = c ? mu / c : 0;
  for (let i = 0; i < n0; i++) { const v = isFinite(sig[i]) ? sig[i] - mu : 0; re[i] = v * (0.5 - 0.5 * Math.cos(2 * Math.PI * i / Math.max(1, n0 - 1))); }
  for (let i = 1, j = 0; i < n; i++) { let bit = n >> 1; for (; j & bit; bit >>= 1) j ^= bit; j ^= bit; if (i < j) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; } }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = -2 * Math.PI / len, wr = Math.cos(ang), wi = Math.sin(ang), h = len >> 1;
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < h; k++) {
        const a = i + k, b = a + h, vr = re[b] * cr - im[b] * ci, vi = re[b] * ci + im[b] * cr;
        re[b] = re[a] - vr; im[b] = im[a] - vi; re[a] += vr; im[a] += vi;
        const t = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = t;
      }
    }
  }
  const P = new Float64Array(n / 2 + 1); for (let k = 0; k <= n / 2; k++) P[k] = re[k] * re[k] + im[k] * im[k];
  return { P, n };
}
function welch(sig, dt) { // Welch PSD: Hann segments with 50% overlap; one-sided, units of signal^2 per Hz
  const n0 = sig.length; let mu = 0, c = 0; for (const v of sig) if (isFinite(v)) { mu += v; c++; } mu = c ? mu / c : 0;
  // 7 half-overlapping segments for long records (1,600+ samples), 3 for medium ones, a single window for short ones
  const L7 = Math.floor(2 * n0 / 8), L3 = Math.floor(2 * n0 / 4);
  const [K, L] = L7 >= 400 ? [7, L7] : L3 >= 400 ? [3, L3] : [1, n0], hop = Math.floor(L / 2); // keep segments long enough to resolve the shedding peak
  let nfft = 1; while (nfft < 2 * L) nfft <<= 1;
  const acc = new Float64Array(nfft / 2 + 1), win = new Float64Array(L); let wss = 0;
  for (let i = 0; i < L; i++) { win[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / Math.max(1, L - 1)); wss += win[i] * win[i]; }
  let used = 0;
  for (let s = 0; s + L <= n0 && used < K; s += hop, used++) {
    const re = new Float64Array(nfft), im = new Float64Array(nfft);
    for (let i = 0; i < L; i++) { const v = sig[s + i]; re[i] = (isFinite(v) ? v - mu : 0) * win[i]; }
    fftInPlace(re, im);
    for (let k = 0; k <= nfft / 2; k++) acc[k] += (re[k] * re[k] + im[k] * im[k]) * (k === 0 || k === nfft / 2 ? 1 : 2) * dt / wss;
  }
  for (let k = 0; k < acc.length; k++) acc[k] /= Math.max(1, used);
  return { P: acc, nfft, df: 1 / (nfft * dt), segs: used, res: 1 / (L * dt) };
}
function fftInPlace(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) { let bit = n >> 1; for (; j & bit; bit >>= 1) j ^= bit; j ^= bit; if (i < j) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; } }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = -2 * Math.PI / len, wr = Math.cos(ang), wi = Math.sin(ang), h = len >> 1;
    for (let i = 0; i < n; i += len) { let cr = 1, ci = 0; for (let k = 0; k < h; k++) { const a = i + k, b = a + h, vr = re[b] * cr - im[b] * ci, vi = re[b] * ci + im[b] * cr; re[b] = re[a] - vr; im[b] = im[a] - vi; re[a] += vr; im[a] += vi; const t = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = t; } }
  }
}
// dominant frequency, reported only when the peak clearly stands out: >= 15 dB above the median of the spectrum
// and the signal rms above a floor; resolution from the record length
function peakSt(sig, dt, chord, speed, rmsFloor = 1e-3) {
  if (sig.length < 32) return null;
  let mu = 0; for (const v of sig) mu += v; mu /= sig.length; let s2 = 0; for (const v of sig) s2 += (v - mu) ** 2;
  if (Math.sqrt(s2 / sig.length) < rmsFloor) return null;
  const { P, n } = fftPower(sig), T = sig.length * dt, kmin = Math.max(2, Math.ceil(2 / T * n * dt));
  let kb = kmin; for (let k = kmin; k < P.length - 1; k++) if (P[k] > P[kb]) kb = k;
  if (kb <= kmin || kb >= P.length - 1) return null;
  const band = Array.from(P.slice(kmin)).sort((x, y) => x - y), med = band[band.length >> 1] || 0;
  if (!(P[kb] > 31.6 * med)) return null;
  const a = P[kb - 1], b = P[kb], c = P[kb + 1], d = 0.5 * (a - c) / (a - 2 * b + c || 1);
  const f = (kb + d) / (n * dt);
  return { St: f * chord / speed, f, cycles: f * T, dSt: chord / (speed * T), snr: 10 * Math.log10(P[kb] / Math.max(1e-30, med)) };
}

// ---------- run statistics ----------
async function loadSeries(M, kind, width, count) {
  const parts = [];
  for (let i = 0; i < count; i++) { const z = await Store.get(`${M.id}/${kind}/${i}`); if (z) parts.push(new Float32Array(await unzip(z))); }
  const rows = []; for (const a of parts) for (let o = 0; o + width <= a.length; o += width) rows.push(a.subarray(o, o + width));
  return rows;
}

// ---------- playback ----------
const SPEEDS = [[1, 'Real time'], [0.5, '1/2 ×'], [0.2, '1/5 ×'], [0.1, '1/10 ×'], [0.05, '1/20 ×'], [0.02, '1/50 ×'], [0.01, '1/100 ×'], [0.005, '1/200 ×'], [0.002, '1/500 ×']];
async function openReview(id) {
  if (G.rec) { toast('Finish or stop the recording before reviewing a run.'); return; }
  const M = await Store.getRun(id); if (!M) return;
  if (M.mode !== MODE) { toast(`This run was recorded with the ${M.mode.toUpperCase()} solver and can't be shown on this device.`); return; }
  const prevProbes = S.probes, prevSeq = probeSeq;
  applySettings(M.settings); initGrid({ lite: true }); // the solver is not needed to replay a run
  S.running = false; syncRunButton(); if (MODE === 'cpu') engine.post({ type: 'run', on: false });
  S.probes = (M.probeAbs || []).map(p => ({ ...p }));
  const means = await loadStats(M.id);
  const forces = await loadSeries(M, 'c', 4, M.forceChunks);
  const probes = M.np ? await loadSeries(M, 'p', 1 + 3 * M.np, M.probeChunks) : [];
  const frameDt = M.steps.frameEvery * M.dtStep, sampleDt = M.steps.probeEvery * M.dtStep;
  const shed = peakSt(forces.map(r => r[1]), sampleDt, M.chord, M.speed, 1e-3);
  let speed = SPEEDS[SPEEDS.length - 1][0]; for (const [v] of SPEEDS) if (v / frameDt <= 12) { speed = v; break; }
  G.play = { M, means, forces, probes, shed, frames: new Map(), loading: new Set(), idx: -1, want: 0, time: 0, playing: true, speed, view: 'inst', frameDt, sampleDt, last: performance.now(), prevProbes, prevSeq, mv: null };
  G.hist = forces.map(r => ({ t: r[0], cl: r[1], cd: r[2], cm: r[3] })); // run time: t = 0 at the first sample
  G.since = 1e12; G.firstRun = false; G.stag = null; G.arc = null; G.cp = null; G.t = 0;
  if (means && means.cpSum && G.polyF && means.cpSum.length === G.polyF.length) { // run-averaged surface pressure
    const n = means.cpSum.length, C = { n, m: new Float32Array(n), v: new Float32Array(n), ok: new Uint8Array(n), w: 1 };
    for (let k = 0; k < n; k++) { const c = means.cpCnt[k]; if (c) { const mu = means.cpSum[k] / c; C.m[k] = mu; C.v[k] = Math.max(0, means.cpSq[k] / c - mu * mu); C.ok[k] = 1; } }
    G.cp = C;
  }
  $('#fhNote').textContent = 'whole run'; $('#cpNote').innerHTML = '−C<sub>p</sub> vs x/c, run average';
  setLocked(true);
  $('#toolbar').hidden = true; $('#player').hidden = false;
  const sp = $('#pbSpeed'); sp.innerHTML = SPEEDS.map(([v, t]) => `<option value="${v}">${t}</option>`).join(''); sp.value = String(speed);
  $('#pbView').value = 'inst';
  const sc = $('#pbScrub'); sc.max = String(Math.max(0, M.frames - 1)); sc.value = '0';
  const warn = $('#pbWarn');
  if (shed && frameDt > 0.25 / shed.f) {
    warn.textContent = `Frames are ${(frameDt * shed.f).toFixed(2)} shedding periods apart, so vortices will appear to jump or stand still. For smooth motion record at least ${Math.ceil(4 * shed.f * M.steps.ft * M.dtStep)} frames per flow-through.`;
    warn.hidden = false;
  } else warn.hidden = true;
  syncPlayButton(); renderProbeUI(); renderRecordUI(); updateState(null);
  await loadFrame(0); showFrame(0);
}
function closeReview() {
  const P = G.play; if (!P) return;
  G.play = null; S.probes = P.prevProbes; probeSeq = P.prevSeq;
  $('#player').hidden = true; $('#toolbar').hidden = false; $('#pbWarn').hidden = true;
  setLocked(false); initGrid(); S.running = true; syncRunButton();
  if (MODE === 'cpu') engine.post({ type: 'run', on: true });
  $('#fhNote').textContent = 'last 4 flow-throughs'; $('#cpNote').innerHTML = '−C<sub>p</sub> vs x/c, averaged';
  renderProbeUI(); renderRecordUI();
}
async function loadFrame(i) {
  const P = G.play; if (!P || P.frames.has(i) || P.loading.has(i) || i < 0 || i >= P.M.frames) return;
  P.loading.add(i);
  try {
    const z = await Store.get(`${P.M.id}/f/${i}`); if (!z || G.play !== P) return;
    P.frames.set(i, unpackFrame(await unzip(z)));
    if (P.frames.size > 24) { for (const k of P.frames.keys()) { if (Math.abs(k - P.want) > 10) P.frames.delete(k); if (P.frames.size <= 16) break; } }
  } catch (e) { console.error(e); } finally { P.loading.delete(i); }
}
function meanFrame() { // time-averaged or fluctuation fields (float)
  const P = G.play; if (P.mv && P.mv.view === P.view) return P.mv;
  const A = P.means, N = G.N, M = P.M, m = M.cap.cw * M.cap.ch, rms = P.view === 'rms';
  const ux = new Float32Array(N), uy = new Float32Array(N), cp = new Float32Array(N), vort = new Float32Array(N), sm = new Uint8Array(2 * m);
  if (A && A.n) {
    const n = A.n, u0 = A.u0 || U0, rr = A.refRho || 1, ur = A.refU || U0, q = 2 / (3 * ur * ur);
    for (let i = 0; i < N; i++) {
      if (G.mask[i]) continue;
      const mu = A.su[i] / n, mv = A.sv[i] / n, mp = A.sp[i] / n, mw = A.sw[i] / n;
      if (rms) {
        ux[i] = Math.sqrt(Math.max(0, A.suu[i] / n - mu * mu)); uy[i] = Math.sqrt(Math.max(0, A.svv[i] / n - mv * mv));
        cp[i] = Math.sqrt(Math.max(0, A.spp[i] / n - mp * mp)) * q; vort[i] = Math.sqrt(Math.max(0, A.sww[i] / n - mw * mw));
      } else { ux[i] = u0 + mu; uy[i] = mv; cp[i] = (mp - (rr - 1)) * q; vort[i] = mw; }
    }
    const nF = Math.max(1, A.nF || 1);
    for (let i = 0; i < m; i++) { sm[2 * i] = Math.min(255, Math.round(A.ss[i] / nF)); sm[2 * i + 1] = Math.min(255, Math.round(A.sh[i] / nF)); }
  }
  const hi = A && A.n && A.hi && MODE === 'gpu' ? hiFields(A, rms) : null;
  return (P.mv = { view: P.view, float: true, t: 0, nx: G.nx, ny: G.ny, vw: G.nx, vh: G.ny, ux, uy, cp, vort, sm, hi });
}
function hiFields(A, rms) { // display fields on the capture grid from the high-resolution statistics
  const { cw, ch, d } = A.hi, m = cw * ch, n = A.n, u0 = A.u0 || U0, rr = A.refRho || 1, ur = A.refU || U0, q = 2 / (3 * ur * ur);
  const speed = new Float32Array(m), cp = new Float32Array(m), vort = new Float32Array(m), ok = new Uint8Array(m);
  for (let b = 0; b < m; b++) {
    if (!d[b]) continue; ok[b] = 1;
    const mu = d[m + b] / n, mv = d[2 * m + b] / n, mp = d[3 * m + b] / n, mw = d[4 * m + b] / n;
    if (rms) {
      speed[b] = Math.sqrt(Math.max(0, d[5 * m + b] / n - mu * mu) + Math.max(0, d[6 * m + b] / n - mv * mv)) / U0;
      cp[b] = Math.sqrt(Math.max(0, d[7 * m + b] / n - mp * mp)) * q; vort[b] = Math.sqrt(Math.max(0, d[8 * m + b] / n - mw * mw));
    } else { speed[b] = Math.hypot(u0 + mu, mv) / U0; cp[b] = (mp - (rr - 1)) * q; vort[b] = mw; }
  }
  // carry the nearest fluid values a few bins into the model, so the bicubic display does not pull the edge toward zero
  for (let pass = 0; pass < 3; pass++) {
    const add = [];
    for (let y = 0; y < ch; y++) for (let x = 0; x < cw; x++) {
      const b = y * cw + x; if (ok[b]) continue;
      let c = 0, s0 = 0, s1 = 0, s2 = 0;
      for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) { const X = x + dx, Y = y + dy; if (X < 0 || Y < 0 || X >= cw || Y >= ch) continue; const e = Y * cw + X; if (ok[e]) { c++; s0 += speed[e]; s1 += cp[e]; s2 += vort[e]; } }
      if (c) add.push(b, s0 / c, s1 / c, s2 / c);
    }
    for (let i = 0; i < add.length; i += 4) { const b = add[i]; speed[b] = add[i + 1]; cp[b] = add[i + 2]; vort[b] = add[i + 3]; ok[b] = 1; }
  }
  return { cw, ch, speed, cp, vort };
}
function showFrame(i) {
  const P = G.play; if (!P) return;
  const fr = P.view === 'inst' ? P.frames.get(i) : meanFrame(); if (!fr) return;
  applyFrame(fr, i);
}
// Motion-compensated in-between frame: frame i carried forward along its own velocity by s of a frame interval, frame
// i + 1 carried back by 1 - s, then blended. Used for smooth video when frames are shown slower than they were recorded.
function blendFrames(A, B, s) {
  const P = G.play, M = P.M, nx = A.nx, ny = A.ny, cw = A.cw, ch = A.ch, m = cw * ch, N = nx * ny;
  const su = U0 / QU, D = M.steps.frameEvery, k = M.grid.F.k;
  const dR = D / k * su; // displacement per frame interval per unit of stored velocity, in readback cells
  const lerp2 = (a, w, h, x, y) => { // bilinear sample of an Int16/Uint8 array, clamped
    x = x < 0 ? 0 : x > w - 1 ? w - 1 : x; y = y < 0 ? 0 : y > h - 1 ? h - 1 : y;
    const x0 = x | 0, y0 = y | 0, x1 = Math.min(w - 1, x0 + 1), y1 = Math.min(h - 1, y0 + 1), fx = x - x0, fy = y - y0;
    const r0 = y0 * w, r1 = y1 * w;
    return (a[r0 + x0] * (1 - fx) + a[r0 + x1] * fx) * (1 - fy) + (a[r1 + x0] * (1 - fx) + a[r1 + x1] * fx) * fy;
  };
  // trace along the recorded velocity in a few sub-steps when the shift is large, so curved paths round vortices are followed
  let umax = 0; for (let i = 0; i < N; i++) umax = Math.max(umax, Math.abs(A.ux[i]), Math.abs(A.uy[i]), Math.abs(B.ux[i]), Math.abs(B.uy[i]));
  const nSub = Math.max(1, Math.min(4, Math.ceil(umax * dR * Math.max(s, 1 - s) / 2)));
  let tx = 0, ty = 0;
  const trace = (F, x, y, d) => { const h = d / nSub; for (let q = 0; q < nSub; q++) { const u = lerp2(F.ux, nx, ny, x, y), v = lerp2(F.uy, nx, ny, x, y); x -= h * u; y -= h * v; } tx = x; ty = y; };
  const ux = new Float32Array(N), uy = new Float32Array(N), cp = new Float32Array(N), mask = G.mask;
  for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
    const i = y * nx + x; if (mask[i]) continue;
    trace(A, x, y, s * dR); const ax = tx, ay = ty; trace(B, x, y, -(1 - s) * dR); const bx = tx, by = ty;
    ux[i] = ((1 - s) * lerp2(A.ux, nx, ny, ax, ay) + s * lerp2(B.ux, nx, ny, bx, by)) * su;
    uy[i] = ((1 - s) * lerp2(A.uy, nx, ny, ax, ay) + s * lerp2(B.uy, nx, ny, bx, by)) * su;
    cp[i] = ((1 - s) * lerp2(A.cp, nx, ny, ax, ay) + s * lerp2(B.cp, nx, ny, bx, by)) / QC;
  }
  const vort = new Float32Array(m), sm = new Uint8Array(2 * m), sA = new Uint8Array(m), hA = new Uint8Array(m), sB = new Uint8Array(m), hB = new Uint8Array(m);
  for (let i = 0; i < m; i++) { sA[i] = A.sm[2 * i]; hA[i] = A.sm[2 * i + 1]; sB[i] = B.sm[2 * i]; hB[i] = B.sm[2 * i + 1]; }
  const gx = nx / cw, gy = ny / ch; // capture pixel -> readback cell
  for (let y = 0; y < ch; y++) for (let x = 0; x < cw; x++) {
    const i = y * cw + x, rx = (x + 0.5) * gx - 0.5, ry = (y + 0.5) * gy - 0.5;
    trace(A, rx, ry, s * dR); const ax = (tx + 0.5) / gx - 0.5, ay = (ty + 0.5) / gy - 0.5;
    trace(B, rx, ry, -(1 - s) * dR); const bx = (tx + 0.5) / gx - 0.5, by = (ty + 0.5) / gy - 0.5;
    vort[i] = ((1 - s) * lerp2(A.vort, cw, ch, ax, ay) + s * lerp2(B.vort, cw, ch, bx, by)) / QV;
    sm[2 * i] = Math.round((1 - s) * lerp2(sA, cw, ch, ax, ay) + s * lerp2(sB, cw, ch, bx, by));
    sm[2 * i + 1] = Math.round((1 - s) * lerp2(hA, cw, ch, ax, ay) + s * lerp2(hB, cw, ch, bx, by));
  }
  return { float: true, blend: true, t: A.t + s * (B.t - A.t), nx, ny, cw, ch, vw: cw, vh: ch, ux, uy, cp, vort, sm };
}
function applyFrame(fr, i) {
  const P = G.play;
  P.idx = i; P.cur = fr;
  P.hi = fr.hi || null;
  if (fr.float) { G.ux.set(fr.ux); G.uy.set(fr.uy); P.cp = fr.cp; P.vort = fr.hi ? fr.hi.vort : fr.vort; P.vw = fr.hi ? fr.hi.cw : fr.vw; P.vh = fr.hi ? fr.hi.ch : fr.vh; if (fr.blend) G.t = fr.t; }
  else {
    const su = U0 / QU; for (let k = 0; k < G.N; k++) { G.ux[k] = fr.ux[k] * su; G.uy[k] = fr.uy[k] * su; }
    P.cp = Float32Array.from(fr.cp, v => v / QC); P.vort = Float32Array.from(fr.vort, v => v / QV); P.vw = fr.cw; P.vh = fr.ch;
    G.t = fr.t;
  }
  if (P.view === 'rms') { for (let k = 0; k < G.N; k++) { G.ux[k] = Math.hypot(G.ux[k], G.uy[k]); G.uy[k] = 0; } } // fluctuation magnitude, for display only
  P.sm = fr.sm;
  G.rhoRef = 1; G.uRef = U0;
  const q = 1.5 * U0 * U0; for (let k = 0; k < G.N; k++) G.rho[k] = 1 + P.cp[k] * q; // for the probe readout
  if (!G.cpP || G.cpP.length !== G.N) G.cpP = new Float32Array(G.N);
  G.cpP.set(P.cp); G.pw = new Float32Array(G.N); G.pmgValid = true;
  if (MODE === 'cpu') { const s = P.sm; for (let k = 0; k < G.N; k++) G.dye[k] = s[2 * k] / 255 / 0.32; }
  $('#pbScrub').value = String(i);
  updatePlayLabel(); dirty = true;
}
function playbackFrame() {
  const P = G.play, now = performance.now(), dt = (now - P.last) / 1000; P.last = now;
  const loop = P.M.frames * P.frameDt; // each frame is shown for one frame interval
  if (P.playing && P.view === 'inst' && P.M.frames > 1) { P.time += dt * P.speed; if (P.time >= loop) P.time %= loop; }
  const want = P.view === 'inst' ? Math.min(P.M.frames - 1, Math.floor(P.time / P.frameDt)) : P.idx < 0 ? 0 : P.idx;
  P.want = want;
  for (let k = 0; k < 6; k++) loadFrame((want + k) % Math.max(1, P.M.frames));
  if (want !== P.idx && P.frames.has(want)) showFrame(want);
  if (S.field === 'piv' && S.pivView === 'raw' && P.playing) { advectPIV(Math.min(40, dt * P.speed / P.M.dtStep)); dirty = true; }
}
function updatePlayLabel() {
  const P = G.play; if (!P) return;
  const ms = P.idx * P.frameDt * 1000, tot = P.M.steps.total * P.M.dtStep * 1000;
  const A = P.means, ft = P.M.steps.total / P.M.steps.ft;
  const what = `${A ? A.n.toLocaleString() : 0} samples over ${ft.toFixed(1)} flow-through${ft.toFixed(1) === '1.0' ? '' : 's'}${P.shed ? `, about ${Math.round(P.shed.cycles)} shedding cycles` : ''}`;
  const conv = P.shed && P.shed.cycles < 20 ? ' · under 20 cycles, so treat the statistics as preliminary' : '';
  $('#pbTime').textContent = P.view === 'inst' ? `${ms.toFixed(1)} / ${tot.toFixed(1)} ms · frame ${P.idx + 1}/${P.M.frames}` : (P.view === 'mean' ? `Average of ${what}${conv}` : `RMS of ${what}${conv}`);
  const fps = P.speed / P.frameDt;
  $('#pbFps').textContent = P.view === 'inst' ? `${fps >= 1 ? Math.round(fps) : fps.toFixed(1)} frames/s` : '';
}
function syncPlayButton() { const b = $('#pbPlay'); if (!b || !G.play) return; b.innerHTML = G.play.playing ? ICON.pause : ICON.play; b.setAttribute('aria-label', G.play.playing ? 'Pause' : 'Play'); }
// scalar field for the GPU display in playback
function playbackScalar(mode) {
  const P = G.play, M = P.M;
  if (mode === 1) gpu.pbScalar(P.vort, P.vw, P.vh);
  else if (P.hi) gpu.pbScalar(mode === 2 ? P.hi.cp : P.hi.speed, P.hi.cw, P.hi.ch); // averaged views: high-resolution statistics
  else if (mode === 2) gpu.pbScalar(P.cp, G.nx, G.ny);
  else {
    const s = new Float32Array(G.N); for (let k = 0; k < G.N; k++) s[k] = G.mask[k] ? 0 : Math.hypot(G.ux[k], G.uy[k]) / U0;
    gpu.pbScalar(s, G.nx, G.ny);
  }
  if (P.sm) gpu.pbDye(P.sm, M.cap.cw, M.cap.ch);
}
function playRange() { // colour ranges for the fluctuation view
  const P = G.play; if (!P || P.view !== 'rms') return null;
  const k = S.contrast, f = S.field;
  if (f === 'vort') return { lo: 0, mid: 10 / k, hi: 20 / k, div: false, step: 2 };
  if (f === 'press') return { lo: 0, mid: 0.2 / k, hi: 0.4 / k, div: false, step: 0.04 };
  return { lo: 0, mid: 0.25 / k, hi: 0.5 / k, div: false, step: 0.05 };
}

// ---------- locking controls while recording or reviewing ----------
function setLocked(on) {
  for (const sec of CTRL.slice(0, 3)) for (const c of sec.items) if (c.wrap) c.wrap.querySelectorAll('input,select,button').forEach(el => { el.disabled = on; });
  for (const id of ['btnReset', 'btnCentre']) { const b = $('#' + id); if (b) b.disabled = on; }
  document.querySelectorAll('#secProbes input, #secProbes button').forEach(el => { if (el.id !== 'probeCsv' && el.id !== 'probeSel') el.disabled = on; });
  $('#controls').classList.toggle('locked', on);
}
function syncRunButton() { const b = $('#btnRun'); b.textContent = S.running ? 'Pause' : 'Run'; b.setAttribute('aria-pressed', String(S.running)); }

// ---------- icons ----------
const ICON = {
  play: '<svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true"><path d="M4 2.5v11l9-5.5z" fill="currentColor"/></svg>',
  pause: '<svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true"><path d="M3.5 2.5h3v11h-3zM9.5 2.5h3v11h-3z" fill="currentColor"/></svg>',
  prev: '<svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true"><path d="M3 2.5h2v11H3zM14 2.5v11L6 8z" fill="currentColor"/></svg>',
  next: '<svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true"><path d="M11 2.5h2v11h-2zM2 2.5v11L10 8z" fill="currentColor"/></svg>',
};

// ---------- UI: probes section ----------
function buildFeatureUI() {
  const ps = document.createElement('section'); ps.className = 'sec'; ps.id = 'secProbes';
  ps.innerHTML = `<h2>Probes<small id="probeTag"></small></h2>
    <div class="ctl"><div class="checks"><label><input type="checkbox" id="probeTap"> Tap the flow to add a probe, tap a probe to remove it</label></div></div>
    <div class="ctl"><label>Probe grid</label><span class="note-inline" id="pgCount"></span>
      <div class="grid2">
        <label for="pgNx">Columns<input type="number" id="pgNx" min="1" max="30" step="1" value="6" inputmode="numeric"></label>
        <label for="pgNy">Rows<input type="number" id="pgNy" min="1" max="30" step="1" value="5" inputmode="numeric"></label>
        <label for="pgDx">Spacing x (c)<input type="number" id="pgDx" min="0.01" max="5" step="0.05" value="0.25" inputmode="decimal"></label>
        <label for="pgDy">Spacing y (c)<input type="number" id="pgDy" min="0.01" max="5" step="0.05" value="0.20" inputmode="decimal"></label>
        <label for="pgCx">Centre x/c<input type="number" id="pgCx" step="0.1" value="1.60" inputmode="decimal"></label>
        <label for="pgCy">Centre y/c<input type="number" id="pgCy" step="0.1" value="0.00" inputmode="decimal"></label>
      </div>
      <div class="row"><button type="button" id="pgAdd">Add probe grid</button></div>
      <p class="note">Positions are in tunnel axes from the leading edge, in chords: x downstream, y up. Probes that would sit inside the model are skipped.</p>
    </div>
    <div class="ctl"><label for="probeRate">Sample rate</label><span class="note-inline" id="probeHz"></span>
      <select id="probeRate"><option value="100">100 per flow-through</option><option value="200" selected>200 per flow-through</option><option value="500">500 per flow-through</option><option value="1000">1,000 per flow-through</option></select>
      <p class="note">Probes and forces are sampled at an exact, fixed step interval. While recording, the field statistics for the averaged views use the same rate.</p>
    </div>
    <div class="ctl"><label>Probe data</label><span class="note-inline" id="probeCount">no probes</span>
      <div class="row"><button type="button" id="probeCsv">Export probe data (CSV)</button><button type="button" id="probeClear">Remove all</button></div>
      <p class="note">The live export holds the last four flow-throughs. A recording keeps every sample for the whole run.</p>
    </div>`;
  const rs = document.createElement('section'); rs.className = 'sec'; rs.id = 'secRecord';
  rs.innerHTML = `<h2>Record &amp; review<small id="recTag"></small></h2>
    <div id="recResume" class="card" hidden></div>
    <div id="recSetup">
      <div class="ctl"><label for="recFT">Run length</label><output id="recFT_o"></output><input type="range" id="recFT" min="1" max="50" step="1" value="${S.recFT}"></div>
      <div class="ctl"><label for="recFPF">Frames kept per flow-through</label><output id="recFPF_o"></output><input type="range" id="recFPF" min="4" max="60" step="1" value="${S.recFPF}"></div>
      <p class="note" id="recEst"></p>
      <div class="row"><button type="button" class="primary" id="recStart">Start recording</button></div>
      <p class="note">Place probes before you start: each one is sampled at the rate set under Probes for the whole run.</p>
      <p class="note">While it records, the screen updates about once a second so the device can put its effort into the solver. Keep the page open and the device plugged in. If the page closes, the run resumes from its last checkpoint next time you open it on this device.</p>
    </div>
    <div id="recProgress" hidden>
      <div class="bar"><i id="recBarFill"></i></div>
      <p class="rec-status" id="recStatus"></p>
      <div class="row"><button type="button" id="recStop">Stop and keep</button><button type="button" id="recDiscard">Discard</button></div>
    </div>
    <div class="ctl runs-head"><label>Recorded runs</label><span class="note-inline" id="storeUse"></span></div>
    <ul class="runs" id="runList"><li class="empty">No runs recorded on this device yet.</li></ul>`;
  panel.appendChild(ps); panel.appendChild(rs);

  // probe events
  $('#probeTap').addEventListener('change', (e) => { S.probeTap = e.target.checked; view.classList.toggle('probe-mode', S.probeTap); });
  const gridVals = () => ({ nx: Math.max(1, Math.min(30, Math.round(+$('#pgNx').value || 1))), ny: Math.max(1, Math.min(30, Math.round(+$('#pgNy').value || 1))),
    dx: Math.max(0.01, +$('#pgDx').value || 0.25), dy: Math.max(0.01, +$('#pgDy').value || 0.2), cx: +$('#pgCx').value || 0, cy: +$('#pgCy').value || 0 });
  const showGridCount = () => { const v = gridVals(); $('#pgCount').textContent = `${v.nx * v.ny} probes`; };
  ['pgNx', 'pgNy', 'pgDx', 'pgDy', 'pgCx', 'pgCy'].forEach(id => $('#' + id).addEventListener('input', showGridCount)); showGridCount();
  $('#pgAdd').addEventListener('click', () => {
    const v = gridVals(); let added = 0, skipped = 0;
    for (let j = 0; j < v.ny; j++) for (let i = 0; i < v.nx; i++) {
      const x = v.cx + (i - (v.nx - 1) / 2) * v.dx, y = v.cy + (j - (v.ny - 1) / 2) * v.dy;
      if (addProbe(absX(x), absY(y))) added++; else skipped++;
    }
    probesChanged();
    toast(skipped ? `Added ${added} probes; ${skipped} would have been inside the model or outside the test section.` : `Added ${added} probes.`);
  });
  $('#probeCsv').addEventListener('click', () => { if (G.play) exportRunCSV(G.play.M.id, 'p'); else exportLiveProbes(); });
  $('#probeClear').addEventListener('click', () => { S.probes = []; probesChanged(); });
  $('#probeSel').addEventListener('change', () => { dirty = true; });
  const showHz = () => { if (G) $('#probeHz').textContent = `${Math.round(1 / (probeEvery() * flowTime(1))).toLocaleString()} Hz`; };
  $('#probeRate').addEventListener('change', (e) => { S.probeRate = +e.target.value; sendProbes(); if (G) G.pbuf = newProbeBuffer(); showHz(); });
  showHz(); setInterval(() => { if (G && !G.rec && !G.play) showHz(); }, 2000);
  $('#probeVar').addEventListener('change', () => {});

  // record events
  const recOut = () => {
    $('#recFT_o').textContent = `${S.recFT} flow-throughs`;
    $('#recFPF_o').textContent = `${S.recFPF}`;
    if (G) {
      const e = recEstimate();
      const ck = MODE === 'gpu' ? planCheck(gridPlan(S.res)) : null, memWarn = ck && (ck.level === 'tight' || ck.level === 'over') ? ` Recording on this grid needs ≈ ${fmtGB(gridPlan(S.res).needRec)} of memory, which may be more than this device has.` : '';
      $('#recEst').textContent = `${e.frames} frames, ${e.realMs.toFixed(0)} ms of real tunnel time. About ${fmtDur(e.secs)} on this device and ${fmtBytes(e.bytes)} of storage.${memWarn}`;
    }
  };
  $('#recFT').addEventListener('input', (e) => { S.recFT = +e.target.value; recOut(); });
  $('#recFPF').addEventListener('input', (e) => { S.recFPF = +e.target.value; recOut(); });
  recOut(); setInterval(() => { if (!G.rec && !G.play) recOut(); }, 2000);
  $('#recStart').addEventListener('click', startRecording);
  $('#recStop').addEventListener('click', () => { const R = G.rec; if (!R) return; if (R.phase === 'settle') abortRecording('Recording cancelled before it began.'); else finishRecording(true); });
  $('#recDiscard').addEventListener('click', () => {
    const b = $('#recDiscard');
    if (b.dataset.armed) { abortRecording('Recording discarded.'); delete b.dataset.armed; b.textContent = 'Discard'; }
    else { b.dataset.armed = '1'; b.textContent = 'Tap again to discard'; setTimeout(() => { delete b.dataset.armed; b.textContent = 'Discard'; }, 4000); }
  });

  // player
  $('#player').innerHTML = `<div class="prow">
      <button type="button" class="icon" id="pbPrev" aria-label="Previous frame">${ICON.prev}</button>
      <button type="button" class="icon primary" id="pbPlay" aria-label="Pause">${ICON.pause}</button>
      <button type="button" class="icon" id="pbNext" aria-label="Next frame">${ICON.next}</button>
      <input type="range" id="pbScrub" min="0" max="0" step="1" value="0" aria-label="Frame">
    </div>
    <div class="prow">
      <span class="mono" id="pbTime"></span><span class="mono muted" id="pbFps"></span>
      <label class="inline" for="pbSpeed">Speed <select id="pbSpeed"></select></label>
      <label class="inline" for="pbView">Show <select id="pbView"><option value="inst">Instantaneous</option><option value="mean">Time-averaged</option><option value="rms">Fluctuations (rms)</option></select></label>
      <button type="button" id="pbImg">Save image</button>
      <button type="button" id="pbVid">Save video</button>
      <button type="button" id="pbExit">Back to live tunnel</button>
    </div>
    <p class="note pbwarn" id="pbWarn" hidden></p>`;
  $('#pbPlay').addEventListener('click', () => { const P = G.play; if (!P) return; P.playing = !P.playing; P.last = performance.now(); syncPlayButton(); });
  const step = (d) => { const P = G.play; if (!P) return; P.playing = false; syncPlayButton(); P.view = 'inst'; $('#pbView').value = 'inst'; const i = (Math.max(0, P.idx) + d + P.M.frames) % P.M.frames; P.time = (i + 0.5) * P.frameDt; loadFrame(i).then(() => showFrame(i)); };
  $('#pbPrev').addEventListener('click', () => step(-1)); $('#pbNext').addEventListener('click', () => step(1));
  $('#pbScrub').addEventListener('input', (e) => { const P = G.play; if (!P) return; const i = +e.target.value; P.time = (i + 0.5) * P.frameDt; P.want = i; if (P.view !== 'inst') { P.view = 'inst'; $('#pbView').value = 'inst'; } loadFrame(i).then(() => { if (G.play === P) showFrame(i); }); });
  $('#pbSpeed').addEventListener('change', (e) => { if (G.play) { G.play.speed = +e.target.value; updatePlayLabel(); } });
  $('#pbView').addEventListener('change', (e) => { const P = G.play; if (!P) return; P.view = e.target.value; P.mv = null; if (P.view === 'inst') { P.idx = -1; } else showFrame(P.idx < 0 ? 0 : P.idx); dirty = true; });
  $('#pbExit').addEventListener('click', closeReview);
  $('#pbImg').addEventListener('click', () => openExport('image')); $('#pbVid').addEventListener('click', () => openExport('video'));
  $('#btnImg').addEventListener('click', () => openExport('image'));
  buildExportUI();
  $('#csvClose').addEventListener('click', () => { $('#csvDialog').hidden = true; });
  $('#csvCopy').addEventListener('click', async () => { const ta = $('#csvText'); ta.select(); try { await navigator.clipboard.writeText(ta.value); toast('Copied.'); } catch (e) { toast('Select the text and copy it with your device\'s copy command.'); } });
  $('#runList').addEventListener('click', onRunListClick);
  renderProbeUI(); refreshRuns(); checkResumable();
}
function renderProbeUI() {
  const n = S.probes.length;
  $('#probeTag').textContent = n ? `${n} placed` : '';
  $('#probeCount').textContent = n ? `${n} probe${n > 1 ? 's' : ''}` : 'no probes';
  const sel = $('#probeSel'), cur = sel.value;
  sel.innerHTML = n ? S.probes.map(p => `<option value="${p.id}">P${p.id} · x/c ${relX(p.x).toFixed(2)}, y/c ${relY(p.y).toFixed(2)}</option>`).join('') : '<option value="">no probes</option>';
  if (S.probes.some(p => String(p.id) === cur)) sel.value = cur;
}
function probeChartSel() { const v = +$('#probeSel').value; return isFinite(v) ? v : -1; }
function renderRecordUI() {
  const R = G && G.rec, P = G && G.play;
  $('#recSetup').hidden = !!R || !!P;
  $('#recProgress').hidden = !R;
  $('#recTag').textContent = R ? (R.phase === 'settle' ? 'settling' : 'recording') : P ? 'reviewing' : '';
  $('#stage').classList.toggle('recording', !!R);
  $('#recBadge').hidden = !R;
  if (R) updateRecProgress();
}
function updateRecProgress() {
  const R = G.rec; if (!R) return;
  const M = R.meta;
  let frac, txt;
  if (R.phase === 'settle') {
    const need = (G.firstRun ? 2 : 1) * FT(); frac = Math.min(1, G.since / need);
    txt = `Letting the flow settle before recording: ${Math.round(frac * 100)}%`; frac = 0;
  } else {
    frac = M.steps.done / M.steps.total;
    const el = (performance.now() - R.etaT0) / 1000, rate = el > 3 ? (M.steps.done - R.etaS0) / el : 0;
    const left = rate > 0 ? (M.steps.total - M.steps.done) / rate : NaN;
    txt = `${Math.round(frac * 100)}% · ${M.frames} frames · ${fmtBytes(M.bytes)} · ${isFinite(left) ? `about ${fmtDur(left)} left` : 'measuring speed…'}`;
  }
  $('#recBarFill').style.width = `${(frac * 100).toFixed(1)}%`;
  $('#recStatus').textContent = txt;
  $('#recBadgeTxt').textContent = R.phase === 'settle' ? 'Settling' : `REC ${Math.round(frac * 100)}%`;
}
async function refreshRuns() {
  let runs = [];
  try { runs = (await Store.runs()) || []; } catch (e) { $('#runList').innerHTML = '<li class="empty">This browser does not allow the page to store runs.</li>'; return; }
  const done = runs.filter(r => r.status === 'done').sort((a, b) => b.created - a.created);
  const ul = $('#runList');
  if (!done.length) ul.innerHTML = '<li class="empty">No runs recorded on this device yet.</li>';
  else ul.innerHTML = done.map(r => {
    const d = new Date(r.created), when = d.toLocaleDateString(undefined, { day: 'numeric', month: 'short' }) + ' ' + d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
    return `<li data-id="${r.id}">
      <div class="run-h"><b>${r.title}</b><span class="mono">Re ${r.Re.toLocaleString()}</span></div>
      <div class="run-m mono">${when} · ${r.plan.ft} flow-throughs · ${r.frames} frames · ${r.np} probes · ${fmtBytes(r.bytes)}</div>
      <div class="row"><button type="button" class="primary" data-act="play">Review</button><button type="button" data-act="p"${r.np ? '' : ' disabled'}>Probe CSV</button><button type="button" data-act="c">Force CSV</button><button type="button" data-act="f">Mean field CSV</button><button type="button" data-act="del">Delete</button></div>
    </li>`;
  }).join('');
  try { if (navigator.storage && navigator.storage.estimate) { const e = await navigator.storage.estimate(); $('#storeUse').textContent = `${fmtBytes(e.usage || 0)} used`; } } catch (e) { /* optional */ }
}
function onRunListClick(e) {
  const b = e.target.closest('button'); if (!b) return;
  const id = b.closest('li').dataset.id, act = b.dataset.act;
  if (act === 'play') openReview(id);
  else if (act === 'p' || act === 'c' || act === 'f') exportRunCSV(id, act);
  else if (act === 'del') {
    if (b.dataset.armed) { if (G.play && G.play.M.id === id) closeReview(); Store.deleteRun(id).then(refreshRuns); }
    else { b.dataset.armed = '1'; b.textContent = 'Tap again to delete'; setTimeout(() => { if (b.isConnected) { delete b.dataset.armed; b.textContent = 'Delete'; } }, 4000); }
  }
}
async function exportRunCSV(id, kind) {
  const M = await Store.getRun(id); if (!M) return;
  const name = `${M.title.replace(/[^A-Za-z0-9]+/g, '_')}_Re${M.Re}_${new Date(M.created).toISOString().slice(0, 16).replace(/[:T]/g, '-')}`;
  const sampleDt = M.steps.probeEvery * M.dtStep;
  const base = { Re_simulated: M.Re, cells_per_chord: M.grid.F.Nc, solver: M.mode === 'gpu' ? 'GPU' : 'CPU', recorded: new Date(M.created).toISOString(),
    run_length_flow_throughs: (M.steps.total / M.steps.ft).toFixed(2), sample_rate_Hz: (1 / sampleDt).toFixed(3), sample_interval_steps: M.steps.probeEvery, time_origin: 'first sample of the run' };
  if (kind === 'p') {
    const rows = await loadSeries(M, 'p', 1 + 3 * M.np, M.probeChunks);
    deliverCSV(`probes_${name}.csv`, probeCSV(rows, M.probes, M.dtStep, M.chord, M.speed, metaFor(M.settings, Object.assign({}, base, { samples: rows.length,
      coordinates: 'x/c and y/c in tunnel axes from the leading edge', velocity: 'u/U and v/U with U from the pitot-static probe (time-averaged)',
      Cp: 'solver static pressure at the probe, referenced to the pitot-static probe', probe_model: 'ideal point sensor (bilinear sample of the solver grid)' }))));
  } else if (kind === 'c') {
    const rows = await loadSeries(M, 'c', 4, M.forceChunks);
    const cdMean = rows.reduce((a, r) => a + r[2], 0) / Math.max(1, rows.length);
    const saved = { H: S.H, walls: S.walls, preset: S.preset, t: S.t };
    Object.assign(S, { H: M.settings.H, walls: M.settings.walls, preset: M.settings.preset, t: M.settings.t });
    const c0 = corrections(0, cdMean, 0);
    const out = ['"time_ms","t_U_over_c","CL_raw","CD_raw","Cm_quarter_chord_raw","CL_corrected","CD_corrected","Cm_corrected"'];
    for (const r of rows) { const ts = r[0] * M.dtStep, c = corrections(r[1], r[2], r[3], cdMean); out.push([(ts * 1000).toFixed(4), (ts * M.speed / M.chord).toFixed(5), r[1].toFixed(5), r[2].toFixed(5), r[3].toFixed(5), c ? c.cl.toFixed(5) : '', c ? c.cd.toFixed(5) : '', c ? c.cm.toFixed(5) : ''].join(',')); }
    Object.assign(S, saved);
    const sh = peakSt(rows.map(r => r[1]), sampleDt, M.chord, M.speed);
    deliverCSV(`forces_${name}.csv`, csvMeta(metaFor(M.settings, Object.assign({}, base, { samples: rows.length,
      coefficients: 'per unit span, from momentum exchange on the model surface; Cm about the quarter chord, nose-up positive',
      wall_correction: c0 ? `Allen and Vincenti; sigma=${c0.sigma.toFixed(5)}, epsilon=${c0.e.toFixed(5)} (wake blockage from the run-mean CD=${cdMean.toFixed(5)})` : 'none (slip walls)',
      shedding_St_fc_over_U: sh ? sh.St.toFixed(4) : 'no clear peak' }))) + out.join('\n'));
  } else {
    const A = await Store.get(`${M.id}/m/0`); if (!A || !A.n) { toast('This run has no field statistics.'); return; }
    const g = M.grid, n = A.n, u0 = A.u0 || U0, ur = A.refU || u0, rr = A.refRho || 1, q = 2 / (3 * ur * ur);
    const pos = M.settings.pos, a = M.settings.aoa * Math.PI / 180, lx = pos.x - 0.25 * Math.cos(a), ly = pos.y + 0.25 * Math.sin(a);
    const out = ['"i","j","x_over_c","y_over_c","zone","U_over_Uinf","V_over_Uinf","u_rms_over_Uinf","v_rms_over_Uinf","uv_over_Uinf2","Cp_mean","Cp_rms","omega_c_over_Uinf_mean","omega_rms"'];
    const L = M.settings.L, mask = rasterize(polyForSettings(M.settings, g.Nc), g.nx, g.ny);
    for (let y = 0; y < g.ny; y++) for (let x = 0; x < g.nx; x++) {
      const i = y * g.nx + x, mu = A.su[i] / n, mv = A.sv[i] / n, X = (x + 0.5) / g.Nc;
      const zone = mask[i] ? 'model' : X < 0.5 ? 'contraction' : X > L * 0.92 ? 'diffuser' : 'test_section';
      if (mask[i]) { out.push([x, y, (X - lx).toFixed(4), ((y + 0.5) / g.Nc - ly).toFixed(4), zone, '', '', '', '', '', '', '', '', ''].join(',')); continue; }
      const vu = Math.max(0, A.suu[i] / n - mu * mu), vv = Math.max(0, A.svv[i] / n - mv * mv), uv = A.suv[i] / n - mu * mv;
      const mp = A.sp[i] / n, vp = Math.max(0, A.spp[i] / n - mp * mp), mw = A.sw[i] / n, vw = Math.max(0, A.sww[i] / n - mw * mw);
      out.push([x, y, (X - lx).toFixed(4), ((y + 0.5) / g.Nc - ly).toFixed(4), zone, ((u0 + mu) / ur).toFixed(5), (mv / ur).toFixed(5), (Math.sqrt(vu) / ur).toFixed(5), (Math.sqrt(vv) / ur).toFixed(5), (uv / (ur * ur)).toFixed(6),
        ((mp - (rr - 1)) * q).toFixed(5), (Math.sqrt(vp) * q).toFixed(5), mw.toFixed(4), Math.sqrt(vw).toFixed(4)].join(','));
    }
    deliverCSV(`field_${name}.csv`, csvMeta(metaFor(M.settings, Object.assign({}, base, { samples_averaged: n, grid: `${g.nx}x${g.ny} cells, ${g.Nc} per chord (block-averaged from the solver grid)`,
      coordinates: 'cell centres, x/c and y/c in tunnel axes from the leading edge; i, j are grid indices', layout: `full ${g.nx} x ${g.ny} grid, row-major in j then i; cells inside the model are blank`,
      zones: 'contraction and diffuser are the inlet and outlet buffer layers, not measurement regions', velocity: 'normalised by the run-mean pitot-static U',
      Cp: 'solver static pressure, referenced to the run-mean pitot-static probe' }))) + out.join('\n'));
  }
}
async function checkResumable() {
  let runs = []; try { runs = (await Store.runs()) || []; } catch (e) { return; }
  const open = runs.filter(r => r.status === 'recording').sort((a, b) => b.created - a.created);
  const box = $('#recResume');
  if (!open.length) { box.hidden = true; return; }
  const r = open[0], pct = Math.round(100 * r.steps.done / Math.max(1, r.steps.total));
  box.innerHTML = `<p><b>Unfinished recording</b><br>${r.title}, Re ${r.Re.toLocaleString()} · ${pct}% done</p>
    <div class="row"><button type="button" class="primary" id="resumeBtn">Resume recording</button><button type="button" id="resumeKeep">Keep what was recorded</button><button type="button" id="resumeDrop">Discard</button></div>`;
  box.hidden = false;
  $('#resumeBtn').addEventListener('click', () => { box.hidden = true; resumeRecording(r.id); });
  $('#resumeKeep').addEventListener('click', async () => {
    box.hidden = true; const M = await Store.getRun(r.id); if (!M) return;
    M.status = 'done'; M.steps.total = M.steps.done; M.plan.ft = +(M.steps.done / M.steps.ft).toFixed(2);
    await Store.del(`${r.id}/k/0`); await Store.putRun(M); refreshRuns();
  });
  $('#resumeDrop').addEventListener('click', async () => { box.hidden = true; await Store.deleteRun(r.id); refreshRuns(); });
  // runs left behind before any checkpoint are useless
  for (const x of open.slice(1)) Store.deleteRun(x.id);
}

// ---------- probe signal chart ----------
const chP = $('#chProbe');
function drawProbeChart() {
  if (!chP) return;
  const ctx = chP.getContext('2d'), W = chP.width, H = chP.height, T = tokens();
  const pad = { l: 40 * dpr, r: 8 * dpr, t: 8 * dpr, b: 30 * dpr };
  const P = G.play, rows = P ? P.probes : (G.pbuf ? G.pbuf.rows : []);
  const sel = probeChartSel(), idx = S.probes.findIndex(p => p.id === sel), qv = $('#probeVar').value, psd = $('#probeMode').value === 'psd';
  ctx.clearRect(0, 0, W, H);
  if (idx < 0 || rows.length < 4) {
    axes(ctx, W, H, pad, [0, 1], [0, 1], T, psd ? 'St = f c/U' : 'ms', [], []);
    ctx.fillStyle = T.muted; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(S.probes.length ? 'collecting probe data…' : 'add probes under Probes to see their signals', W / 2, H / 2);
    return;
  }
  const get = (r) => { const u = r[1 + 3 * idx], v = r[2 + 3 * idx], c = r[3 + 3 * idx]; return qv === 'u' ? u : qv === 'v' ? v : qv === 'cp' ? c : Math.hypot(u, v); };
  const dtStep = P ? P.M.dtStep : flowTime(1), chord = P ? P.M.chord : S.chord, speed = P ? P.M.speed : S.speed, t0 = rows[0][0];
  if (psd) { // Welch power spectral density against Strouhal number, absolute units per unit St
    const sig = rows.map(get), dt = (rows[1][0] - rows[0][0]) * dtStep, W8 = welch(sig, dt), toSt = chord / speed;
    const xs = [], ys = []; let ymax = -1e9;
    for (let k = 1; k < W8.P.length; k++) { const st = k * W8.df * toSt, y = 10 * Math.log10(W8.P[k] / toSt + 1e-30); xs.push(st); ys.push(y); ymax = Math.max(ymax, y); }
    const pk = peakSt(sig, dt, chord, speed, qv === 'cp' ? 2e-3 : 1e-3), stNyq = xs[xs.length - 1];
    const xmax = Math.min(stNyq, pk ? Math.max(4, Math.ceil(pk.St * 3)) : 6), top = Math.ceil(ymax / 10) * 10 + 5, bot = top - 75;
    const { X, Y } = axes(ctx, W, H, pad, [0, xmax], [bot, top], T, 'St = f c/U∞', niceTicks(0, xmax, 5), niceTicks(bot, top, 4));
    ctx.strokeStyle = T.acc; ctx.lineWidth = 1.4 * dpr; ctx.beginPath(); let started = false;
    for (let k = 0; k < xs.length; k++) { if (xs[k] > xmax) break; const y = Y(Math.max(bot, ys[k])); if (started) ctx.lineTo(X(xs[k]), y); else { ctx.moveTo(X(xs[k]), y); started = true; } }
    ctx.stroke();
    ctx.fillStyle = T.muted; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    const unit = qv === 'cp' ? 'Cp²' : '(u/U∞)²';
    ctx.fillText(`PSD, dB re 1 ${unit} per unit St · Welch, ${W8.segs} Hann segment${W8.segs > 1 ? 's' : ''}, ΔSt ${(W8.res * toSt).toFixed(2)}`, pad.l + 4 * dpr, pad.t);
    ctx.fillStyle = T.ink;
    ctx.fillText(pk ? `peak St ${pk.St.toFixed(2)} ± ${pk.dSt.toFixed(2)} · ${pk.f.toFixed(1)} Hz · ${pk.snr.toFixed(0)} dB above the median` : 'no dominant frequency', pad.l + 4 * dpr, pad.t + 14 * dpr);
    return;
  }
  let lo = 1e9, hi = -1e9; for (const r of rows) { const y = get(r); if (isFinite(y)) { lo = Math.min(lo, y); hi = Math.max(hi, y); } }
  if (!(hi > lo)) { lo -= 0.1; hi += 0.1; } const padY = (hi - lo) * 0.08; lo -= padY; hi += padY;
  const xr = [0, Math.max(1e-6, (rows[rows.length - 1][0] - t0) * dtStep * 1000)];
  const { X, Y } = axes(ctx, W, H, pad, xr, [lo, hi], T, 'ms', niceTicks(xr[0], xr[1], 5), niceTicks(lo, hi, 4));
  ctx.strokeStyle = T.acc; ctx.lineWidth = 1.4 * dpr; ctx.beginPath(); let pen = false;
  const stride = Math.max(1, Math.floor(rows.length / (W / dpr)));
  for (let k = 0; k < rows.length; k += stride) { const y = get(rows[k]); if (!isFinite(y)) { pen = false; continue; } const x = X((rows[k][0] - t0) * dtStep * 1000); if (pen) ctx.lineTo(x, Y(y)); else { ctx.moveTo(x, Y(y)); pen = true; } }
  ctx.stroke();
  if (P && P.view === 'inst' && P.idx >= 0) { // playhead
    const tf = P.idx * P.frameDt * 1000; ctx.strokeStyle = T.ink; ctx.globalAlpha = 0.6; ctx.lineWidth = dpr; ctx.beginPath(); ctx.moveTo(X(tf), pad.t); ctx.lineTo(X(tf), H - pad.b); ctx.stroke(); ctx.globalAlpha = 1;
  }
}
