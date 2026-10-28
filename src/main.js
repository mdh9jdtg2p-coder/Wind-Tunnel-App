(() => {
'use strict';
const $ = (s) => document.querySelector(s);
const U0 = 0.09;
const NCR = 50; // readback grid resolution (cells per chord) used for overlays, probes and the pressure solve
const RES_GPU = { coarse: 1, std: 2, fine: 3, ultra: 4, extreme: 6 }; // multiples of NCR: 50, 100, 150, 200, 300 cells per chord
const RES_CPU = { coarse: 32, std: 40, fine: 52, ultra: 68, extreme: 68 };
const RES_ORDER = ['coarse', 'std', 'fine', 'ultra', 'extreme'];
const RES_NAME = { coarse: 'Coarse', std: 'Standard', fine: 'Fine', ultra: 'Ultra', extreme: 'Extreme' };
const MAXCELLS_CPU = 60000;
const PRESETS = { '0012': [0, 4, 12], '2412': [2, 4, 12], '4412': [4, 4, 12], '6409': [6, 4, 9], '0021': [0, 4, 21], 'plate': [0, 4, 2] };
const FIELD_IDX = { speed: 0, vort: 1, press: 2, smoke: 3, schl: 4, piv: 5 };

const S = {
  preset: '2412', m: 2, p: 4, t: 12, aoa: 6, flap: 'none', flapDef: 0, flapC: 0.25, flapPreset: '',
  speed: 1.6, chord: 0.06, turb: 0.15,
  L: 6, H: 4, walls: 'solid', res: 'std',
  field: 'speed', cmap: 'auto', stream: true, contours: false, vectors: false, contrast: 1, pivView: 'proc', smokeSrc: 'rake',
  pos: { x: 2.25, y: 2 }, running: true, probe: null,
};
let MODE = GPUTunnel.probe() ? 'gpu' : 'cpu';
let G = null, gpu = null, engine = null;

// ---------- colour maps ----------
function lutFrom(fn) { const L = new Uint32Array(256), C = []; for (let k = 0; k < 256; k++) { const [r, g, b] = fn(k / 255).map(v => Math.max(0, Math.min(255, Math.round(v * 255)))); L[k] = (255 << 24) | (b << 16) | (g << 8) | r; C.push([r, g, b]); } return { L, C }; }
function polyc(c, t) { return c.reduceRight((a, v) => a * t + v, 0); }
const MAPS = {
  viridis: lutFrom(t => {
    const c0 = [0.2777, 0.0054, 0.3341], c1 = [0.1051, 1.4046, 1.3846], c2 = [-0.3309, 0.2148, 0.0951], c3 = [-4.6342, -5.7991, -19.3324], c4 = [6.2283, 14.1799, 56.6906], c5 = [4.7764, -13.7451, -65.3530], c6 = [-5.4355, 4.6459, 26.3124];
    return [0, 1, 2].map(i => polyc([c0[i], c1[i], c2[i], c3[i], c4[i], c5[i], c6[i]], t));
  }),
  turbo: lutFrom(t => [
    polyc([0.13572138, 4.6153926, -42.66032258, 132.13108234, -152.94239396, 59.28637943], t),
    polyc([0.09140261, 2.19418839, 4.84296658, -14.18503333, 4.27729857, 2.82956604], t),
    polyc([0.1066733, 12.64194608, -60.58204836, 110.36276771, -89.90310912, 27.34824973], t)]),
  coolwarm: lutFrom(t => {
    const K = [[0, 59, 76, 192], [0.25, 124, 159, 249], [0.5, 221, 221, 221], [0.75, 238, 133, 105], [1, 180, 4, 38]];
    let k = 0; while (k < K.length - 2 && t > K[k + 1][0]) k++;
    const a = K[k], b = K[k + 1], u = (t - a[0]) / (b[0] - a[0]);
    return [1, 2, 3].map(i => (a[i] + (b[i] - a[i]) * u) / 255);
  }),
  grey: lutFrom(t => [t, t, t]),
};
const FIELDS = {
  speed: { hud: '|u| / U∞', name: 'Velocity magnitude', desc: 'normalised by free stream', map: 'viridis' },
  vort: { hud: 'ω c / U∞', name: 'Vorticity', desc: 'clockwise blue, anticlockwise red', map: 'coolwarm' },
  press: { hud: 'Cp', name: 'Pressure coefficient', desc: 'suction blue, stagnation red', map: 'coolwarm' },
  smoke: { hud: 'Smoke rake', name: 'Smoke visualisation', desc: 'filaments released from an upstream rake', map: null },
  schl: { hud: 'Schlieren · heated model', name: 'Schlieren', desc: 'horizontal knife edge (∂ρ/∂y), model heated to show its thermal layer and wake', map: 'grey' },
  piv: { hud: 'PIV · |u| / U∞', name: 'PIV vector field', desc: 'tracer image, validated vectors coloured by speed', map: 'turbo' },
};
function fieldRange() {
  const k = S.contrast, f = S.field;
  if (f === 'vort') return { lo: -30 / k, mid: 0, hi: 30 / k, div: true, step: 3 };
  if (f === 'press') return { lo: -2 / k, mid: 0, hi: 1 / k, div: true, step: 0.2 };
  return { lo: 0, mid: 0.8 / k, hi: 1.6 / k, div: false, step: 0.1 };
}

// ---------- geometry ----------
const flapAllowed = () => S.preset !== 'cyl' && S.t >= 6;
const flapOn = () => S.flap !== 'none' && flapAllowed();
function shapeChord() {
  if (S.preset === 'cyl') return cylinderShape(0.5, 200);
  if (flapOn()) return flapShape(S.m / 100, S.p / 10, S.t / 100, S.flap, S.flapC, S.flapDef, 180);
  return nacaShape(S.m / 100, S.p / 10, S.t / 100, 180);
}
const FLAP_NAMES = { plain: 'plain flap', split: 'split flap', slotted: 'slotted flap', fowler: 'Fowler flap' };
function flapLabel(st) { st = st || S; return st.flap && st.flap !== 'none' && st.preset !== 'cyl' && st.t >= 6 ? `${FLAP_NAMES[st.flap]} ${Math.round(st.flapC * 100)}% c at ${(+st.flapDef).toFixed(0)}°` : ''; }
// lattice coordinate x means the centre of cell x: physical position (x + 0.5) / Nc chords
function polyForSettings(st, Nc) { // model outline for stored run settings
  const save = { preset: S.preset, m: S.m, p: S.p, t: S.t, aoa: S.aoa, pos: S.pos, flap: S.flap, flapDef: S.flapDef, flapC: S.flapC };
  Object.assign(S, { preset: st.preset, m: st.m, p: st.p, t: st.t, aoa: st.aoa, pos: st.pos, flap: st.flap || 'none', flapDef: st.flapDef || 0, flapC: st.flapC || 0.25 });
  const P = polyFor(Nc); Object.assign(S, save); return P;
}
function polyFor(Nc) {
  const a = -S.aoa * Math.PI / 180, c = Math.cos(a), s = Math.sin(a);
  const cx = S.pos.x * Nc - 0.5, cy = S.pos.y * Nc - 0.5;
  return shapeChord().map(p => { const x = p.x - 0.25, y = p.y; return { x: cx + Nc * (c * x - s * y), y: cy + Nc * (s * x + c * y), xc: p.xc, s: p.s, e: p.e | 0, hid: !!p.hid, te: !!p.te, cove: !!p.cove }; });
}
function rasterize(poly, nx, ny) { // union of the elements' interiors
  const mask = new Uint8Array(nx * ny), ring = elementRing(poly);
  for (const [r0, r1] of ring.ranges) {
    let y0 = 1e9, y1 = -1e9; for (let k = r0; k < r1; k++) { y0 = Math.min(y0, poly[k].y); y1 = Math.max(y1, poly[k].y); }
    for (let y = Math.max(1, Math.floor(y0)); y <= Math.min(ny - 2, Math.ceil(y1)); y++) {
      const xs = [];
      for (let k = r0; k < r1; k++) { const a = poly[k], b = poly[ring.next[k]]; if ((a.y > y) !== (b.y > y)) xs.push(a.x + (y - a.y) / (b.y - a.y) * (b.x - a.x)); }
      xs.sort((a, b) => a - b);
      for (let k = 0; k + 1 < xs.length; k += 2) for (let x = Math.max(1, Math.ceil(xs[k])); x <= Math.min(nx - 2, Math.floor(xs[k + 1])); x++) mask[y * nx + x] = 1;
    }
  }
  return mask;
}
function profileName() { return S.preset === 'cyl' ? 'Cylinder' : S.preset === 'plate' ? 'Flat plate' : `NACA ${S.m}${S.p}${String(S.t).padStart(2, '0')}`; }
function modelName() { const f = flapLabel(); return f ? `${profileName()} with ${f}` : profileName(); }

// ---------- CPU engine (worker with main-thread fallback) ----------
function makeEngine(onmsg) {
  try {
    const src = $('#coreSrc').textContent + '\n' + $('#workerSrc').textContent;
    const w = new Worker(URL.createObjectURL(new Blob([src], { type: 'text/javascript' })));
    w.onmessage = (e) => onmsg(e.data);
    return { post: (m) => w.postMessage(m), kill: () => w.terminate() };
  } catch (err) {
    let sim = null, running = true, want = false, acc = [0, 0, 0, 0, 0, 0, 0], alive = true, probe = null, samples = [];
    const handle = (m) => {
      if (m.type === 'init') { sim = new LBM(m.nx, m.ny); sim.walls = m.walls; sim.turb = m.turb; sim.setFlow(m.u0, m.Re, m.Nc); sim.refX = m.refX; sim.refY = m.refY; if (m.poly) sim.setBody(m.poly); acc = [0, 0, 0, 0, 0, 0, 0]; }
      else if (!sim) return;
      else if (m.type === 'flow') { sim.setFlow(m.u0, m.Re, m.Nc); sim.turb = m.turb; }
      else if (m.type === 'walls') sim.walls = m.walls;
      else if (m.type === 'body') { sim.refX = m.refX; sim.refY = m.refY; sim.setBody(m.poly); }
      else if (m.type === 'reset') { sim.initUniform(0); acc = [0, 0, 0, 0, 0, 0, 0]; }
      else if (m.type === 'run') running = m.on;
      else if (m.type === 'get') want = true;
      else if (m.type === 'getState') onmsg({ type: 'state', f: sim.f.slice(), t: sim.t });
      else if (m.type === 'setState') { sim.f.set(m.f); sim.t = m.t; samples = []; }
      else if (m.type === 'probes') { probe = m.pts.length ? { pts: m.pts, every: m.every } : null; samples = []; }
    };
    const tick = () => {
      if (!alive) return;
      if (sim && running) { const t0 = performance.now(); while (performance.now() - t0 < 8) { sim.step(); acc[0] += sim.Fx; acc[1] += sim.Fy; acc[2] += sim.Mz; acc[3]++; acc[4] += sim.FxF; acc[5] += sim.FyF; acc[6] += sim.MzF; if (probe && sim.t % probe.every === 0) samples.push(sampleProbesLBM(sim, probe.pts)); } if (!sim.isHealthy()) { sim.initUniform(0); onmsg({ type: 'unstable' }); } }
      if (want && sim) { want = false; const n = acc[3] || 1; onmsg({ type: 'snap', t: sim.t, n: acc[3], Fx: acc[0] / n, Fy: acc[1] / n, Mz: acc[2] / n, FxF: acc[4] / n, FyF: acc[5] / n, MzF: acc[6] / n, rho: sim.rho.slice(), ux: sim.ux.slice(), uy: sim.uy.slice(), nx: sim.nx, ny: sim.ny, samples: packSamples(samples), ns: samples.length }); samples = []; acc = [0, 0, 0, 0, 0, 0, 0]; }
      setTimeout(tick, 4);
    };
    tick();
    return { post: handle, kill: () => { alive = false; } };
  }
}

// ---------- derived settings ----------
function nuMin() { return MODE === 'gpu' ? (0.5015 - 0.5) / 3 : (0.5035 - 0.5) / 3; }
const NU_AIR = 1.5e-5;
function flowParams() {
  const reMax = Math.floor(G.F.Nc * U0 / nuMin() * (G.reScale || 1) / 100) * 100;
  const ReReal = S.speed * S.chord / NU_AIR;
  const Re = Math.max(50, Math.min(reMax, Math.round(ReReal)));
  return { Re, ReReal, reMax, capped: ReReal > reMax };
}
const speedFromPos = (v) => { const s = 0.2 * Math.pow(300, v / 1000); return s < 2 ? Math.round(s * 100) / 100 : s < 10 ? Math.round(s * 10) / 10 : Math.round(s); };
const posFromSpeed = (s) => Math.round(1000 * Math.log(s / 0.2) / Math.log(300));
function clampPos() {
  S.pos.x = Math.min(S.L - 1.5, Math.max(0.95, S.pos.x));
  S.pos.y = Math.min(S.H - 0.35, Math.max(0.35, S.pos.y));
}
const FT = () => G.F.nx / U0; // steps for one flow-through of the test section

// ---------- grid cost: memory and run time ----------
function gridPlan(res, L = S.L, H = S.H) {
  const k = RES_GPU[res], nxr = Math.round(L * NCR), nyr = Math.round(H * NCR), nx = nxr * k, ny = nyr * k, N = nx * ny;
  const ds = N * 4 <= 1.5e6 ? 2 : N * 2.25 <= 1.5e6 ? 1.5 : 1, Nd = Math.round(nx * ds) * Math.round(ny * ds);
  const cw = Math.min(nx, 900), chh = Math.round(cw * ny / nx);
  // graphics memory per solver cell: populations, post-collision populations, two macroscopic fields, force and two
  // scalar fields (RGBA32F, 16 bytes each, 11 textures), velocity (RGBA16F), two link-distance textures and two masks.
  // Smoke and heat: five RGBA16F textures per dye cell. Recording adds six RGBA32F statistics accumulators.
  const gpuB = N * 218 + Nd * 40 + nxr * nyr * 16, recB = N * 96 + cw * chh * 4;
  // processor-side memory: link distances and masks, plus the checkpoint copy of the populations while recording
  const host = N * 34, hostRec = N * 34 + N * 72;
  return { res, k, Nc: NCR * k, nx, ny, N, gpu: gpuB, gpuRec: gpuB + recB, need: gpuB + host, needRec: gpuB + recB + hostRec, ft: nx / U0,
    reMax: Math.floor(NCR * k * U0 / ((0.5015 - 0.5) / 3) / 100) * 100 };
}
function deviceBudget() {
  const ua = navigator.userAgent || '', mobile = /Android|iPhone|iPad|iPod|Mobile/i.test(ua) || (navigator.maxTouchPoints > 1 && /Macintosh/.test(ua));
  const rep = navigator.deviceMemory || 0, gb = rep || (mobile ? 4 : 8);
  // Browsers do not say how much graphics memory is free. Phones and tablets share memory with the system and close
  // heavy pages early, so they get a smaller working limit than desktops.
  const lim = (mobile ? Math.min(1.2, 0.3 * gb) : Math.min(4, 0.45 * gb)) * 1e9;
  let maxDim = 4096;
  if (gpu && !gpu.gl.isContextLost()) { const gl = gpu.gl, vp = gl.getParameter(gl.MAX_VIEWPORT_DIMS); maxDim = Math.min(gl.getParameter(gl.MAX_TEXTURE_SIZE), vp[0], vp[1]); }
  return { mobile, reported: rep, gb, lim, maxDim };
}
function planCheck(p) {
  const B = deviceBudget(), dimOK = p.nx <= B.maxDim && p.ny <= B.maxDim;
  const level = !dimOK ? 'impossible' : p.need > B.lim ? 'over' : p.needRec > B.lim ? 'tight' : p.needRec > 0.7 * B.lim ? 'close' : 'ok';
  return { B, level };
}
// measured solver throughput on this device, in cell updates per second (remembered between visits)
let CPS = 0;
try { CPS = +localStorage.getItem('wt.cps') || 0; } catch (e) { /* storage blocked */ }
function noteThroughput(cells, steps, ms) {
  if (!(ms > 3) || steps < 1) return;
  const c = cells * steps / (ms / 1000);
  CPS = CPS ? 0.93 * CPS + 0.07 * c : c;
}
setInterval(() => { try { if (CPS > 0) localStorage.setItem('wt.cps', String(Math.round(CPS))); } catch (e) { /* storage blocked */ } }, 10000);
// recording pace in cell updates per second, learned from finished recordings on this device (includes smoke, samples and frames)
let REC_CPS = 0;
try { REC_CPS = +localStorage.getItem('wt.recCps') || 0; } catch (e) { /* storage blocked */ }
function learnRecPace(c) { if (!(c > 0)) return; REC_CPS = REC_CPS ? 0.5 * REC_CPS + 0.5 * c : c; try { localStorage.setItem('wt.recCps', String(Math.round(REC_CPS))); } catch (e) { /* storage blocked */ } }
// one flow-through while recording: the learned pace, or the bare solver plus about 15% for smoke, samples and frames
const flowSecs = (p) => REC_CPS > 0 ? p.N * p.ft / REC_CPS : CPS > 0 ? p.N * p.ft / CPS * 1.15 : NaN;
const fmtGB = (b) => b >= 9.5e8 ? `${(b / 1e9).toFixed(1)} GB` : `${Math.max(10, Math.round(b / 1e7) * 10)} MB`;
const fmtCells = (n) => n >= 1e6 ? `${(n / 1e6).toFixed(2)} million` : `${Math.round(n / 1e3)} thousand`;

function initGrid(opts) {
  clampPos();
  const prevCfg = G && G.cfg, lite = !!(opts && opts.lite) && MODE === 'gpu'; // lite: review only, the solver is not run
  let F, R;
  if (MODE === 'gpu') {
    const k = RES_GPU[S.res];
    const nxr = Math.round(S.L * NCR), nyr = Math.round(S.H * NCR);
    R = { Nc: NCR, nx: nxr, ny: nyr };
    F = { Nc: NCR * k, nx: nxr * k, ny: nyr * k, k };
  } else {
    let Nc = RES_CPU[S.res];
    if (S.L * S.H * Nc * Nc > MAXCELLS_CPU) Nc = Math.floor(Math.sqrt(MAXCELLS_CPU / (S.L * S.H)));
    const nx = Math.round(S.L * Nc), ny = Math.round(S.H * Nc);
    R = { Nc, nx, ny }; F = { Nc, nx, ny, k: 1 };
  }
  const N = R.nx * R.ny;
  G = { ...R, N, F, rho: new Float32Array(N).fill(1), ux: new Float32Array(N), uy: new Float32Array(N),
    dye: new Float32Array(N), dye2: new Float32Array(N), sc: new Float32Array(N), t: 0, hist: [], since: 0, firstRun: true,
    img: new ImageData(R.nx, R.ny), off: document.createElement('canvas'), poly: null, polyF: null, mask: null, rhoRef: 1, uRef: U0, refInit: false, pmg: null, pw: null, cpP: null, pmgValid: false,
    streaks: [], emitAcc: 0, shadow: null, cp: null, cpT: 0, macBox: null, piv: null, frame: 0, spf: 8, unhealthy: 0 };
  G.off.width = R.nx; G.off.height = R.ny;
  G.poly = polyFor(G.Nc); G.polyF = polyFor(F.Nc); G.mask = rasterize(G.poly, G.nx, G.ny);
  initStreaks(); initPIV();
  const fp = flowParams();
  G.cfg = { res: S.res, L: S.L, H: S.H }; G.lite = lite;
  if (MODE === 'gpu') {
    const kg = lite ? 1 : F.k; let fail = null;
    try { gpu.configure(R.nx * kg, R.ny * kg, kg); fail = gpuAllocCheck(); } catch (err) { fail = String((err && err.message) || err); }
    if (fail) {
      console.warn('grid allocation failed:', fail);
      if (gpu.gl.isContextLost()) { lostDown = true; return; } // the context-restored handler rebuilds on a coarser grid
      const lower = lowerLevel(prevCfg);
      if (!lower) throw new Error(fail);
      const from = RES_NAME[S.res]; S.res = lower; ctlById('res').sync(); syncTags();
      toast(`There wasn't enough graphics memory for the ${from} grid on this device, so the tunnel is back on ${RES_NAME[lower]}.`);
      return initGrid(opts);
    }
    gpu.walls = S.walls; gpu.rampSteps = Math.round(0.35 * FT());
    gpu.setFlow(U0, fp.Re, NCR * kg, S.turb / 100);
    gpu.setBody(lite ? polyFor(NCR) : G.polyF, S.pos.x * NCR * kg - 0.5, S.pos.y * NCR * kg - 0.5);
    G.builtAt = performance.now();
  } else {
    if (engine) engine.kill();
    engine = makeEngine(onEngine);
    engine.post({ type: 'init', nx: F.nx, ny: F.ny, walls: S.walls, turb: S.turb / 100, u0: U0, Re: fp.Re, Nc: F.Nc, poly: G.polyF.map(p => ({ x: p.x, y: p.y, e: p.e })), refX: S.pos.x * F.Nc - 0.5, refY: S.pos.y * F.Nc - 0.5 });
    engine.post({ type: 'run', on: S.running });
    waiting = false;
  }
  $('#stage').dataset.walls = S.walls;
  G.pbuf = newProbeBuffer();
  if (S.probes) sendProbes();
  sizeCanvas();
  updateResUI();
}
function lowerLevel(prevCfg) { // the grid to fall back to: the previous one if it was coarser, else one step down
  const i = RES_ORDER.indexOf(S.res);
  if (prevCfg && RES_ORDER.indexOf(prevCfg.res) < i) return prevCfg.res;
  return i > 0 ? RES_ORDER[i - 1] : null;
}
// After allocating a grid, make sure it really exists: an out-of-memory error, a lost context or a solver step that
// reads back garbage all mean the graphics processor could not hold it.
function gpuAllocCheck() {
  const gl = gpu.gl; let e, oom = false;
  const drain = () => { while ((e = gl.getError()) !== gl.NO_ERROR && !gl.isContextLost()) if (e === gl.OUT_OF_MEMORY) oom = true; };
  drain(); if (gl.isContextLost()) return 'context lost'; if (oom) return 'out of memory';
  gpu.step(1); const d = gpu.readDown(); gpu.reset();
  drain(); if (gl.isContextLost()) return 'context lost'; if (oom) return 'out of memory';
  for (let k = 0; k < 24; k++) { const r = d[4 * ((Math.random() * (d.length / 4)) | 0)]; if (!(r > 0.9 && r < 1.1)) return 'solver read-back failed'; }
  return null;
}
function sendBody() {
  if (G.lite) return; // reviewing a run: the model is fixed
  G.poly = polyFor(G.Nc); G.polyF = polyFor(G.F.Nc); G.mask = rasterize(G.poly, G.nx, G.ny); G.shadow = null; G.pmg = null; G.pmgValid = false;
  const rx = S.pos.x * G.F.Nc - 0.5, ry = S.pos.y * G.F.Nc - 0.5;
  if (MODE === 'gpu') gpu.setBody(G.polyF, rx, ry);
  else engine.post({ type: 'body', poly: G.polyF.map(p => ({ x: p.x, y: p.y, e: p.e })), refX: rx, refY: ry });
  dirty = true;
}
function sendFlow() {
  if (G.lite) return;
  const fp = flowParams();
  if (MODE === 'gpu') gpu.setFlow(U0, fp.Re, G.F.Nc, S.turb / 100);
  else engine.post({ type: 'flow', u0: U0, Re: fp.Re, Nc: G.F.Nc, turb: S.turb / 100 });
  updateState(null);
}

// ---------- shared post-processing ----------
function pushForces(Fx, Fy, Mz, n, FxF, FyF, MzF) { // F: the flap's share, when there is a flap
  if (n <= 0) return;
  G.since += n;
  const qn = 0.5 * U0 * U0 * G.F.Nc;
  if (G.t > 0.4 * FT()) {
    const h = { t: G.t, cl: Fy / qn, cd: Fx / qn, cm: -Mz / (qn * G.F.Nc) };
    if (flapOn() && FxF !== undefined) { h.clF = FyF / qn; h.cdF = FxF / qn; h.cmF = -MzF / (qn * G.F.Nc); }
    G.hist.push(h);
  }
  const maxT = 4 * FT();
  while (G.hist.length && G.hist[0].t < G.t - maxT) G.hist.shift();
}
// reference static and dynamic pressure, as the tunnel's upstream pitot-static reads them (time-averaged over two acoustic transits)
function updateRhoRef(nsteps) {
  let s = 0, u = 0, c = 0; const x = Math.min(G.nx - 1, Math.round(0.7 * G.Nc));
  const yb = Math.round(0.25 * G.Nc);
  for (let y = yb; y < G.ny - yb; y++) { const i = y * G.nx + x; if (!G.mask[i]) { s += G.rho[i]; u += G.ux[i]; c++; } }
  if (!c) return;
  const r = s / c, ur = u / c;
  if (ur < 0.3 * U0) { G.rhoRef = r; G.uRef = U0; G.refInit = false; return; }
  const tau = 2 * G.F.ny / 0.577, a = G.refInit ? 1 - Math.exp(-Math.max(1, nsteps || 1) / tau) : 1;
  G.rhoRef += a * (r - G.rhoRef); G.uRef += a * (ur - G.uRef); G.refInit = true;
}

// ---------- hydrodynamic pressure field (Poisson), blended with the solver pressure next to the model ----------
function poissonSetup() {
  G.pmg = new PoissonMG(G.nx, G.ny, G.mask); G.pmgCold = true;
  // blend weight for the solver pressure: 1 within 0.08c of the surface, 0 beyond 0.2c
  const { nx, ny, mask } = G, D = new Float32Array(nx * ny);
  for (let i = 0; i < D.length; i++) D[i] = mask[i] ? 0 : 1e9;
  for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) { const i = y * nx + x; let v = D[i]; if (x > 0) v = Math.min(v, D[i - 1] + 1); if (y > 0) { v = Math.min(v, D[i - nx] + 1); if (x > 0) v = Math.min(v, D[i - nx - 1] + 1.414); if (x < nx - 1) v = Math.min(v, D[i - nx + 1] + 1.414); } D[i] = v; }
  for (let y = ny - 1; y >= 0; y--) for (let x = nx - 1; x >= 0; x--) { const i = y * nx + x; let v = D[i]; if (x < nx - 1) v = Math.min(v, D[i + 1] + 1); if (y < ny - 1) { v = Math.min(v, D[i + nx] + 1); if (x < nx - 1) v = Math.min(v, D[i + nx + 1] + 1.414); if (x > 0) v = Math.min(v, D[i + nx - 1] + 1.414); } D[i] = v; }
  const a = 0.08 * G.Nc, b = 0.2 * G.Nc; G.pw = new Float32Array(nx * ny);
  for (let i = 0; i < D.length; i++) { const t = Math.min(1, Math.max(0, (D[i] - a) / (b - a))); G.pw[i] = 1 - t * t * (3 - 2 * t); }
  G.cpP = new Float32Array(nx * ny); G.pmgValid = false;
}
function updatePoisson(cycles) {
  if (!G.pmg) poissonSetup();
  const p = G.pmg.solve(G.ux, G.uy, G.pmgCold ? 8 : cycles); G.pmgCold = false;
  let s = 0, c = 0; const x = Math.min(G.nx - 1, Math.round(0.7 * G.Nc)), yb = Math.round(0.25 * G.Nc);
  for (let y = yb; y < G.ny - yb; y++) { const i = y * G.nx + x; if (!G.mask[i]) { s += p[i]; c++; } }
  const pref = c ? s / c : 0, q = 0.5 * G.uRef * G.uRef;
  for (let i = 0; i < p.length; i++) G.cpP[i] = G.mask[i] ? 0 : (p[i] - pref) / q;
  G.pmgValid = true;
  if (MODE === 'gpu') gpu.setPoisson(G.cpP, G.pw, G.nx, G.ny);
}
function cpAt(x, y) { // blended pressure coefficient at a readback-grid position
  const r = sampleField(G.rho, x, y, G.nx, G.ny), cpL = (r - G.rhoRef) * 2 / (3 * G.uRef * G.uRef);
  if (!G.pmgValid) return Math.min(cpLimit(), cpL);
  const w = sampleField(G.pw, x, y, G.nx, G.ny), cpH = sampleField(G.cpP, x, y, G.nx, G.ny);
  return Math.min(cpLimit(), w * cpL + (1 - w) * cpH);
}

// ---------- CPU engine messages ----------
let waiting = false, dirty = true;
function onEngine(m) {
  if (m.type === 'unstable') { if (G.rec) abortRecording('The flow diverged during the recording, so it stopped.'); unstable(); return; }
  if (m.type === 'state') { const w = cpuStateWaiter; cpuStateWaiter = null; if (w) w(m); return; }
  if (m.type !== 'snap' || m.nx !== G.F.nx || m.ny !== G.F.ny) { waiting = false; return; }
  waiting = false;
  G.rho = m.rho; G.ux = m.ux; G.uy = m.uy; G.t = m.t;
  if (m.n > 0) { advectDye(m.n); pushForces(m.Fx, m.Fy, m.Mz, m.n, m.FxF, m.FyF, m.MzF); advectPIV(m.n); }
  updateRhoRef(m.n);
  if (G.frame % 2 === 0) updateCp(m.n);
  if ((S.field === 'press' || S.probe) && G.frame % 2 === 0) updatePoisson(1);
  const rows = workerRows(m);
  if (m.n > 0) { const now = performance.now(); if (G.lastSnapAt) { const s = m.n / ((now - G.lastSnapAt) / 1000); G.sps = G.sps ? 0.9 * G.sps + 0.1 * s : s; } G.lastSnapAt = now; if (!G.rec) liveProbeSampleCPU(rows); }
  if (G.rec) recordSnapshotCPU(m, rows);
  dirty = true;
}

// ---------- smoke for the CPU path (dye + streaklines) ----------
function sampleField(F, x, y, nx, ny) {
  if (x < 0) x = 0; else if (x > nx - 1.001) x = nx - 1.001;
  if (y < 0) y = 0; else if (y > ny - 1.001) y = ny - 1.001;
  const x0 = x | 0, y0 = y | 0, fx = x - x0, fy = y - y0, i = y0 * nx + x0;
  return (F[i] * (1 - fx) + F[i + 1] * fx) * (1 - fy) + (F[i + nx] * (1 - fx) + F[i + nx + 1] * fx) * fy;
}
// where smoke is released, in readback-grid cells: a rake across the whole contraction exit, or a fine wire 0.3c ahead of the model
function smokeRelease() {
  if (S.smokeSrc === 'wire') {
    let x0 = 1e9; for (const p of G.poly) x0 = Math.min(x0, p.x);
    const xw = Math.max(2, x0 - 0.3 * G.Nc), yc = S.pos.y * G.Nc - 0.5;
    return { x0: xw, x1: xw + 0.75, y0: Math.max(1, yc - 1.1 * G.Nc), y1: Math.min(G.ny - 2, yc + 1.1 * G.Nc), sp: 0.035 * G.Nc, off: 0, w: 0.22 };
  }
  const sp = G.ny / 18, yt = S.pos.y * G.Nc - 0.5 - 0.02 * G.Nc; // aim one streak at the dividing streamline
  return { x0: 0, x1: 1.75, y0: 0, y1: G.ny, sp, off: ((yt % sp) + sp) % sp, w: 0.28 };
}
function initStreaks() {
  const n = Math.max(14, Math.round(G.ny / 6)), arr = [];
  for (let k = 0; k < n; k++) arr.push({ y: (k + 0.5) * (G.ny - 1) / n, x: [], yy: [] });
  G.streaks = arr; G.emitAcc = 0;
}
function advectStreaks(nsteps) {
  const { nx, ny, ux, uy, mask } = G, EMIT = 3.5;
  const sub = Math.max(1, Math.ceil(nsteps * U0 * 1.7));
  const dt = nsteps / sub;
  for (let s = 0; s < sub; s++) {
    G.emitAcc += dt;
    const emit = G.emitAcc >= EMIT; if (emit) G.emitAcc -= EMIT;
    for (const st of G.streaks) {
      const X = st.x, Y = st.yy;
      for (let k = 0; k < X.length; k++) {
        const x = X[k], y = Y[k];
        const u = sampleField(ux, x, y, nx, ny), v = sampleField(uy, x, y, nx, ny);
        const xm = x + 0.5 * dt * u, ym = y + 0.5 * dt * v;
        const um = sampleField(ux, xm, ym, nx, ny), vm = sampleField(uy, xm, ym, nx, ny);
        const xn = x + dt * um, yn = Math.min(ny - 1, Math.max(0, y + dt * vm));
        const xi = Math.round(xn), yi = Math.round(yn);
        if (xi >= 0 && xi < nx && mask[yi * nx + xi]) continue;
        X[k] = xn; Y[k] = yn;
      }
      while (X.length && X[0] > nx - 2) { X.shift(); Y.shift(); }
      if (emit) { X.push(1.5); Y.push(st.y); }
    }
  }
}
function advectDye(nsteps) {
  advectStreaks(nsteps);
  const { nx, ny, ux, uy, mask } = G;
  let D = G.dye, D2 = G.dye2;
  const sub = Math.max(1, Math.ceil(nsteps * U0 * 1.8 / 1.2));
  const dt = nsteps / sub;
  for (let s = 0; s < sub; s++) {
    for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
      const i = y * nx + x;
      if (mask[i]) { D2[i] = 0; continue; }
      const u = ux[i], v = uy[i];
      const xm = x - 0.5 * dt * u, ym = y - 0.5 * dt * v;
      const um = sampleField(ux, xm, ym, nx, ny), vm = sampleField(uy, xm, ym, nx, ny);
      D2[i] = sampleField(D, x - dt * um, y - dt * vm, nx, ny) * 0.9995;
    }
    const t = D; D = D2; D2 = t;
    const R = smokeRelease(), sp = Math.max(2, Math.round(R.sp));
    for (let y = Math.max(0, Math.round(R.y0 + R.off)); y < Math.min(ny - 1, R.y1); y += sp) for (let x = Math.max(1, Math.floor(R.x0)); x <= Math.max(2, Math.ceil(R.x1)); x++) D[y * nx + x] = 1;
  }
  G.dye = D; G.dye2 = D2;
}

// ---------- PIV tracers (both paths) ----------
function rand() { return Math.random(); }
function initPIV() {
  const n = Math.round(Math.min(6500, G.nx * G.ny * 0.2));
  const P = new Float32Array(n * 3);
  for (let k = 0; k < n; k++) { P[k * 3] = rand() * (G.nx - 1); P[k * 3 + 1] = rand() * (G.ny - 1); P[k * 3 + 2] = 0.35 + 0.65 * rand(); }
  G.piv = P;
}
function advectPIV(nsteps) {
  const P = G.piv, { nx, ny, ux, uy, mask } = G; if (!P) return;
  const dt = Math.min(nsteps, 40);
  for (let k = 0; k < P.length; k += 3) {
    let x = P[k], y = P[k + 1];
    const u = sampleField(ux, x, y, nx, ny), v = sampleField(uy, x, y, nx, ny);
    const um = sampleField(ux, x + 0.5 * dt * u, y + 0.5 * dt * v, nx, ny), vm = sampleField(uy, x + 0.5 * dt * u, y + 0.5 * dt * v, nx, ny);
    x += dt * um; y += dt * vm;
    let respawn = x > nx - 1 || y < 0 || y > ny - 1 || x < 0 || mask[Math.round(y) * nx + Math.round(x)] === 1;
    if (!respawn && rand() < 0.002 * dt / 10) respawn = true; // particles leave the light sheet out of plane
    if (respawn) { x = rand() < 0.5 ? rand() * 2 : rand() * (nx - 1); y = rand() * (ny - 1); P[k + 2] = 0.35 + 0.65 * rand(); }
    P[k] = x; P[k + 1] = y;
  }
}
// laser sheet enters from above: a cell is shadowed when the model blocks the ray back to the light source
function shadowGrid() {
  if (G.shadow && G.shadow.poly === G.poly) return G.shadow.m;
  const { nx, ny, mask } = G, m = new Uint8Array(nx * ny);
  const lx = S.pos.x * G.Nc - 0.5, ly = ny + 1.6 * ny; // light sheet optics well above the roof
  let bx0 = 1e9, bx1 = -1e9, by0 = 1e9, by1 = -1e9;
  for (const p of G.poly) { bx0 = Math.min(bx0, p.x); bx1 = Math.max(bx1, p.x); by0 = Math.min(by0, p.y); by1 = Math.max(by1, p.y); }
  const r0 = Math.max(0, Math.floor(by0)), r1 = Math.min(ny - 1, Math.ceil(by1));
  for (let y = 0; y <= r1; y++) for (let x = 0; x < nx; x++) {
    if (mask[y * nx + x]) continue;
    for (let yy = Math.max(r0, y + 1); yy <= r1; yy++) {
      const xs = Math.round(x + (lx - x) * (yy - y) / (ly - y));
      if (xs >= 0 && xs < nx && mask[yy * nx + xs]) { m[y * nx + x] = 1; break; }
    }
  }
  G.shadow = { poly: G.poly, m };
  return m;
}

// ---------- surface pressure (time-averaged, wall-extrapolated) ----------
function rhoFullAt(x, y) {
  // bilinear sample of density on the full-resolution grid, fluid nodes only
  const x0 = Math.floor(x), y0 = Math.floor(y); let s = 0, w = 0;
  for (let dy = 0; dy <= 1; dy++) for (let dx = 0; dx <= 1; dx++) {
    const xi = x0 + dx, yi = y0 + dy; let r;
    if (MODE === 'gpu') {
      const B = G.macBox; if (!B) return NaN;
      if (xi < B.x0 || yi < B.y0 || xi >= B.x0 + B.w || yi >= B.y0 + B.h) continue;
      if (gpu.solid[yi * G.F.nx + xi]) continue;
      r = B.buf[((yi - B.y0) * B.w + (xi - B.x0)) * 4];
    } else {
      if (xi < 0 || yi < 0 || xi >= G.nx || yi >= G.ny || G.mask[yi * G.nx + xi]) continue;
      r = G.rho[yi * G.nx + xi];
    }
    const wt = (1 - Math.abs(x - xi)) * (1 - Math.abs(y - yi)); s += wt * r; w += wt;
  }
  return w > 0.05 ? s / w : NaN;
}
function velFullAt(x, y) { // bilinear velocity on the full-resolution grid, fluid nodes only
  if (MODE !== 'gpu') return [sampleField(G.ux, x, y, G.nx, G.ny), sampleField(G.uy, x, y, G.nx, G.ny)];
  const B = G.macBox; if (!B) return [NaN, NaN];
  const x0 = Math.floor(x), y0 = Math.floor(y); let su = 0, sv = 0, w = 0;
  for (let dy = 0; dy <= 1; dy++) for (let dx = 0; dx <= 1; dx++) {
    const xi = x0 + dx, yi = y0 + dy;
    if (xi < B.x0 || yi < B.y0 || xi >= B.x0 + B.w || yi >= B.y0 + B.h || gpu.solid[yi * G.F.nx + xi]) continue;
    const o = ((yi - B.y0) * B.w + (xi - B.x0)) * 4, wt = (1 - Math.abs(x - xi)) * (1 - Math.abs(y - yi));
    su += wt * B.buf[o + 1]; sv += wt * B.buf[o + 2]; w += wt;
  }
  return w > 0.05 ? [su / w, sv / w] : [NaN, NaN];
}
function cpLimit() { // stagnation pressure can exceed 1 only by viscous effects of order 1/Re at the leading-edge radius
  const rLE = S.preset === 'cyl' ? 0.25 : 1.1019 * Math.pow(Math.max(S.t, 2) / 100, 2);
  return 1 + 8 / Math.max(1, flowParams().Re * rLE);
}
function updateCp(nsteps) {
  const P = G.polyF, n = P.length;
  if (MODE === 'gpu') {
    let x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9; for (const p of P) { x0 = Math.min(x0, p.x); x1 = Math.max(x1, p.x); y0 = Math.min(y0, p.y); y1 = Math.max(y1, p.y); }
    G.macBox = gpu.readMacBox(Math.floor(x0) - 4, Math.floor(y0) - 4, Math.ceil(x1 - x0) + 9, Math.ceil(y1 - y0) + 9);
  }
  if (!P.ring) P.ring = elementRing(P);
  const ring = P.ring, sgE = [];
  for (const [r0, r1] of ring.ranges) { let area = 0; for (let k = r0; k < r1; k++) { const a = P[k], b = P[ring.next[k]]; area += a.x * b.y - b.x * a.y; } sgE[P[r0].e | 0] = area > 0 ? 1 : -1; }
  const q = 2 / (3 * G.uRef * G.uRef), r0 = G.rhoRef;
  if (!G.cp || G.cp.n !== n) G.cp = { n, m: new Float32Array(n), v: new Float32Array(n), ok: new Uint8Array(n), w: 0 };
  const tau = 0.5 * FT(), a = G.cp.w === 0 ? 1 : 1 - Math.exp(-(nsteps || 1) / tau);
  // stagnation pressure can exceed 1 only by viscous effects of order 1/Re at the leading-edge radius
  const cpMax = cpLimit();
  const cur = new Float32Array(n).fill(NaN), raw = new Float32Array(n).fill(NaN), cpN = new Float32Array(n).fill(NaN);
  for (let k = 0; k < n; k++) {
    const A = P[ring.prev[k]], B = P[ring.next[k]], p = P[k];
    if (p.hid) continue; // inside another element: not wetted
    const sg = sgE[p.e | 0];
    let tx = B.x - A.x, ty = B.y - A.y; const L = Math.hypot(tx, ty) || 1; tx /= L; ty /= L;
    const nx = sg * ty, ny = -sg * tx;
    const r1 = rhoFullAt(p.x + nx, p.y + ny), r2 = rhoFullAt(p.x + 2 * nx, p.y + 2 * ny), r3 = rhoFullAt(p.x + 3 * nx, p.y + 3 * ny);
    let r;
    if ((p.te || (!p.e && p.xc > 0.9)) && S.preset !== 'cyl') r = r1; // thin trailing edges, lips and plates: no extrapolation across them
    else if (isFinite(r1) && isFinite(r2) && isFinite(r3)) {
      // least-squares line through the three points, extrapolated to the wall, then limited
      r = (4 * r1 + r2 - 2 * r3) / 3;
      const lo = Math.min(r1, r2, r3), hi = Math.max(r1, r2, r3), span = hi - lo;
      r = Math.min(hi + 0.5 * span, Math.max(lo - 0.5 * span, r));
    } else r = isFinite(r1) ? r1 : r2;
    if (!isFinite(r)) continue;
    raw[k] = (r - r0) * q; cur[k] = Math.min(cpMax, raw[k]);
    if ((p.xc < 0.1 && !p.e) || S.preset === 'cyl') cpN[k] = (r1 - r0) * q;
  }
  // stagnation tap: parabolic fit through the three highest wall readings round the nose, averaged in arc length
  if (!G.arc || G.arc.poly !== P) {
    let kLE = 0; for (let k = 0; k < n; k++) if (!P[k].e && P[k].xc < P[kLE].xc) kLE = k;
    const s = new Float32Array(n);
    for (let k = kLE - 1; k >= 0; k--) s[k] = s[k + 1] + Math.hypot(P[k].x - P[k + 1].x, P[k].y - P[k + 1].y);
    for (let k = kLE + 1; k < n; k++) s[k] = s[k - 1] - Math.hypot(P[k].x - P[k - 1].x, P[k].y - P[k - 1].y);
    G.arc = { poly: P, s, kLE };
  }
  const AS = G.arc.s;
  let ks = -1, best = -1e9;
  for (let k = 1; k < n - 1; k++) if (((P[k].xc < 0.1 && !P[k].e) || S.preset === 'cyl') && isFinite(raw[k]) && raw[k] > best) { best = raw[k]; ks = k; }
  if (ks > 0 && isFinite(raw[ks - 1]) && isFinite(raw[ks + 1])) {
    const s0 = AS[ks - 1], s1 = AS[ks], s2 = AS[ks + 1], y0 = raw[ks - 1], y1 = raw[ks], y2 = raw[ks + 1];
    const d01 = (y1 - y0) / (s1 - s0), d12 = (y2 - y1) / (s2 - s1), c2 = (d12 - d01) / (s2 - s0);
    let sp = s1, cp = y1;
    if (c2 < 0) { sp = 0.5 * (s0 + s1) - d01 / (2 * c2); sp = Math.min(Math.max(sp, Math.min(s0, s2)), Math.max(s0, s2)); cp = y1 + d01 * (sp - s1) + c2 * (sp - s0) * (sp - s1); }
    const st = G.stag, aa = st ? 1 - Math.exp(-(nsteps || 1) / (0.5 * FT())) : 1;
    G.stag = st ? { arc: st.arc + aa * (sp - st.arc), cp: st.cp + aa * (cp - st.cp) } : { arc: sp, cp };
  }
  if (G.rec) surfaceCpAccumulate(cur);
  for (let k = 0; k < n; k++) {
    if (!isFinite(cur[k])) continue;
    const cp = cur[k], C = G.cp;
    if (!C.ok[k]) { C.m[k] = cp; C.v[k] = 0; C.ok[k] = 1; continue; }
    const d = cp - C.m[k];
    C.m[k] += a * d; C.v[k] = (1 - a) * (C.v[k] + a * d * d);
  }
  G.cp.w += nsteps || 1;
}

// ---------- rendering ----------
const view = $('#view'), vctx = view.getContext('2d'), glc = $('#gl');
let scale = 1, dpr = 1;
function sizeCanvas() {
  dpr = Math.min(2, window.devicePixelRatio || 1);
  const w = view.parentElement.clientWidth;
  view.width = Math.round(w * dpr); view.height = Math.round(w * dpr * G.ny / G.nx);
  view.style.height = (w * G.ny / G.nx) + 'px';
  if (MODE === 'gpu') { glc.width = view.width; glc.height = view.height; }
  scale = view.width / G.nx;
  sizeCharts(); dirty = true;
}
const toScr = (x, y) => [(x + 0.5) * scale, (G.ny - 0.5 - y) * scale];
const toLat = (X, Y) => [X / scale - 0.5, G.ny - 0.5 - Y / scale];
function fieldMapName() { if (S.cmap === 'auto' && G && G.play && G.play.view === 'rms') return 'viridis'; return S.cmap === 'auto' ? (FIELDS[S.field].map || 'viridis') : S.cmap; } // fluctuation magnitudes are one-signed: sequential map

function computeScalarCPU() {
  const { nx, ny, ux, uy, rho, mask, sc, Nc } = G, f = S.field;
  if (f === 'speed' || f === 'piv') { for (let i = 0; i < G.N; i++) sc[i] = mask[i] ? 0 : Math.hypot(ux[i], uy[i]) / U0; return; }
  if (f === 'press') { const q = 2 / (3 * G.uRef * G.uRef), r0 = G.rhoRef, P = G.pmgValid ? G.cpP : null, W = G.pw, cap = cpLimit(); for (let i = 0; i < G.N; i++) { if (mask[i]) { sc[i] = 0; continue; } const cpL = (rho[i] - r0) * q; sc[i] = Math.min(cap, P ? W[i] * cpL + (1 - W[i]) * P[i] : cpL); } return; }
  if (f === 'vort' && G.play && G.play.vort && G.play.vort.length === G.N) { sc.set(G.play.vort); return; }
  for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
    const i = y * nx + x; if (mask[i]) { sc[i] = 0; continue; }
    const xl = x > 0 && !mask[i - 1] ? i - 1 : i, xr = x < nx - 1 && !mask[i + 1] ? i + 1 : i;
    const yd = y > 0 && !mask[i - nx] ? i - nx : i, yu = y < ny - 1 && !mask[i + nx] ? i + nx : i;
    const hx = (xr - xl) || 1, hy = ((yu - yd) / nx) || 1;
    if (f === 'vort') sc[i] = ((uy[xr] - uy[xl]) / hx - (ux[yu] - ux[yd]) / hy) * Nc / U0;
    else sc[i] = (rho[xr] - rho[xl]) / hx * 2 * 0.5 / (U0 * U0);
  }
}
function renderCPU() {
  const { nx, ny, mask, img } = G;
  const buf = new Uint32Array(img.data.buffer);
  const f = S.field, R = playRange() || fieldRange();
  if (f === 'smoke') {
    const D = G.dye;
    for (let y = 0; y < ny; y++) { const row = (ny - 1 - y) * nx; for (let x = 0; x < nx; x++) {
      const i = y * nx + x; const d = mask[i] ? 0 : Math.min(1, D[i] * 0.32 * S.contrast); const v = Math.pow(d, 0.75); const l = 0.03 + 0.03 * (x / nx);
      buf[row + x] = (255 << 24) | (((255 * (l + 0.02 + v * 0.93)) | 0) << 16) | (((255 * (l + 0.005 + v * 0.91)) | 0) << 8) | ((255 * (l + v * 0.9)) | 0); } }
  } else if (f === 'piv') {
    buf.fill((255 << 24) | (14 << 16) | (11 << 8) | 9);
  } else {
    computeScalarCPU();
    const map = f === 'schl' ? 'grey' : fieldMapName(); const L = MAPS[map].L, sc = G.sc;
    const lo = f === 'schl' ? -0.35 / S.contrast : R.lo, hi = f === 'schl' ? 0.35 / S.contrast : R.hi, mid = f === 'schl' ? 0 : R.mid, div = f === 'schl' ? false : R.div;
    for (let y = 0; y < ny; y++) { const row = (ny - 1 - y) * nx; for (let x = 0; x < nx; x++) {
      const v = sc[y * nx + x]; let t = div ? (v < mid ? 0.5 * (v - lo) / (mid - lo) : 0.5 + 0.5 * (v - mid) / (hi - mid)) : (v - lo) / (hi - lo);
      t = t < 0 ? 0 : t > 1 ? 1 : t; buf[row + x] = L[(t * 255) | 0]; } }
  }
  G.off.getContext('2d').putImageData(img, 0, 0);
  vctx.imageSmoothingEnabled = true; vctx.imageSmoothingQuality = 'high';
  vctx.drawImage(G.off, 0, 0, view.width, view.height);
  if (f === 'smoke') drawStreaks();
}
function renderGPU() {
  const f = S.field, R = playRange() || fieldRange();
  const mode = FIELD_IDX[f];
  if (G.play) { playbackScalar(mode); if (mode <= 2) gpu.setLUT(fieldMapName(), MAPS[fieldMapName()].C); }
  else if (mode <= 2) { gpu.fields(mode, G.rhoRef, G.F.Nc, G.uRef); gpu.setLUT(fieldMapName(), MAPS[fieldMapName()].C); }
  const W = glc.width, H = glc.height;
  const ax = (S.pos.x * G.F.Nc) / G.F.nx * W, ay = (S.pos.y * G.F.Nc) / G.F.ny * H;
  gpu.display({ pb: !!G.play, cpCap: cpLimit(), psn: G.pmgValid, mode, contours: S.contours && mode <= 2, div: R.div, lo: R.lo, hi: R.hi, mid: R.mid, cstep: R.step, contrast: S.contrast, frame: G.frame,
    bodyY: (() => { let a = 1e9, b = -1e9; for (const p of G.polyF) { a = Math.min(a, p.y); b = Math.max(b, p.y); } return [(a + 0.5) / G.F.ny, (b + 0.5) / G.F.ny]; })(), schK: 4, aperC: [Math.min(Math.max(ax + 0.12 * H, H * 0.49), W - H * 0.49), H * 0.5], aperR: H * 0.49, laser: [S.pos.x / S.L, 2.6] });
  vctx.clearRect(0, 0, view.width, view.height);
}
function render() {
  const f = S.field;
  if (MODE === 'gpu') renderGPU(); else renderCPU();
  const lightBg = (f === 'vort' || f === 'press') && fieldMapName() === 'coolwarm' || (f === 'schl') || ((f === 'vort' || f === 'press') && fieldMapName() === 'grey');
  if (S.contours && MODE === 'cpu' && f !== 'smoke' && f !== 'piv' && f !== 'schl') drawContours(lightBg);
  const flucView = G.play && G.play.view === 'rms';
  if (S.stream && !flucView && f !== 'smoke' && f !== 'schl' && f !== 'piv') drawStreamlines(lightBg);
  if (f === 'piv' && !flucView) { if (S.pivView === 'raw') drawPIVParticles(); else drawPIVProcessed(); }
  else if (S.vectors && !flucView) drawVectors(lightBg);
  drawBody(f);
  if (f === 'piv' && S.pivView === 'raw') drawGlare();
  if (f === 'smoke') drawSmokeWire();
  drawDiffuser(f);
  drawContraction(f);
  drawProbes();
  drawProbe();
  drawColorbar();
}
function drawStreaks() {
  const bins = [new Path2D(), new Path2D(), new Path2D(), new Path2D()];
  for (const st of G.streaks) {
    const X = st.x, Y = st.yy; let cur = -1;
    for (let k = X.length - 1; k > 0; k--) {
      const d = Math.hypot(X[k] - X[k - 1], Y[k] - Y[k - 1]);
      if (d > 2.5) { cur = -1; continue; }
      const b = d < 0.7 ? 0 : d < 1.2 ? 1 : d < 1.8 ? 2 : 3;
      const [x2, y2] = toScr(X[k - 1], Y[k - 1]);
      if (b !== cur) { const [x1, y1] = toScr(X[k], Y[k]); bins[b].moveTo(x1, y1); cur = b; }
      bins[b].lineTo(x2, y2);
    }
  }
  const A = [0.95, 0.7, 0.42, 0.18].map(a => Math.min(1, a * S.contrast));
  vctx.save(); vctx.globalCompositeOperation = 'lighter'; vctx.lineCap = 'butt'; vctx.lineJoin = 'round';
  bins.forEach((p, b) => {
    vctx.strokeStyle = `rgba(150,165,180,${A[b] * 0.16})`; vctx.lineWidth = 5.5 * dpr * scale / 3.4; vctx.stroke(p);
    vctx.strokeStyle = `rgba(235,240,244,${A[b] * 0.75})`; vctx.lineWidth = 1.3 * dpr * Math.max(1, scale / 3.4); vctx.stroke(p);
  });
  vctx.restore();
}
function drawBody(f) {
  const P = G.poly; if (!P) return;
  if (!P.ring) P.ring = elementRing(P);
  const ring = P.ring, fillP = new Path2D(), edge = new Path2D();
  for (const [r0, r1] of ring.ranges) {
    for (let k = r0; k < r1; k++) { const [X, Y] = toScr(P[k].x, P[k].y); k > r0 ? fillP.lineTo(X, Y) : fillP.moveTo(X, Y); }
    fillP.closePath();
    let pen = false; // outline only the wetted surface, so overlapping elements read as one body
    for (let k = r0; k <= r1; k++) {
      const j = k < r1 ? k : r0, p = P[j], [X, Y] = toScr(p.x, p.y);
      if (p.hid) { pen = false; continue; }
      if (pen) edge.lineTo(X, Y); else { edge.moveTo(X, Y); pen = true; }
    }
  }
  const [, ay] = toScr(0, S.pos.y * G.Nc - 0.5 + 0.25 * G.Nc);
  const grd = vctx.createLinearGradient(0, ay, 0, ay + 0.5 * G.Nc * scale);
  if (f === 'schl' || (f === 'piv' && S.pivView === 'raw')) { grd.addColorStop(0, f === 'schl' ? '#2a2c2e' : '#060708'); grd.addColorStop(1, f === 'schl' ? '#141516' : '#030404'); } // opaque model: a silhouette in the schlieren beam and the PIV camera
  else { grd.addColorStop(0, '#cdd3d8'); grd.addColorStop(0.45, '#8e979f'); grd.addColorStop(1, '#4b535a'); }
  vctx.fillStyle = grd; vctx.fill(fillP, 'nonzero');
  vctx.lineWidth = Math.max(1, dpr); vctx.strokeStyle = 'rgba(10,12,14,.85)'; vctx.lineJoin = 'round'; vctx.stroke(edge);
  const [px, py] = toScr(S.pos.x * G.Nc - 0.5, S.pos.y * G.Nc - 0.5);
  vctx.beginPath(); vctx.arc(px, py, 2.6 * dpr, 0, 6.283); vctx.fillStyle = '#1b1f23'; vctx.fill();
  vctx.lineWidth = dpr; vctx.strokeStyle = '#e6eaee'; vctx.stroke();
}
function vel(x, y) { return [sampleField(G.ux, x, y, G.nx, G.ny), sampleField(G.uy, x, y, G.nx, G.ny)]; }
function isSolid(x, y) { const xi = Math.round(x), yi = Math.round(y); if (xi < 0 || yi < 0 || xi >= G.nx || yi >= G.ny) return true; return G.mask[yi * G.nx + xi] === 1; }
function drawStreamlines(lightBg) {
  const { nx, ny } = G;
  const n = Math.max(16, Math.round(ny / 5));
  vctx.lineWidth = Math.max(0.8, 0.9 * dpr);
  vctx.strokeStyle = lightBg ? 'rgba(20,26,32,.5)' : 'rgba(255,255,255,.5)';
  vctx.beginPath();
  const h = 0.6;
  for (let k = 0; k < n; k++) {
    let x = 1, y = (k + 0.5) * (ny - 1) / n;
    let [X, Y] = toScr(x, y); vctx.moveTo(X, Y);
    for (let s = 0; s < nx * 4; s++) {
      let [u, v] = vel(x, y); let m = Math.hypot(u, v); if (m < 1e-5) break;
      const xm = x + 0.5 * h * u / m, ym = y + 0.5 * h * v / m;
      [u, v] = vel(xm, ym); m = Math.hypot(u, v); if (m < 1e-5) break;
      x += h * u / m; y += h * v / m;
      if (x >= nx - 1 || x < 0 || y < 0 || y > ny - 1 || isSolid(x, y)) break;
      if (s % 2 === 0) { [X, Y] = toScr(x, y); vctx.lineTo(X, Y); }
    }
  }
  vctx.stroke();
}
function gauss() { let u = 0, v = 0; while (u === 0) u = rand(); while (v === 0) v = rand(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(6.283 * v); }
function arrow(X, Y, dx, dy, col) {
  vctx.strokeStyle = vctx.fillStyle = col;
  const X2 = X + dx, Y2 = Y + dy;
  vctx.beginPath(); vctx.moveTo(X, Y); vctx.lineTo(X2, Y2); vctx.stroke();
  const a = Math.atan2(dy, dx), hl = Math.min(4.5 * dpr, 0.35 * Math.hypot(dx, dy));
  vctx.beginPath(); vctx.moveTo(X2, Y2);
  vctx.lineTo(X2 - hl * Math.cos(a - 0.45), Y2 - hl * Math.sin(a - 0.45));
  vctx.lineTo(X2 - hl * Math.cos(a + 0.45), Y2 - hl * Math.sin(a + 0.45)); vctx.fill();
}
function drawVectors(lightBg) {
  const { nx, ny } = G, sp = Math.max(2.5, nx / 34);
  vctx.lineWidth = Math.max(1, 1.1 * dpr);
  const col = lightBg ? 'rgba(20,26,32,.75)' : 'rgba(255,255,255,.8)';
  for (let y = sp / 2; y < ny; y += sp) for (let x = sp / 2; x < nx; x += sp) {
    if (isSolid(x, y)) continue;
    const [u, v] = vel(x, y); const [X, Y] = toScr(x, y);
    arrow(X, Y, (u / U0) * sp * 0.85 * scale, -(v / U0) * sp * 0.85 * scale, col);
  }
}
let sprite = null;
function particleSprite() {
  if (sprite) return sprite;
  const c = document.createElement('canvas'); c.width = c.height = 16; const x = c.getContext('2d');
  const g = x.createRadialGradient(8, 8, 0, 8, 8, 8); g.addColorStop(0, 'rgba(255,255,255,1)'); g.addColorStop(0.35, 'rgba(235,245,240,0.55)'); g.addColorStop(1, 'rgba(235,245,240,0)');
  x.fillStyle = g; x.fillRect(0, 0, 16, 16); return (sprite = c);
}
function drawPIVParticles() {
  const P = G.piv, sh = shadowGrid(); if (!P) return;
  const spr = particleSprite();
  if (G.frame % 3 === 0 || !G.pulse) G.pulse = 1 + 0.08 * gauss(); // laser pulse energy varies shot to shot
  const lx = (S.pos.x + 0.75) * G.Nc - 0.5, wSheet = 0.36 * G.nx; // sheet centred a little downstream of the model
  vctx.save(); vctx.globalCompositeOperation = 'lighter';
  for (let k = 0; k < P.length; k += 3) {
    const x = P[k], y = P[k + 1]; const i = Math.round(y) * G.nx + Math.round(x);
    if (sh[i]) continue;
    // uneven seeding: smooth clumps that drift with the flow
    const seed = 0.55 + 0.45 * Math.sin(0.11 * x - 0.0011 * G.t + 1.7 * Math.sin(0.07 * y)) * Math.sin(0.09 * y + 0.6);
    if (P[k + 2] > 0.35 + 0.65 * Math.min(1, seed + 0.25)) continue;
    const depth = P[k + 2]; // position across the light sheet: brightest at its centre
    const sheet = (0.6 + 0.4 * (y / G.ny)) * Math.exp(-(((x - lx) / wSheet) ** 2)) * G.pulse;
    vctx.globalAlpha = Math.min(1, depth * depth * sheet);
    // particle images differ in size and are slightly astigmatic
    const hsh = Math.sin(k * 12.9898) * 43758.5453, fr = hsh - Math.floor(hsh);
    const s = (0.9 + 1.9 * depth * depth) * dpr, ax = 0.8 + 0.45 * fr, ay = 1.25 - 0.45 * fr;
    const [X, Y] = toScr(x, y); vctx.drawImage(spr, X - s * ax, Y - s * ay, 2 * s * ax, 2 * s * ay);
  }
  vctx.restore();
}
// Processed PIV: 0.15c interrogation windows on a 0.05c grid (67% overlap), the usual DaVis-style
// colour map of |V| with every second vector drawn; no data on the model or in the laser shadow.
function computePIV() {
  const { nx, ny, ux, uy, mask } = G, sh = shadowGrid();
  const win = 6, step = 2, gw = Math.floor((nx - win) / step) + 1, gh = Math.floor((ny - win) / step) + 1;
  const P = G.pivF && G.pivF.gw === gw && G.pivF.gh === gh ? G.pivF : (G.pivF = { gw, gh, win, step, u: new Float32Array(gw * gh), v: new Float32Array(gw * gh), ok: new Uint8Array(gw * gh) });
  for (let j = 0; j < gh; j++) for (let k = 0; k < gw; k++) {
    const x0 = k * step, y0 = j * step; let su = 0, sv = 0, su2 = 0, sv2 = 0, n = 0, lit = 0;
    for (let y = y0; y < y0 + win; y++) for (let x = x0; x < x0 + win; x++) {
      const i = y * nx + x; if (mask[i]) continue; if (!sh[i]) lit++;
      const u = ux[i], v = uy[i]; su += u; sv += v; su2 += u * u; sv2 += v * v; n++;
    }
    const id = j * gw + k, cells = win * win, ci = (y0 + (win >> 1)) * nx + x0 + (win >> 1);
    if (n < 0.6 * cells || mask[ci] || sh[ci] || x0 < 0.5 * G.Nc || x0 + win > nx - 0.08 * G.F.nx / G.F.k) { P.ok[id] = 0; continue; }
    let u = su / n, v = sv / n;
    const spread = Math.sqrt(Math.max(0, su2 / n - u * u + sv2 / n - v * v)) / U0;
    const err = (0.003 + 0.2 * spread) * U0;
    u += gauss() * err; v += gauss() * err;
    if (rand() < Math.min(0.05, 0.25 * spread * spread)) { const a = rand() * 6.283, s = (0.3 + rand()) * U0; u = s * Math.cos(a); v = s * Math.sin(a); } // outlier the median test missed
    P.u[id] = u; P.v[id] = v; P.ok[id] = 1;
  }
  P.t = G.t;
}
function drawPIVProcessed() {
  if (!G.pivF || G.frame % 3 === 0 || G.pivF.t === undefined) computePIV();
  const P = G.pivF, { gw, gh, step, win } = P, C = MAPS.turbo.C;
  if (!G.pivImg || G.pivImg.width !== gw) { G.pivImg = document.createElement('canvas'); G.pivImg.width = gw; G.pivImg.height = gh; }
  const ctx = G.pivImg.getContext('2d'), img = ctx.createImageData(gw, gh), d = img.data;
  for (let j = 0; j < gh; j++) for (let k = 0; k < gw; k++) {
    const id = j * gw + k, o = ((gh - 1 - j) * gw + k) * 4;
    if (!P.ok[id]) { d[o + 3] = 0; continue; }
    const c = C[Math.min(255, Math.round(Math.hypot(P.u[id], P.v[id]) / U0 / 1.6 * S.contrast * 255))];
    d[o] = c[0]; d[o + 1] = c[1]; d[o + 2] = c[2]; d[o + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  const xc0 = win / 2 - 0.5, yc0 = win / 2 - 0.5;
  const [X0, Ytop] = toScr(xc0 - step / 2, yc0 + (gh - 0.5) * step);
  vctx.imageSmoothingEnabled = true; vctx.drawImage(G.pivImg, X0, Ytop, gw * step * scale, gh * step * scale);
  // vectors
  const shafts = new Path2D(), heads = new Path2D(), L = 2 * step * 0.9 * scale;
  for (let j = 0; j < gh; j += 2) for (let k = 0; k < gw; k += 2) {
    const id = j * gw + k; if (!P.ok[id]) continue;
    const [X, Y] = toScr(xc0 + k * step, yc0 + j * step);
    const dx = P.u[id] / U0 * L, dy = -P.v[id] / U0 * L, X2 = X + dx, Y2 = Y + dy;
    shafts.moveTo(X, Y); shafts.lineTo(X2, Y2);
    const a = Math.atan2(dy, dx), hl = Math.min(4 * dpr, 0.35 * Math.hypot(dx, dy));
    heads.moveTo(X2, Y2); heads.lineTo(X2 - hl * Math.cos(a - 0.45), Y2 - hl * Math.sin(a - 0.45)); heads.lineTo(X2 - hl * Math.cos(a + 0.45), Y2 - hl * Math.sin(a + 0.45)); heads.closePath();
  }
  vctx.strokeStyle = vctx.fillStyle = 'rgba(12,14,16,0.9)'; vctx.lineWidth = Math.max(1, 0.9 * dpr);
  vctx.stroke(shafts); vctx.fill(heads);
}
function drawPIVVectors() {
  const { nx, ny, ux, uy, mask } = G, sh = shadowGrid();
  const win = Math.max(2, Math.round(24 / G.F.k)), step = win / 2;
  const C = MAPS.turbo.C;
  vctx.lineWidth = Math.max(1, 1.1 * dpr);
  for (let y0 = 0; y0 + win <= ny; y0 += step) for (let x0 = 0; x0 + win <= nx; x0 += step) {
    let su = 0, sv = 0, su2 = 0, sv2 = 0, n = 0, lit = 0;
    for (let y = y0; y < y0 + win; y++) for (let x = x0; x < x0 + win; x++) {
      const i = y * nx + x; if (mask[i]) continue;
      if (!sh[i]) lit++;
      const u = ux[i], v = uy[i]; su += u; sv += v; su2 += u * u; sv2 += v * v; n++;
    }
    const cells = win * win;
    if (n < 0.6 * cells || lit < 0.5 * cells) continue; // window on the model or in the laser shadow: no vector
    let u = su / n, v = sv / n;
    const grad = Math.sqrt(Math.max(0, su2 / n - u * u + sv2 / n - v * v)) / U0; // in-window velocity spread
    const err = (0.004 + 0.25 * grad) * U0;
    u += gauss() * err; v += gauss() * err;
    const bad = rand() < Math.min(0.08, 0.0015 + 0.35 * grad * grad);
    const xc = x0 + win / 2 - 0.5, yc = y0 + win / 2 - 0.5;
    const [X, Y] = toScr(xc, yc);
    const m = Math.hypot(u, v) / U0;
    let col;
    if (bad) col = '#ffffff'; // failed the median test and was replaced by the neighbour median
    else { const c = C[Math.min(255, Math.round(m / 1.6 * S.contrast * 255))]; col = `rgb(${c[0]},${c[1]},${c[2]})`; }
    arrow(X, Y, (u / U0) * step * 0.9 * scale, -(v / U0) * step * 0.9 * scale, col);
  }
}
// laser light scattered off the illuminated (upper) surface of the model
function drawGlare() {
  const P = G.poly; if (!P.ring) P.ring = elementRing(P);
  const lit = (p) => p.s === 1 && !p.hid; // the lit (upper) surface, broken wherever it is covered or changes element
  vctx.save(); vctx.globalCompositeOperation = 'lighter';
  vctx.beginPath(); P.forEach((p, k) => { if (!lit(p)) return; const [X, Y] = toScr(p.x, p.y); const a = P[P.ring.prev[k]]; lit(a) && P.ring.prev[k] === k - 1 ? vctx.lineTo(X, Y) : vctx.moveTo(X, Y); });
  vctx.strokeStyle = 'rgba(210,255,225,0.55)'; vctx.lineWidth = 3.2 * dpr; vctx.shadowColor = 'rgba(160,255,200,0.9)'; vctx.shadowBlur = 9 * dpr; vctx.stroke();
  vctx.shadowBlur = 0; vctx.strokeStyle = 'rgba(245,255,248,0.9)'; vctx.lineWidth = 1.2 * dpr; vctx.stroke();
  vctx.restore();
}
// inlet region (absorbing layer and turbulence grid) drawn as the contraction exit, with the pitot-static probe
function drawContraction(f) {
  const xe = 0.5 * G.Nc * scale, H = view.height;
  const dark = f === 'smoke' || f === 'schl' || f === 'piv';
  const g = vctx.createLinearGradient(0, 0, xe, 0);
  g.addColorStop(0, dark ? 'rgba(8,10,13,0.94)' : 'rgba(20,26,32,0.82)'); g.addColorStop(0.62, dark ? 'rgba(8,10,13,0.78)' : 'rgba(20,26,32,0.55)'); g.addColorStop(1, 'rgba(20,26,32,0)');
  vctx.fillStyle = g; vctx.fillRect(0, 0, xe, H);
  vctx.save(); vctx.beginPath(); vctx.rect(0, 0, 0.62 * xe, H); vctx.clip();
  vctx.strokeStyle = 'rgba(255,255,255,0.08)'; vctx.lineWidth = dpr;
  for (let y = -xe; y < H + xe; y += 7 * dpr) { vctx.beginPath(); vctx.moveTo(0, y + xe); vctx.lineTo(xe, y); vctx.stroke(); }
  vctx.restore();
  vctx.save(); vctx.translate(0.3 * xe, H / 2); vctx.rotate(-Math.PI / 2);
  vctx.font = `600 ${10 * dpr}px "Barlow Condensed", "Arial Narrow", sans-serif`; vctx.fillStyle = 'rgba(230,236,240,0.7)'; vctx.textAlign = 'center';
  vctx.fillText('CONTRACTION', 0, 0); vctx.restore();
  // pitot-static probe on a strut from the floor, nose pointing upstream
  const [px, py] = toScr(0.7 * G.Nc - 0.5, 0.32 * G.Nc), [, pyFloor] = toScr(0, -0.5);
  const col = (f === 'vort' || f === 'press' || f === 'schl') ? 'rgba(30,36,42,0.85)' : 'rgba(225,232,238,0.85)';
  vctx.lineCap = 'round'; vctx.beginPath(); vctx.moveTo(px + 0.12 * G.Nc * scale, pyFloor); vctx.lineTo(px + 0.12 * G.Nc * scale, py); vctx.moveTo(px - 0.06 * G.Nc * scale, py); vctx.lineTo(px + 0.12 * G.Nc * scale, py);
  const darkLine = col.startsWith('rgba(30'); // outline in the opposite tone, so the probe reads on light and dark fields alike
  vctx.strokeStyle = darkLine ? 'rgba(235,240,244,0.55)' : 'rgba(6,9,12,0.55)'; vctx.lineWidth = 3.6 * dpr; vctx.stroke();
  vctx.strokeStyle = col; vctx.lineWidth = 1.6 * dpr; vctx.stroke();
  vctx.font = `500 ${9.5 * dpr}px "JetBrains Mono", monospace`; vctx.textAlign = 'left';
  const lx = px + 0.18 * G.Nc * scale, tw = vctx.measureText('pitot-static').width; // label on a dark plate, like the HUD, so it reads on any field
  vctx.fillStyle = 'rgba(6,9,12,0.62)'; vctx.fillRect(lx - 4 * dpr, py - 7 * dpr, tw + 8 * dpr, 14 * dpr);
  vctx.fillStyle = '#e3e9ee'; vctx.fillText('pitot-static', lx, py + 3.5 * dpr);
}
function drawSmokeWire() {
  const R = smokeRelease(); if (S.smokeSrc !== 'wire') return;
  const [x, y0] = toScr(R.x0, R.y0), [, y1] = toScr(R.x0, R.y1);
  vctx.strokeStyle = 'rgba(200,170,120,0.55)'; vctx.lineWidth = Math.max(1, 0.8 * dpr);
  vctx.beginPath(); vctx.moveTo(x, y0 + 4 * dpr); vctx.lineTo(x, y1 - 4 * dpr); vctx.stroke();
}
// outlet absorbing layer drawn as the diffuser entry
function drawDiffuser(f) {
  const sw = Math.max(8, Math.round(G.F.nx * 0.08)) / G.F.k;
  const x0 = (G.nx - sw) * scale, W = view.width - x0, H = view.height;
  const g = vctx.createLinearGradient(x0, 0, view.width, 0);
  const dark = f === 'smoke' || f === 'schl' || f === 'piv';
  g.addColorStop(0, dark ? 'rgba(8,10,13,0)' : 'rgba(20,26,32,0)'); g.addColorStop(0.35, dark ? 'rgba(8,10,13,0.75)' : 'rgba(20,26,32,0.55)'); g.addColorStop(1, dark ? 'rgba(8,10,13,0.92)' : 'rgba(20,26,32,0.8)');
  vctx.fillStyle = g; vctx.fillRect(x0, 0, W, H);
  vctx.save(); vctx.beginPath(); vctx.rect(x0 + 0.35 * W, 0, 0.65 * W, H); vctx.clip();
  vctx.strokeStyle = 'rgba(255,255,255,0.08)'; vctx.lineWidth = dpr;
  for (let y = -W; y < H + W; y += 7 * dpr) { vctx.beginPath(); vctx.moveTo(x0, y); vctx.lineTo(x0 + W, y + W); vctx.stroke(); }
  vctx.restore();
  vctx.save(); vctx.translate(view.width - 0.32 * W, H / 2); vctx.rotate(-Math.PI / 2);
  vctx.font = `600 ${10 * dpr}px "Barlow Condensed", "Arial Narrow", sans-serif`; vctx.fillStyle = 'rgba(230,236,240,0.7)'; vctx.textAlign = 'center';
  vctx.fillText('DIFFUSER', 0, 0); vctx.restore();
}
function drawContours(lightBg) {
  const { nx, ny, sc, mask } = G; const R = fieldRange();
  vctx.lineWidth = Math.max(0.6, 0.7 * dpr);
  vctx.strokeStyle = lightBg ? 'rgba(15,20,25,.45)' : 'rgba(255,255,255,.4)';
  vctx.beginPath();
  for (let iso = Math.ceil(R.lo / R.step) * R.step; iso < R.hi; iso += R.step) {
    for (let y = 0; y < ny - 1; y++) for (let x = 0; x < nx - 1; x++) {
      const i = y * nx + x;
      if (mask[i] | mask[i + 1] | mask[i + nx] | mask[i + nx + 1]) continue;
      const a = sc[i], b = sc[i + 1], c = sc[i + nx + 1], d = sc[i + nx];
      if ((a > iso) === (b > iso) && (b > iso) === (c > iso) && (c > iso) === (d > iso)) continue;
      const pts = [];
      const e = (p, q, xp, yp, xq, yq) => { const t = (iso - p) / (q - p); pts.push([xp + (xq - xp) * t, yp + (yq - yp) * t]); };
      if ((a > iso) !== (b > iso)) e(a, b, x, y, x + 1, y);
      if ((b > iso) !== (c > iso)) e(b, c, x + 1, y, x + 1, y + 1);
      if ((c > iso) !== (d > iso)) e(c, d, x + 1, y + 1, x, y + 1);
      if ((d > iso) !== (a > iso)) e(d, a, x, y + 1, x, y);
      for (let k = 0; k + 1 < pts.length; k += 2) { const [X1, Y1] = toScr(pts[k][0], pts[k][1]), [X2, Y2] = toScr(pts[k + 1][0], pts[k + 1][1]); vctx.moveTo(X1, Y1); vctx.lineTo(X2, Y2); }
    }
  }
  vctx.stroke();
}
function drawProbe() {
  const el = $('#hudProbe');
  if (!S.probe) { el.hidden = true; return; }
  const { x, y } = S.probe; const [X, Y] = toScr(x, y);
  vctx.strokeStyle = '#ffffff'; vctx.lineWidth = 1.2 * dpr;
  vctx.beginPath(); vctx.arc(X, Y, 6 * dpr, 0, 6.283);
  for (const [a, b, c, d] of [[-11, 0, -3, 0], [3, 0, 11, 0], [0, -11, 0, -3], [0, 3, 0, 11]]) { vctx.moveTo(X + a * dpr, Y + b * dpr); vctx.lineTo(X + c * dpr, Y + d * dpr); }
  vctx.stroke();
  if (isSolid(x, y)) { el.textContent = 'inside model'; el.hidden = false; return; }
  const [u, v] = vel(x, y);
  const xc = (x + 0.5) / G.Nc - S.pos.x + 0.25, yc = (y + 0.5) / G.Nc - S.pos.y;
  el.textContent = `x/c ${xc.toFixed(2)} y/c ${yc.toFixed(2)} · |u|/U∞ ${(Math.hypot(u, v) / U0).toFixed(2)} at ${(Math.atan2(v, u) * 180 / Math.PI).toFixed(0)}° · Cp ${cpAt(x, y).toFixed(2)}`;
  el.hidden = false;
}
const cbar = $('#cbar'), cbctx = cbar.getContext('2d');
function colourbarSpec() { // colour scale of the current view: colours, end labels and names (also used by the exports)
  const f = S.field, F = FIELDS[f], R = playRange() || fieldRange(), rms = !!(G && G.play && G.play.view === 'rms');
  const C = f === 'smoke' ? null : MAPS[f === 'piv' ? 'turbo' : f === 'schl' ? 'grey' : fieldMapName()].C;
  const col = (t) => { if (!C) { const v = 1 - Math.exp(-2.6 * t); const a = Math.round(255 * (0.03 + v * 0.9)); return [a, a, Math.min(255, a + 4)]; } return C[Math.max(0, Math.min(255, Math.round(t * 255)))]; };
  const fmt = (v) => (Math.abs(v) < 1 && v !== 0 ? v.toFixed(2) : v.toFixed(1)).replace('-', '−');
  let lo, hi, mid = '';
  if (f === 'smoke') { lo = 'clear'; hi = 'dense'; }
  else if (f === 'schl') { lo = '−∂ρ/∂y'; hi = '+∂ρ/∂y'; }
  else if (f === 'piv') { lo = '0'; hi = fmt(1.6 / S.contrast); }
  else { lo = fmt(R.lo); hi = fmt(R.hi); if (R.div) mid = `centre ${fmt(R.mid)}`; }
  const unit = rms ? { speed: '√(u′² + v′²) / U∞', vort: 'ω′ c / U∞ (rms)', press: 'Cp′ (rms)', piv: '|u| / U∞', schl: '', smoke: '' }[f]
    : { speed: '|u| / U∞', vort: 'ω c / U∞', press: 'Cp', piv: '|u| / U∞', schl: '', smoke: '' }[f];
  // five labelled ticks; a diverging map is linear on each side of its centre, so the quarter ticks sit half-way to each end
  let ticks = [];
  if (f !== 'smoke' && f !== 'schl') {
    const a = f === 'piv' ? 0 : R.lo, b = f === 'piv' ? 1.6 / S.contrast : R.hi;
    const tv = R.div && f !== 'piv' ? [[0, a], [0.25, (a + R.mid) / 2], [0.5, R.mid], [0.75, (R.mid + b) / 2], [1, b]] // piecewise: quarter ticks half-way to each end
      : niceTicks(a, b, 5).map(v => [(v - a) / (b - a), v]);
    const dec = (v) => { for (let d = 0; d < 3; d++) if (Math.abs(+v.toFixed(d) - v) < 1e-6 * Math.max(1, Math.abs(b - a))) return d; return 2; };
    const d = Math.max(...tv.map(([, v]) => dec(v)));
    ticks = tv.map(([t, v]) => [t, v.toFixed(d).replace(/^-(?=0?\.?0*$)/, '').replace('-', '−')]);
  }
  return { col, lo, hi, mid, ticks, name: F.name, desc: F.desc, unit };
}
function drawColorbar() {
  const s = colourbarSpec(), F = FIELDS[S.field];
  for (let k = 0; k < 512; k++) { const c = s.col(k / 511); cbctx.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`; cbctx.fillRect(k, 0, 1, 10); }
  $('#cbMin').textContent = s.lo; $('#cbMax').textContent = s.hi;
  const pv = G.play ? (G.play.view === 'mean' ? 'time-averaged · ' : G.play.view === 'rms' ? 'fluctuation rms · ' : 'recorded · ') : '';
  $('#cbName').innerHTML = `<b>${F.name}</b> ${pv}${F.desc}`;
  $('#cbMid').textContent = s.mid;
  $('#hudField').textContent = F.hud;
}

// ---------- charts ----------
const chF = $('#chForces'), chC = $('#chCp');
function sizeCharts() { for (const c of [chF, chC, $('#chProbe')]) { if (!c) continue; const w = c.clientWidth || 300; c.width = Math.round(w * dpr); c.height = Math.round(170 * dpr); } }
function tokens() { const cs = getComputedStyle(document.documentElement); const g = (n) => cs.getPropertyValue(n).trim(); return { ink: g('--ink'), muted: g('--muted'), line: g('--line'), acc: g('--accent'), c2: g('--c2'), panel: g('--panel') }; }
function axes(ctx, W, H, pad, xr, yr, T, xlab, xt, yt) {
  ctx.clearRect(0, 0, W, H);
  ctx.font = `${10.5 * dpr}px "JetBrains Mono", monospace`; ctx.fillStyle = T.muted; ctx.strokeStyle = T.line; ctx.lineWidth = dpr;
  const X = (v) => pad.l + (v - xr[0]) / (xr[1] - xr[0]) * (W - pad.l - pad.r);
  const Y = (v) => H - pad.b - (v - yr[0]) / (yr[1] - yr[0]) * (H - pad.t - pad.b);
  ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
  for (const v of yt) { ctx.beginPath(); ctx.moveTo(pad.l, Y(v)); ctx.lineTo(W - pad.r, Y(v)); ctx.stroke(); ctx.fillText(String(+v.toFixed(2)).replace('-', '−'), pad.l - 4 * dpr, Y(v)); }
  ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  for (const v of xt) ctx.fillText(String(+v.toFixed(2)), X(v), H - pad.b + 4 * dpr);
  if (xlab) { ctx.textAlign = 'right'; ctx.fillText(xlab, W - pad.r, H - pad.b + 4 * dpr + 12 * dpr); }
  return { X, Y };
}
function niceTicks(lo, hi, n) { const st = Math.pow(10, Math.floor(Math.log10((hi - lo) / n))); const s = [1, 2, 2.5, 5, 10].map(m => m * st).find(s => (hi - lo) / s <= n) || st * 10; const out = []; for (let v = Math.ceil(lo / s) * s; v <= hi + 1e-9; v += s) out.push(+v.toFixed(6)); return out; }
function drawForceChart() {
  const ctx = chF.getContext('2d'), W = chF.width, H = chF.height, T = tokens();
  const h = G.hist;
  const pad = { l: 36 * dpr, r: 8 * dpr, t: 8 * dpr, b: 30 * dpr };
  let lo = 0, hi = 0.5;
  for (const p of h) { lo = Math.min(lo, p.cl, p.cd); hi = Math.max(hi, p.cl, p.cd); }
  lo = Math.floor(lo * 4) / 4; hi = Math.ceil(hi * 4) / 4;
  const tp = (G.play ? G.play.M.dtStep : flowTime(1)) * 1000, t1 = h.length ? h[h.length - 1].t : 1, t0 = G.play && h.length ? h[0].t : Math.max(0, t1 - 4 * FT());
  const xr = [t0 * tp, Math.max(t1, t0 + 1) * tp];
  const { X, Y } = axes(ctx, W, H, pad, xr, [lo, hi], T, 'ms', niceTicks(xr[0], xr[1], 4), niceTicks(lo, hi, 4));
  ctx.strokeStyle = T.ink; ctx.globalAlpha = 0.5; ctx.beginPath(); ctx.moveTo(pad.l, Y(0)); ctx.lineTo(W - pad.r, Y(0)); ctx.stroke(); ctx.globalAlpha = 1;
  // shade the settling period
  const settleEnd = G.play ? -1 : G.t - G.since + (G.firstRun ? 2 : 1) * FT();
  if (G.play && G.play.view === 'inst' && G.play.idx >= 0) { ctx.strokeStyle = T.ink; ctx.globalAlpha = 0.6; ctx.lineWidth = dpr; const xp = X(G.t * tp); ctx.beginPath(); ctx.moveTo(xp, pad.t); ctx.lineTo(xp, H - pad.b); ctx.stroke(); ctx.globalAlpha = 1; }
  if (settleEnd > t0) { ctx.fillStyle = T.line; ctx.globalAlpha = 0.45; ctx.fillRect(X(t0 * tp), pad.t, Math.max(0, X(Math.min(settleEnd, t1) * tp) - X(t0 * tp)), H - pad.t - pad.b); ctx.globalAlpha = 1; }
  for (const [key, col] of [['cd', T.c2], ['cl', T.acc]]) {
    ctx.strokeStyle = col; ctx.lineWidth = 1.6 * dpr; ctx.beginPath();
    h.forEach((p, k) => { const x = X(p.t * tp), y = Y(p[key]); k ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.stroke();
    if (h.length) { const p = h[h.length - 1]; ctx.fillStyle = col; ctx.beginPath(); ctx.arc(X(p.t * tp), Y(p[key]), 3 * dpr, 0, 6.283); ctx.fill(); }
  }
  if (!h.length) { ctx.fillStyle = T.muted; ctx.textAlign = 'center'; ctx.fillText('flow accelerating to speed…', W / 2, H / 2 - 8 * dpr); }
}
function drawCpChart() {
  const ctx = chC.getContext('2d'), W = chC.width, H = chC.height, T = tokens();
  const C = G.cp, P = G.polyF;
  const pad = { l: 36 * dpr, r: 8 * dpr, t: 8 * dpr, b: 30 * dpr };
  let lo = -1, hi = 1.5;
  const pts = [], fpts = []; // main element; flap (plotted dashed); both at their stations along the stowed chord line, as the model sits
  const inst = (xc) => S.preset === 'cyl' || xc <= 0.99;
  let xmax = 1, fte = -1e9;
  for (const p of P) if (p.e) fte = Math.max(fte, p.xc);
  if (C && C.n === P.length) for (let k = 0; k < C.n; k++) {
    const p = P[k]; if (!C.ok[k] || p.hid || p.cove) continue; // no taps inside the flap cove or on the shroud lip
    const sd = Math.sqrt(C.v[k]), o = { xc: p.xc, s: p.s, e: p.e | 0, m: -C.m[k], sd };
    if (o.e) { if (p.xc > fte - 0.025) continue; fpts.push(o); xmax = Math.max(xmax, p.xc); } // no taps in the last 2.5% of the flap: the sharp trailing edge is a singular point on the grid
    else if (inst(p.xc)) pts.push(o); else continue;
    lo = Math.min(lo, o.m - sd); hi = Math.max(hi, o.m + sd);
  }
  lo = Math.max(-2, Math.floor(lo * 2) / 2); hi = Math.min(8, Math.ceil(hi * 2) / 2); xmax = Math.ceil(xmax * 10 - 1e-6) / 10;
  const { X, Y } = axes(ctx, W, H, pad, [0, xmax], [lo, hi], T, fpts.length ? 'x/c (deflected geometry)' : 'x/c', niceTicks(0, xmax, 6), niceTicks(lo, hi, 4));
  ctx.strokeStyle = T.ink; ctx.globalAlpha = 0.5; ctx.beginPath(); ctx.moveTo(pad.l, Y(0)); ctx.lineTo(W - pad.r, Y(0)); ctx.stroke(); ctx.globalAlpha = 1;
  const le = pts.find(p => p.xc === 0); // the leading-edge point is shared by both surfaces
  let stg = null;
  if (G.stag && G.arc && S.preset !== 'cyl') {
    const AS = G.arc.s; let kb = 0; for (let k = 0; k < AS.length; k++) if (Math.abs(AS[k] - G.stag.arc) < Math.abs(AS[kb] - G.stag.arc)) kb = k;
    stg = { xc: P[kb].xc, s: P[kb].s, m: -G.stag.cp, sd: 0 };
  }
  for (const [sgn, col] of [[1, T.acc], [-1, T.c2]]) {
    let s = pts.filter(p => p.s === sgn && p.xc > 0);
    if (le) s.push(le);
    if (stg && stg.s === sgn) { s = s.filter(p => Math.abs(p.xc - stg.xc) > 0.002 || p === le); s.push(stg); }
    s.sort((a, b) => a.xc - b.xc);
    if (s.length < 2) continue;
    ctx.fillStyle = col; ctx.globalAlpha = 0.16; ctx.beginPath();
    s.forEach((p, k) => { const x = X(p.xc), y = Y(p.m + p.sd); k ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    for (let k = s.length - 1; k >= 0; k--) ctx.lineTo(X(s[k].xc), Y(s[k].m - s[k].sd));
    ctx.fill(); ctx.globalAlpha = 1;
    ctx.strokeStyle = col; ctx.lineWidth = 1.4 * dpr; ctx.beginPath();
    s.forEach((p, k) => { const x = X(p.xc), y = Y(p.m); k ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.stroke();
    // pressure taps: dense round the nose, cosine-spaced behind it, as on a real tapped model
    const tapX = [0, 0.0025, 0.005, 0.01, 0.02, 0.035];
    for (let j = 0; j < 12; j++) tapX.push(0.05 + 0.5 * (1 - Math.cos(Math.PI * j / 11)) * 0.94);
    for (const xt of tapX) {
      let best = null; for (const p of s) if (p !== stg && (!best || Math.abs(p.xc - xt) < Math.abs(best.xc - xt))) best = p;
      if (!best || Math.abs(best.xc - xt) > 0.02) continue;
      ctx.beginPath(); ctx.arc(X(best.xc), Y(best.m), 2.6 * dpr, 0, 6.283); ctx.fillStyle = T.panel; ctx.fill(); ctx.lineWidth = 1.3 * dpr; ctx.strokeStyle = col; ctx.stroke();
    }
  }
  if (fpts.length) { // flap: both faces, dashed, with a lighter band
    for (const e of [...new Set(fpts.map(p => p.e))]) for (const [sgn, col] of [[1, T.acc], [-1, T.c2]]) {
      const s = fpts.filter(p => p.e === e && p.s === sgn).sort((a, b) => a.xc - b.xc); if (s.length < 2) continue;
      ctx.fillStyle = col; ctx.globalAlpha = 0.1; ctx.beginPath();
      s.forEach((p, k) => { const x = X(p.xc), y = Y(p.m + p.sd); k ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
      for (let k = s.length - 1; k >= 0; k--) ctx.lineTo(X(s[k].xc), Y(s[k].m - s[k].sd));
      ctx.fill(); ctx.globalAlpha = 1;
      ctx.strokeStyle = col; ctx.lineWidth = 1.4 * dpr; ctx.setLineDash([5 * dpr, 3 * dpr]); ctx.beginPath();
      s.forEach((p, k) => { const x = X(p.xc), y = Y(p.m); k ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.stroke(); ctx.setLineDash([]);
    }
    ctx.fillStyle = T.muted; ctx.textAlign = 'right'; ctx.textBaseline = 'top'; ctx.fillText('dashed: flap', W - pad.r - 2 * dpr, pad.t + 2 * dpr);
  }
  if (stg) {
    ctx.beginPath(); ctx.arc(X(stg.xc), Y(stg.m), 3.6 * dpr, 0, 6.283); ctx.fillStyle = stg.s < 0 ? T.c2 : T.acc; ctx.fill();
    ctx.fillStyle = T.muted; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
    ctx.fillText(`stagnation Cp ${(-stg.m).toFixed(2)}`, X(stg.xc) + 7 * dpr, Y(stg.m) - 7 * dpr);
  }
  if (!pts.length) { ctx.fillStyle = T.muted; ctx.textAlign = 'center'; ctx.fillText('waiting for surface data…', W / 2, H / 2); }
}

// ---------- readouts and data reduction ----------
function flowTime(steps) { return steps * (U0 / G.F.Nc) * (S.chord / S.speed); }
const fmtN = (v, d = 3) => (isFinite(v) ? v.toFixed(d) : '–').replace(/^-/, '−');
function stats(arr) { const m = arr.reduce((a, b) => a + b, 0) / arr.length; const sd = Math.sqrt(arr.reduce((a, b) => a + (b - m) * (b - m), 0) / arr.length); return { m, sd, lo: Math.min(...arr), hi: Math.max(...arr) }; }
// Allen & Vincenti two-dimensional wall corrections for a closed test section
function corrections(cl, cd, cm, cdMean) {
  if (S.walls !== 'solid') return null;
  const ch = 1 / S.H; // chord over tunnel height
  const sigma = Math.PI * Math.PI / 48 * ch * ch;
  const tc = S.preset === 'cyl' ? 0.5 : S.t / 100;
  const Lam = S.preset === 'cyl' ? 0.6 : 2.2 * tc; // body-shape factor, roughly linear in thickness for 4-digit sections
  const esb = Lam * sigma, ewb = ch / 4 * (cdMean !== undefined ? cdMean : cd), e = esb + ewb;
  return { cl: cl * (1 - sigma - 2 * e), cd: cd * (1 - 3 * esb - 2 * ewb), cm: cm * (1 - 2 * e) + sigma * cl / 4,
    aoa: S.aoa + 57.3 * sigma / (2 * Math.PI) * (cl + 4 * cm), q: 1 + 2 * e, sigma, e };
}
function updateReadouts() {
  const h = G.hist;
  $('#clock').innerHTML = `t = <b>${(flowTime(G.t) * 1000).toFixed(1)} ms</b> · step <b>${G.t.toLocaleString()}</b>`;
  const need = (G.firstRun ? 2 : 1) * FT();
  if (G.firstRun && G.since >= need) G.firstRun = false;
  const settled = G.since >= need || (!G.firstRun && G.since >= FT());
  const win = G.play ? h : h.filter(p => p.t > (h.length ? h[h.length - 1].t : 0) - 0.5 * FT());
  if (!settled || win.length < 3) {
    const pc = Math.min(99, Math.round(100 * G.since / need));
    for (const id of ['vCL', 'vCD', 'vLD', 'vCM']) $('#' + id).textContent = '–';
    $('#dCL').textContent = `settling ${pc}%`; $('#dCD').textContent = 'wake clearing'; $('#dLD').textContent = ' '; $('#dCM').textContent = 'nose-up positive';
    updateState(null, pc); return;
  }
  const cl = stats(win.map(p => p.cl)), cd = stats(win.map(p => p.cd)), cm = stats(win.map(p => p.cm));
  const wf = win.filter(p => p.clF !== undefined);
  const cor = corrections(cl.m, cd.m, cm.m);
  // the flap's lift is corrected with the same wall factor as the total (the lift correction is a scale factor), so the share is a like-for-like ratio
  const kL = cor && Math.abs(cl.m) > 0.05 ? cor.cl / cl.m : 1, kD = cor && Math.abs(cd.m) > 1e-3 ? cor.cd / cd.m : 1; // the drag correction is a scale factor too
  G.flapShare = flapOn() && wf.length > 2 ? { cl: kL * wf.reduce((a, p) => a + p.clF, 0) / wf.length, cd: kD * wf.reduce((a, p) => a + p.cdF, 0) / wf.length, clT: kL * cl.m, cor: !!cor } : null;
  const CL = cor ? cor.cl : cl.m, CD = cor ? cor.cd : cd.m, CM = cor ? cor.cm : cm.m;
  $('#vCL').textContent = fmtN(CL); $('#vCD').textContent = fmtN(CD); $('#vCM').textContent = fmtN(CM);
  $('#vLD').textContent = Math.abs(CD) > 1e-4 ? fmtN(CL / CD, 1) : '–';
  const rms = (s) => s.sd > 0.004 ? ` · rms ${fmtN(s.sd, 3)}` : '';
  $('#dCL').textContent = cor ? `raw ${fmtN(cl.m)}${rms(cl)}` : `uncorrected${rms(cl)}`;
  $('#dCD').textContent = cor ? `raw ${fmtN(cd.m)}${rms(cd)}` : `uncorrected${rms(cd)}`;
  $('#dLD').textContent = cor ? `corrected α ${cor.aoa.toFixed(2)}°` : 'slip walls, uncorrected';
  $('#dCM').textContent = cor ? `raw ${fmtN(cm.m)} · nose-up +` : 'nose-up positive';
  updateState(cl);
}
function updateState(cl, pc) {
  const fp = flowParams();
  const uns = cl && cl.sd > Math.max(0.012, 0.035 * Math.abs(cl.m));
  const pill = !cl ? `<span class="pill">${pc != null ? 'Settling' : 'Starting'}</span>` : uns ? '<span class="pill unsteady">Unsteady wake</span>' : '<span class="pill steady">Steady</span>';
  const eng = MODE === 'gpu' ? 'GPU' : 'CPU';
  const re = fp.capped
    ? `<span class="pill unsteady">Re capped</span><span>Rig Re <b>${Math.round(fp.ReReal).toLocaleString()}</b>, simulating <b>${fp.Re.toLocaleString()}</b> (grid limit)</span>`
    : `<span>Re <b>${fp.Re.toLocaleString()}</b></span>`;
  const sh = G && G.play ? G.play.shed : null;
  const shed = G && G.play ? (sh ? `<span>Shedding St <b>${sh.St.toFixed(2)} ± ${sh.dSt.toFixed(2)}</b> (f c/U∞, ${sh.f.toFixed(1)} Hz)</span>` : '<span>No dominant shedding frequency</span>') : '';
  const F = G && G.flapShare && cl ? G.flapShare : null;
  const fs = F ? `<span>Flap carries ${Math.abs(F.clT) > 0.05 ? `<b>${Math.round(100 * F.cl / F.clT)}%</b> of the lift, ` : ''}C<sub>L</sub> <b>${fmtN(F.cl)}</b>, C<sub>D</sub> <b>${fmtN(F.cd)}</b>${F.cor ? ', wall-corrected like the totals' : ', uncorrected'}</span>` : '';
  const fl = flapLabel() ? `<span>${flapLabel().replace(/^./, ch => ch.toUpperCase())}</span>` : '';
  $('#stateLine').innerHTML = `${pill}<span>${profileName()} at <b>${S.aoa.toFixed(1)}°</b></span>${fl}${fs}${shed}${re}<span>Mach <b>${(S.speed / 343).toFixed(3)}</b></span><span>Blockage c/h <b>${(1 / S.H).toFixed(2)}</b></span><span>${eng} grid <b>${G.F.nx}×${G.F.ny}</b> (${G.F.Nc}/chord)</span>`;
}

// ---------- controls ----------
const CTRL = [
  { sec: 'Aerofoil', tag: () => profileName(), items: [
    { id: 'preset', type: 'select', label: 'Profile', options: [['0012', 'NACA 0012 (symmetric)'], ['2412', 'NACA 2412 (light camber)'], ['4412', 'NACA 4412 (high lift)'], ['6409', 'NACA 6409 (thin, high camber)'], ['0021', 'NACA 0021 (thick)'], ['plate', 'Flat plate'], ['cyl', 'Circular cylinder (0.5c)'], ['custom', 'Custom NACA 4-digit']] },
    { id: 'aoa', type: 'range', label: 'Angle of attack', min: -15, max: 25, step: 0.5, fmt: v => `${(+v).toFixed(1)}°` },
    { id: 'm', type: 'range', label: 'Maximum camber', min: 0, max: 9, step: 1, fmt: v => `${v}% c` },
    { id: 'p', type: 'range', label: 'Camber position', min: 1, max: 9, step: 1, fmt: v => `${v * 10}% c` },
    { id: 't', type: 'range', label: 'Thickness', min: 2, max: 30, step: 1, fmt: v => `${v}% c` },
  ] },
  { sec: 'Flap', tag: () => !flapAllowed() ? 'not on this section' : flapOn() ? `${FLAP_NAMES[S.flap].replace(' flap', '')} · ${(+S.flapDef).toFixed(0)}°` : 'none', items: [
    { id: 'flap', type: 'select', label: 'Flap type', options: [['none', 'None (clean wing)'], ['plain', 'Plain flap'], ['split', 'Split flap'], ['slotted', 'Single-slotted flap'], ['fowler', 'Fowler flap']] },
    { id: 'flapDef', type: 'range', label: 'Flap angle', min: 0, max: 40, step: 1, fmt: v => `${(+v).toFixed(0)}° down`, when: () => flapOn() },
    { id: 'flapC', type: 'range', label: 'Flap chord', min: 0.2, max: 0.35, step: 0.01, fmt: v => `${Math.round(v * 100)}% c`, when: () => flapOn() },
    { id: 'flapPreset', type: 'seg', label: 'Presets', options: [['cruise', 'Cruise'], ['takeoff', 'Take-off'], ['landing', 'Landing']], when: () => flapOn() },
  ] },
  { sec: 'Flow', tag: () => `Re ${Math.round(S.speed * S.chord / NU_AIR).toLocaleString()}`, items: [
    { id: 'speed', type: 'range', label: 'Air speed', min: 0, max: 1000, step: 1, map: speedFromPos, unmap: posFromSpeed, fmt: v => `${v} m/s` },
    { id: 'chord', type: 'range', label: 'Model chord', min: 0.02, max: 0.5, step: 0.005, fmt: v => `${Math.round(v * 1000)} mm` },
    { id: 'turb', type: 'range', label: 'Free-stream turbulence', min: 0, max: 5, step: 0.05, fmt: v => `${(+v).toFixed(2)}%` },
  ] },
  { sec: 'Test section', tag: () => `${(S.L * S.chord).toFixed(2)} × ${(S.H * S.chord).toFixed(2)} m`, items: [
    { id: 'L', type: 'range', label: 'Length', min: 3, max: 8, step: 0.5, fmt: v => `${(+v).toFixed(1)} c`, rebuild: true },
    { id: 'H', type: 'range', label: 'Height', min: 1.5, max: 6, step: 0.25, fmt: v => `${(+v).toFixed(2)} c`, rebuild: true },
    { id: 'walls', type: 'select', label: 'Walls', options: [['solid', 'Closed section (no-slip walls)'], ['open', 'Slip walls (low blockage)']] },
    { id: 'res', type: 'select', label: 'Resolution', options: [['coarse', 'Coarse · 50 per chord (slow devices)'], ['std', 'Standard · 100 per chord'], ['fine', 'Fine · 150 per chord'], ['ultra', 'Ultra · 200 per chord (fast GPUs)'], ['extreme', 'Extreme · 300 per chord (high-end GPUs)']], rebuild: true },
  ] },
  { sec: 'Display', items: [
    { id: 'field', type: 'seg', label: 'Field', options: [['speed', 'Speed'], ['vort', 'Vorticity'], ['press', 'Pressure'], ['smoke', 'Smoke'], ['schl', 'Schlieren'], ['piv', 'PIV']] },
    { id: 'cmap', type: 'select', label: 'Colour map', options: [['auto', 'Best for field'], ['viridis', 'Viridis'], ['turbo', 'Turbo'], ['coolwarm', 'Cool–warm (diverging)'], ['grey', 'Greyscale']] },
    { id: 'pivView', type: 'select', label: 'PIV view', when: () => S.field === 'piv', options: [['proc', 'Processed vector field'], ['raw', 'Raw camera image']] },
    { id: 'smokeSrc', type: 'select', label: 'Smoke release', when: () => S.field === 'smoke', options: [['rake', 'Rake at the contraction exit'], ['wire', 'Smoke wire ahead of the model']] },
    { id: 'overlays', type: 'checks', label: 'Overlays', options: [['stream', 'Streamlines'], ['contours', 'Contours'], ['vectors', 'Vectors']] },
    { id: 'contrast', type: 'range', label: 'Contrast', min: 0.4, max: 3, step: 0.1, fmt: v => `×${(+v).toFixed(1)}` },
  ] },
];
const panel = $('#controls');
function buildControls() {
  for (const sec of CTRL) {
    const d = document.createElement('section'); d.className = 'sec';
    d.innerHTML = `<h2>${sec.sec}<small data-tag></small></h2>`;
    sec.el = d.querySelector('[data-tag]');
    for (const c of sec.items) {
      const w = document.createElement('div'); w.className = 'ctl';
      const id = 'c_' + c.id;
      if (c.type === 'range') {
        const map = c.map || ((v) => v), unmap = c.unmap || ((v) => v);
        w.innerHTML = `<label for="${id}">${c.label}</label><output id="${id}_o"></output><input type="range" id="${id}" min="${c.min}" max="${c.max}" step="${c.step}" value="${unmap(S[c.id])}">`;
        const inp = w.querySelector('input'), out = w.querySelector('output');
        const show = () => { out.textContent = c.fmt(S[c.id]); };
        show();
        inp.addEventListener('input', () => { S[c.id] = map(+inp.value); show(); onChange(c, false); });
        inp.addEventListener('change', () => onChange(c, true));
        c.sync = () => { inp.value = unmap(S[c.id]); show(); };
      } else if (c.type === 'select') {
        w.innerHTML = `<label for="${id}">${c.label}</label><span></span><select id="${id}">${c.options.map(([v, t]) => `<option value="${v}">${t}</option>`).join('')}</select>`;
        const sel = w.querySelector('select'); sel.value = S[c.id];
        sel.addEventListener('change', () => { S[c.id] = sel.value; onChange(c, true); });
        c.sync = () => { sel.value = S[c.id]; };
      } else if (c.type === 'seg') {
        w.innerHTML = `<label>${c.label}</label><span></span><div class="seg" role="group" aria-label="${c.label}">${c.options.map(([v, t]) => `<button type="button" id="${id}_${v}" data-v="${v}" aria-pressed="${S[c.id] === v}">${t}</button>`).join('')}</div>`;
        w.querySelectorAll('button').forEach(b => b.addEventListener('click', () => { S[c.id] = b.dataset.v; c.sync(); onChange(c, true); }));
        c.sync = () => w.querySelectorAll('button').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.v === S[c.id])));
      } else if (c.type === 'checks') {
        w.innerHTML = `<label>${c.label}</label><span></span><div class="checks">${c.options.map(([v, t]) => `<label><input type="checkbox" id="${id}_${v}" data-v="${v}" ${S[v] ? 'checked' : ''}>${t}</label>`).join('')}</div>`;
        w.querySelectorAll('input').forEach(b => b.addEventListener('change', () => { S[b.dataset.v] = b.checked; dirty = true; }));
        c.sync = () => w.querySelectorAll('input').forEach(b => { b.checked = !!S[b.dataset.v]; });
      }
      c.wrap = w;
      d.appendChild(w);
    }
    panel.appendChild(d);
  }
  syncTags();
}
function syncTags() {
  for (const s of CTRL) { if (s.tag) s.el.textContent = s.tag(); for (const c of s.items) if (c.when && c.wrap) c.wrap.hidden = !c.when(); }
  syncFlapUI();
}
// ---------- flaps ----------
const PRESET_KEYS = ['cruise', 'takeoff', 'landing'];
function flapPresetOf() { const T = FLAP_TYPES[S.flap]; if (!T) return ''; const i = T.presets.indexOf(+S.flapDef); return i >= 0 ? PRESET_KEYS[i] : ''; }
function syncFlapUI() {
  const sel = $('#c_flap'), inp = $('#c_flapDef'), note = $('#flapNote'); if (!sel || !inp) return;
  sel.disabled = !flapAllowed() || !!(G && (G.rec || G.play));
  const T = FLAP_TYPES[S.flap];
  if (T) { inp.max = String(T.max); if (S.flapDef > T.max) S.flapDef = T.max; inp.value = String(S.flapDef); }
  S.flapPreset = flapPresetOf(); const pc = ctlById('flapPreset'); if (pc && pc.sync) pc.sync();
  if (!note) return;
  if (!flapAllowed()) { note.textContent = 'Flaps fit NACA sections at least 6% thick, not the cylinder or flat plate.'; return; }
  if (!flapOn()) { note.textContent = 'A hinged rear part of the wing: lift and drag are split between the wing and the flap.'; return; }
  const Nc = G ? G.F.Nc : 100, thin = S.flap === 'slotted' || S.flap === 'fowler' ? 0.018 : S.flap === 'split' ? 0.011 : 0;
  const cells = thin * Nc;
  if (S.flap === 'split') note.textContent = `The plate is ${(0.011 * Nc).toFixed(1)} cells thick on this grid: it still blocks the flow, and Fine or finer resolves the flow round its tip better.`;
  else note.textContent = thin ? `The slot is about ${cells.toFixed(1)} cells across on this grid. ${cells < 1.5 ? 'That is too few: the slot is effectively closed, so this behaves like a plain flap. Choose Fine or finer under Test section.' : cells < 2.6 ? 'Only a trickle gets through; choose Fine or finer under Test section to resolve it.' : 'In a check, lift with 2.7 cells came within about 5% of a run with 4.5 cells.'}` : 'Angles are trailing edge down, about a hinge on the camber line.';
}
function flapChanged(c) {
  const T = FLAP_TYPES[S.flap];
  if (c.id === 'flap' && T && !(S.flapDef > 0)) S.flapDef = T.presets[1]; // a newly chosen flap starts at its take-off setting
  if (c.id === 'flapPreset' && T) S.flapDef = T.presets[PRESET_KEYS.indexOf(S.flapPreset)] || 0;
  if (T && S.flapDef > T.max) S.flapDef = T.max;
  ctlById('flapDef').sync();
  if (c.id === 'flapDef' || c.id === 'flapC') throttleBody(); else sendBody();
  resetHistory();
  if (c.id === 'flap' && (S.flap === 'slotted' || S.flap === 'fowler') && G.F.Nc * 0.018 < 2.6) toast(`On this grid the slot is only about ${(G.F.Nc * 0.018).toFixed(1)} cells across, so ${G.F.Nc * 0.018 < 1.5 ? 'it is effectively closed' : 'little air gets through it'}. Choose Fine or finer under Test section to resolve it.`);
}
const ctlById = (id) => CTRL.flatMap(s => s.items).find(c => c.id === id);
let bodyTimer = 0;
function onChange(c, committed) {
  const id = c.id;
  if (id === 'preset') {
    if (PRESETS[S.preset]) { [S.m, S.p, S.t] = PRESETS[S.preset]; ['m', 'p', 't'].forEach(k => ctlById(k).sync()); }
    sendBody(); resetHistory();
  } else if (id === 'm' || id === 'p' || id === 't') {
    const match = Object.entries(PRESETS).find(([, v]) => v[0] === S.m && v[1] === S.p && v[2] === S.t);
    S.preset = match ? match[0] : 'custom'; ctlById('preset').sync();
    throttleBody(); resetHistory();
  } else if (id === 'aoa') { throttleBody(); resetHistory(); }
  else if (id === 'flap' || id === 'flapDef' || id === 'flapC' || id === 'flapPreset') flapChanged(c);
  else if (id === 'speed' || id === 'chord' || id === 'turb') { sendFlow(); if (id !== 'turb') resetHistory(); }
  else if (c.rebuild) { if (committed) requestGrid(); }
  else if (id === 'walls') { if (MODE === 'gpu') gpu.walls = S.walls; else engine.post({ type: 'walls', walls: S.walls }); $('#stage').dataset.walls = S.walls; resetHistory(); }
  else if (id === 'field') { if (S.cmap !== 'auto') { S.cmap = 'auto'; ctlById('cmap').sync(); } }
  else if (id === 'smokeSrc') { if (MODE === 'gpu') gpu.clearDye(); G.dye.fill(0); }
  syncTags(); updateState(null); dirty = true;
}
// ---------- grid changes: memory check and time estimate ----------
function requestGrid() {
  const prev = G.cfg || { res: S.res, L: S.L, H: S.H };
  if (MODE !== 'gpu') { initGrid(); return; }
  const p = gridPlan(S.res), ck = planCheck(p);
  const ask = (S.res === 'extreme' && prev.res !== 'extreme') || !(ck.level === 'ok' || ck.level === 'close');
  if (!ask) { initGrid(); return; }
  const revert = () => { S.res = prev.res; S.L = prev.L; S.H = prev.H; for (const id of ['res', 'L', 'H']) ctlById(id).sync(); syncTags(); updateResUI(); };
  const build = () => { if (p.N > 1e6) { toast(`Building the ${RES_NAME[p.res]} grid…`); setTimeout(initGrid, 80); } else initGrid(); }; // let the message show before the page is busy
  showGridDialog(p, ck, prev, build, revert);
}
function showGridDialog(p, ck, prev, go, cancel) {
  const B = ck.B, t = flowSecs(p), kNow = G.F.k, finer = p.k / kNow;
  const dev = B.reported ? `reports ${B.gb} GB of memory, so the page plans on at most about ${fmtGB(B.lim)}` : `doesn't report its memory; assuming a ${B.mobile ? 'phone or tablet' : 'computer'} with about ${B.gb} GB, so at most about ${fmtGB(B.lim)}`;
  const verdict = {
    ok: ['good', 'Should fit comfortably.'],
    close: ['good', 'Should fit, with little to spare while recording.'],
    tight: ['warn', 'The live view should fit, but a recording may run out of memory on this device.'],
    over: ['bad', `Probably too large for this device. If the graphics memory runs out the tunnel goes back to the previous grid${B.mobile ? ', and on some phones the browser reloads the page' : ''}.`],
    impossible: ['bad', `This graphics processor can't hold a grid ${Math.max(p.nx, p.ny).toLocaleString()} cells across (its limit is ${B.maxDim.toLocaleString()}). Shorten the test section or choose a coarser grid.`],
  }[ck.level];
  const sameLevel = p.res === prev.res;
  $('#gdTitle').textContent = sameLevel ? `Resize the ${RES_NAME[p.res]} grid?` : `Switch to the ${RES_NAME[p.res]} grid?`;
  $('#gdBody').innerHTML = `<p>${p.nx.toLocaleString()} × ${p.ny.toLocaleString()} cells (${fmtCells(p.N)}), ${p.Nc} per chord: Reynolds numbers up to ${p.reMax.toLocaleString()} without capping${!sameLevel && finer > 1 ? `, and ${+finer.toFixed(2)}× finer than now in each direction` : ''}.</p>
    <dl class="gd">
      <dt>Memory</dt><dd>≈ ${fmtGB(p.need)} for the live view, ${fmtGB(p.needRec)} while recording</dd>
      <dt>This device</dt><dd>${dev}</dd>
      <dt>Time</dt><dd>${isFinite(t) ? `≈ ${fmtDur(t)} per flow-through, estimated from this device's measured speed, so a ${S.recFT}-flow-through recording takes ≈ ${fmtDur((S.recFT + 2) * t)} including settling. Each flow-through takes about ${Math.max(1, Math.round(p.N * p.ft / (G.F.nx * G.F.ny * FT())))}× as long as on the current grid.` : 'Not measured yet: let the tunnel run for a few seconds and this estimate appears.'}</dd>
    </dl>
    <p class="gd-verdict ${verdict[0]}">${verdict[1]}</p>${isFinite(t) && t > 600 ? `<p class="gd-verdict warn">Slow on this device: ${t > 3600 ? 'a single flow-through takes hours' : 'each flow-through takes minutes'}, so the live view will crawl. Practical only for unattended recordings, ideally on a fast graphics card.</p>` : ''}`;
  const d = $('#gridDialog'), goB = $('#gdGo'), noB = $('#gdCancel');
  goB.hidden = ck.level === 'impossible'; goB.textContent = ck.level === 'over' ? 'Try anyway' : sameLevel ? 'Resize' : `Use ${RES_NAME[p.res]}`;
  d.hidden = false; (goB.hidden ? noB : goB).focus();
  const close = (ok) => { d.hidden = true; goB.onclick = noB.onclick = null; d.onkeydown = null; if (ok) go(); else cancel(); };
  goB.onclick = () => close(true); noB.onclick = () => close(false);
  d.onkeydown = (e) => { if (e.key === 'Escape') close(false); };
}
function buildFlapUI() {
  ctlById('flap').wrap.insertAdjacentHTML('beforeend', '<p class="note" id="flapNote"></p>');
  syncFlapUI();
}
function buildResUI() {
  const c = ctlById('res');
  c.wrap.insertAdjacentHTML('beforeend', `<p class="note" id="resNote"></p>
    <details class="lvls"><summary>Compare grid levels</summary><table class="lvl"><thead><tr><th>Grid</th><th>Per chord</th><th>Re limit</th><th>Memory</th><th>Flow-through</th></tr></thead><tbody id="resRows"></tbody></table>
    <p class="note">Memory is the peak while recording, graphics and processor together. Times are estimated from the speed measured on this device at the current grid, scaled by cells × steps (about 8× for each doubling of resolution); large grids often run a little faster than that.</p></details>`);
  if (MODE === 'cpu') { const o = c.wrap.querySelector('option[value="extreme"]'); if (o) o.disabled = true; }
  setInterval(() => { if (G && !G.rec && !G.play) updateResUI(); }, 3000);
}
function updateResUI() {
  const el = $('#resNote'); if (!el || !G) return;
  if (MODE === 'cpu') { el.textContent = `CPU solver: ${G.F.nx} × ${G.F.ny} cells, ${G.F.Nc} per chord. The Extreme grid needs the GPU solver.`; $('#resRows').innerHTML = ''; return; }
  const p = gridPlan(G.cfg ? G.cfg.res : S.res), t = flowSecs(p);
  el.textContent = `${p.nx.toLocaleString()} × ${p.ny.toLocaleString()} cells (${fmtCells(p.N)}), Re up to ${p.reMax.toLocaleString()}. Memory ≈ ${fmtGB(p.need)}, ${fmtGB(p.needRec)} while recording. ${isFinite(t) ? `A flow-through takes ≈ ${fmtDur(t)} here.` : 'Measuring this device’s speed…'}`;
  $('#resRows').innerHTML = RES_ORDER.map(r => {
    const q = gridPlan(r), ck = planCheck(q), tt = flowSecs(q), cur = r === p.res;
    const cls = ck.level === 'over' || ck.level === 'impossible' ? 'bad' : ck.level === 'tight' ? 'warn' : '';
    return `<tr${cur ? ' class="cur"' : ''}><td>${RES_NAME[r]}</td><td>${q.Nc}</td><td>${q.reMax.toLocaleString()}</td><td class="${cls}">${ck.level === 'impossible' ? 'too wide' : fmtGB(q.needRec)}</td><td>${isFinite(tt) ? fmtDur(tt) : '–'}</td></tr>`;
  }).join('');
}
function throttleBody() { if (bodyTimer) return; bodyTimer = setTimeout(() => { bodyTimer = 0; sendBody(); }, 60); }
function resetHistory() { G.hist = []; G.since = 0; G.cp = null; G.stag = null; G.arc = null; }

// ---------- interaction ----------
let drag = null;
function hitBody(x, y) {
  let x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9; for (const p of G.poly) { x0 = Math.min(x0, p.x); x1 = Math.max(x1, p.x); y0 = Math.min(y0, p.y); y1 = Math.max(y1, p.y); }
  const m = 0.15 * G.Nc; return x > x0 - m && x < x1 + m && y > y0 - m && y < y1 + m;
}
view.addEventListener('pointerdown', (e) => {
  const r = view.getBoundingClientRect();
  const [x, y] = toLat((e.clientX - r.left) * dpr, (e.clientY - r.top) * dpr);
  const locked = !!(G.rec || G.play);
  if (S.probeTap && !locked) {
    const X = (x + 0.5) / G.Nc, Y = (y + 0.5) / G.Nc, k = probeNear(X, Y);
    if (k >= 0) S.probes.splice(k, 1); else if (!addProbe(X, Y)) { toast('Probes go in the flow, not inside the model.'); return; }
    probesChanged(); return;
  }
  if (!locked && hitBody(x, y)) { drag = { x, y, px: S.pos.x, py: S.pos.y }; view.classList.add('grab'); view.setPointerCapture(e.pointerId); }
  else { S.probe = { x, y }; dirty = true; }
});
view.addEventListener('pointermove', (e) => {
  if (!drag) return;
  const r = view.getBoundingClientRect();
  const [x, y] = toLat((e.clientX - r.left) * dpr, (e.clientY - r.top) * dpr);
  S.pos.x = drag.px + (x - drag.x) / G.Nc; S.pos.y = drag.py + (y - drag.y) / G.Nc; clampPos();
  throttleBody(); resetHistory();
});
const endDrag = () => { if (drag) { drag = null; view.classList.remove('grab'); sendBody(); } };
view.addEventListener('pointerup', endDrag); view.addEventListener('pointercancel', endDrag);

$('#btnRun').addEventListener('click', () => {
  S.running = !S.running; if (MODE === 'cpu') engine.post({ type: 'run', on: S.running });
  syncRunButton();
});
function unstable() {
  G.reScale = (G.reScale || 1) * 0.7;
  sendFlow(); restartFlow();
  toast(`The flow diverged at this setting. Restarted with the Reynolds number limited to ${flowParams().reMax.toLocaleString()} for stability.`);
}
function restartFlow() {
  if (MODE === 'gpu') gpu.reset(); else engine.post({ type: 'reset' });
  G.dye.fill(0); initStreaks(); initPIV(); resetHistory(); G.firstRun = true; G.t = 0;
}
$('#btnReset').addEventListener('click', restartFlow);
$('#btnCentre').addEventListener('click', () => { S.pos = { x: Math.min(2.25, S.L * 0.375), y: S.H / 2 }; sendBody(); resetHistory(); });
document.addEventListener('visibilitychange', () => { if (MODE === 'cpu' && engine && !G.play) engine.post({ type: 'run', on: S.running && !document.hidden }); });
let toastT = 0;
function toast(msg) { const t = $('#toast'); t.textContent = msg; t.hidden = false; clearTimeout(toastT); toastT = setTimeout(() => { t.hidden = true; }, 4500); }
let rz = 0; window.addEventListener('resize', () => { clearTimeout(rz); rz = setTimeout(() => { if (!exporting) sizeCanvas(); }, 120); });
matchMedia('(prefers-color-scheme: dark)').addEventListener?.('change', () => { dirty = true; });

// ---------- GPU frame ----------
let autoChecked = false, benchSteps = 0, benchMs = 0;
function gpuFrame() {
  let n = 0;
  const now0 = performance.now(), frameGap = G.lastFrameAt ? now0 - G.lastFrameAt : 0; G.lastFrameAt = now0;
  if (S.running && !document.hidden) {
    const t0 = performance.now();
    n = Math.round(G.spf);
    liveStep(n);
    const [fx, fy, mz, fxF, fyF, mzF] = gpu.forces(G.rhoRef);
    const dt = performance.now() - t0;
    // keep simulation work near 14 ms a frame
    if (!G.fixedSpf) G.spf = Math.max(G.F.k >= 6 ? 1 : 2, Math.min(160, G.spf * Math.min(1.25, Math.max(0.7, 14 / Math.max(1, dt)))));
    if (G.frame > 40) noteThroughput(G.F.nx * G.F.ny, n, dt);
    G.simMs = dt;
    // first run on a slow GPU: if a flow-through would take longer than about 40 s, drop to the coarse grid once
    if (!G.fixedSpf && !autoChecked && !window.__wtNoAuto && S.res === 'std') {
      benchSteps += n; benchMs += Math.min(frameGap, 500);
      if (benchMs > 4000) {
        autoChecked = true;
        const sps = benchSteps / (benchMs / 1000);
        if (FT() / sps > 40) { S.res = 'coarse'; ctlById('res').sync(); initGrid(); toast('This device simulates slowly, so the grid was set to Coarse. Choose a finer grid under Test section.'); return; }
      }
    }
    G.t = gpu.t;
    pushForces(fx, fy, mz, n, fxF, fyF, mzF);
    if (frameGap > 0) { const s = n / (frameGap / 1000); G.sps = G.sps ? 0.9 * G.sps + 0.1 * s : s; }
  }
  const d = gpu.readDown();
  const N = G.N; let bad = false;
  for (let i = 0, j = 0; i < N; i++, j += 4) { G.rho[i] = d[j]; G.ux[i] = d[j + 1]; G.uy[i] = d[j + 2]; }
  for (let k = 0; k < 16; k++) { const r = G.rho[(Math.random() * N) | 0]; if (!(r > 0.5 && r < 1.6)) bad = true; }
  if (bad) { unstable(); return; }
  updateRhoRef(n);
  if (n > 0) {
    gpu.fields(0, G.rhoRef, G.F.Nc, G.uRef); // velocity texture for the passive scalars
    const R = smokeRelease(), k = G.F.k; // readback cells -> fine cells
    gpu.advect(n, { rake: { x0: R.x0 * k, x1: R.x1 * k, y0: R.y0 * k, y1: R.y1 * k, sp: R.sp * k, off: R.off * k, w: R.w * k }, kHeat: Math.max(gpu.nu / 0.71, 0.008), kSmoke: gpu.nu / 4 });
  }
  try {
    if (n > 0) advectPIV(n);
    if (G.frame % 3 === 0) updateCp(n * 3);
    if ((S.field === 'press' || S.probe) && (G.frame % 2 === 0 || !G.pmgValid)) updatePoisson(1);
  } catch (err) { console.error(err); }
  dirty = true;
}

/*FEATURES*/

// ---------- main loop ----------
let lastChart = 0;
function frame() {
  G.frame++;
  if (ctxLost || exporting) { requestAnimationFrame(frame); return; }
  if (G.play) { try { playbackFrame(); } catch (err) { console.error(err); } }
  else if (MODE === 'gpu') { try { if (G.rec) recordFrameGPU(); else gpuFrame(); } catch (err) { console.error(err); if (G.rec) abortRecording('The recording stopped because of a graphics error.'); else fallbackToCPU(String(err)); } }
  else if (!waiting && engine) { waiting = true; engine.post({ type: 'get' }); }
  if (dirty && G) {
    dirty = false;
    try { render(); } catch (err) { console.error(err); }
  }
  const now = performance.now();
  if (G && now - lastChart > (G.rec ? 1000 : 250)) { lastChart = now; try { drawForceChart(); drawCpChart(); drawProbeChart(); updateReadouts(); } catch (err) { console.error(err); } }
  requestAnimationFrame(frame);
}
function fallbackToCPU(reason) {
  if (MODE === 'cpu') return;
  MODE = 'cpu'; glc.hidden = true; console.warn('GPU solver unavailable, using CPU:', reason);
  const o = ctlById('res').wrap.querySelector('option[value="extreme"]'); if (o) o.disabled = true;
  if (S.res === 'extreme') { S.res = 'ultra'; ctlById('res').sync(); }
  ctxLost = false;
  if (G && G.play) closeReview(); else initGrid();
}
// ---------- graphics context loss (for example when the graphics memory runs out) ----------
let ctxLost = false, lostDown = false, exporting = false;
glc.addEventListener('webglcontextlost', (e) => {
  e.preventDefault(); if (MODE !== 'gpu') return;
  ctxLost = true; if (expAbort) expAbort.abort();
  setTimeout(() => { if (ctxLost && MODE === 'gpu') fallbackToCPU('the graphics context was not restored'); }, 12000);
  if (G && (S.res === 'extreme' || performance.now() - (G.builtAt || 0) < 20000)) lostDown = true;
  if (G && G.rec) { G.rec = null; setLocked(false); wake(false); renderRecordUI(); toast('The graphics processor was reset, probably because it ran out of memory, so the recording paused. Resume it from its last checkpoint under Record & review.'); }
  else toast('The graphics processor was reset, probably because it ran out of memory. Restarting the tunnel…');
});
glc.addEventListener('webglcontextrestored', function rebuild() {
  if (MODE !== 'gpu') return;
  if (exporting) { setTimeout(rebuild, 200); return; } // let a cancelled export unwind first
  try {
    gpu = new GPUTunnel(glc); ctxLost = false;
    if (lostDown) { lostDown = false; const i = RES_ORDER.indexOf(S.res); if (i > 0) { S.res = RES_ORDER[i - 1]; ctlById('res').sync(); syncTags(); toast(`Restarted on the ${RES_NAME[S.res]} grid, which needs less graphics memory.`); } }
    if (G && G.play) closeReview(); else initGrid();
    checkResumable(); refreshRuns();
  } catch (err) { console.error(err); fallbackToCPU(String(err)); }
});
buildControls();
if (MODE === 'gpu') { try { gpu = new GPUTunnel(glc); } catch (err) { console.warn(err); MODE = 'cpu'; } }
buildResUI();
buildFlapUI();
buildFeatureUI();
if (MODE === 'cpu') glc.hidden = true;
try { initGrid(); } catch (err) { console.error(err); fallbackToCPU(String(err)); }
updateState(null);
requestAnimationFrame(frame);
window.__wt = { S, get G() { return G; }, get MODE() { return MODE; }, get gpu() { return gpu; }, Store, checkpoint, openReview, closeReview, statsPayload, gridPlan, planCheck, get CPS() { return CPS; },
  loadState(f, t) { gpu.loadState(f, t); G.t = t; G.since = FT(); G.firstRun = false; G.hist = []; G.cp = null; G.stag = null; G.arc = null; G.refInit = false; } };
})();
