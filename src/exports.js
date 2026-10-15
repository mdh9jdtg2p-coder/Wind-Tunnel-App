// ---------- image and video export ----------
// Exports are laid out as if the stage were EXP_REF CSS pixels wide and then scaled, so a picture looks the same
// whichever device made it. Each picture is the tunnel view between its walls with a caption bar underneath.
const EXP_REF = 960;
const EXP_PAL = { bg: '#0b0f13', band: '#141a20', ink: '#e6ecf0', muted: '#95a2ad', rule: '#26303a', hatch: '#59656f' };
const sleep = (ms) => new Promise(r => setTimeout(r, ms));
let expAbort = null;

function expLayout(W, caption) {
  W = 2 * Math.round(W / 2);
  const u = W / EXP_REF, wall = Math.max(2, Math.round(9 * u)), Hf = Math.round(W * G.ny / G.nx), cap = caption ? Math.round(76 * u) : 0;
  let H = Hf + 2 * wall + cap; if (H % 2) H++; // even dimensions for the video encoders
  return { W, H, u, wall, Hf, cap };
}
// largest layout no wider than W that the graphics processor, the canvas and (for video) the encoders accept
function expFit(W, caption, video) {
  const lim = deviceBudget().maxDim, maxW = video ? 3840 : Math.min(lim, 8192), maxH = video ? 2160 : Math.min(lim, 8192), area = video ? 3840 * 2160 : 16.7e6;
  let w = Math.min(W, maxW);
  for (let k = 0; k < 60; k++) { const L = expLayout(w, caption); if (L.H <= maxH && L.Hf <= lim && L.W * L.H <= area) return L; w = Math.floor(w * 0.95); }
  return expLayout(640, caption);
}
let expSaved = null;
function expBegin(L, o) { // draw the next renders at export size; the stage keeps its on-screen size, so it previews the export
  exporting = true;
  expSaved = { probe: S.probe, probes: S.probes };
  S.probe = null; if (!o.probes) S.probes = [];
  view.width = L.W; view.height = L.Hf; if (MODE === 'gpu') { glc.width = L.W; glc.height = L.Hf; }
  scale = L.W / G.nx; dpr = L.u;
}
function expEnd() {
  if (!expSaved) return;
  S.probe = expSaved.probe; S.probes = expSaved.probes; expSaved = null;
  exporting = false; sizeCanvas(); dirty = true;
}
function wallBand(c, y, W, h, u) {
  c.fillStyle = EXP_PAL.band; c.fillRect(0, y, W, h);
  c.save(); c.beginPath(); c.rect(0, y, W, h); c.clip();
  c.strokeStyle = EXP_PAL.hatch; c.lineWidth = Math.max(1, 2 * u);
  if (S.walls === 'solid') { const sp = 7 * u; c.beginPath(); for (let x = -h; x < W + h; x += sp) { c.moveTo(x, y + h); c.lineTo(x + h, y); } c.stroke(); }
  else { c.setLineDash([10 * u, 8 * u]); c.beginPath(); c.moveTo(0, y + h / 2); c.lineTo(W, y + h / 2); c.stroke(); }
  c.restore();
}
function fitText(c, text, x, y, maxW, align) { // shrink a little, then truncate, so long captions never overrun
  const f0 = c.font, m = /(\d+(?:\.\d+)?)px/.exec(f0), px = m ? +m[1] : 12;
  let w = c.measureText(text).width;
  if (w > maxW && m) { const s = Math.max(0.78, 0.97 * maxW / w); c.font = f0.replace(m[0], `${(px * s).toFixed(2)}px`); w = c.measureText(text).width; }
  let t = text; while (w > maxW && t.length > 4) { t = t.slice(0, -2); w = c.measureText(t + '…').width; if (w <= maxW) t += '…'; }
  c.textAlign = align || 'left'; c.fillText(t, x, y); c.font = f0;
}
function captionInfo(v) { // v: { i, video } for video frames
  const P = G.play, M = P && P.M;
  const Re = M ? M.Re : flowParams().Re, reRig = Math.round(S.speed * S.chord / NU_AIR);
  const title = `${profileName()} · α ${S.aoa.toFixed(1)}° · Re ${Re.toLocaleString()}${reRig > Re * 1.02 ? ` (rig ${reRig.toLocaleString()}; viscosity raised ${(reRig / Re).toFixed(1)}× for the grid)` : ''}`;
  const Nc = M ? M.grid.F.Nc : G.F.Nc;
  const line2 = `2-D lattice-Boltzmann simulation · U∞ ${S.speed} m/s · c ${Math.round(S.chord * 1000)} mm · ${S.walls === 'solid' ? 'closed' : 'slip-wall'} section ${+S.L.toFixed(2)} × ${+S.H.toFixed(2)} c · ${Nc} cells/c · Tu ${(+S.turb.toFixed(2))}%`;
  let line3;
  if (!P) line3 = `t = ${(flowTime(G.t) * 1000).toFixed(1)} ms after the flow started (${(flowTime(G.t) * S.speed / S.chord).toFixed(1)} c/U∞)`;
  else if (P.view === 'inst') {
    // as many decimals as the step between pictures needs, the same for both times
    const x = v ? v.x : Math.max(0, P.idx), stepMs = (v && v.video ? v.video.speed / v.video.fps : P.frameDt) * 1000, dp = Math.min(3, Math.max(1, Math.ceil(-Math.log10(stepMs) + 0.3)));
    const ms = x * P.frameDt * 1000, tot = M.steps.total * M.dtStep * 1000, i = Math.floor(x + 1e-6), between = x - i > 1e-3 && i + 1 < M.frames;
    line3 = `t = ${ms.toFixed(dp)} ms of ${tot.toFixed(dp)} ms recorded (${(x * P.frameDt * M.speed / M.chord).toFixed(2)} c/U∞) · ${between ? `between frames ${i + 1} and ${i + 2}` : `frame ${i + 1} of ${M.frames}`}`;
    if (v && v.video) line3 += ` · ${speedLabel(v.video.speed)}`;
  } else {
    const A = P.means, ft = M.steps.total / M.steps.ft, fts = ft.toFixed(1);
    const cyc = P.shed ? Math.round(P.shed.cycles) : 0, prelim = !P.shed && ft < 3 || cyc && cyc < 20;
    const what = `${A ? A.n.toLocaleString() : 0} samples in ${fts} flow-through${fts === '1.0' ? '' : 's'} (${(M.steps.total * M.dtStep * 1000).toFixed(0)} ms)${cyc ? `, about ${cyc} shedding cycles` : ''}${prelim ? ' · preliminary' : ''}`;
    line3 = `${P.view === 'mean' ? 'Averaged' : 'RMS'} over ${what}`;
  }
  return { title, line2, line3, spec: colourbarSpec() };
}
function speedLabel(sp) { const e = SPEEDS.find(([v]) => v === sp); return sp === 1 ? 'real time' : `${e ? e[1] : sp + ' ×'} real time`; }
function drawCaption(c, y0, W, Hc, u, info) {
  const s = info.spec, pad = 16 * u, cbW = 290 * u, rx = W - pad - cbW, lw = rx - pad - 20 * u;
  c.fillStyle = EXP_PAL.bg; c.fillRect(0, y0, W, Hc);
  c.fillStyle = EXP_PAL.rule; c.fillRect(0, y0, W, Math.max(1, Math.round(u)));
  c.textBaseline = 'alphabetic';
  c.fillStyle = EXP_PAL.ink; c.font = `600 ${20 * u}px "Barlow Condensed", "Arial Narrow", sans-serif`;
  fitText(c, info.title, pad, y0 + 27 * u, lw);
  c.fillStyle = EXP_PAL.muted; c.font = `400 ${11 * u}px "JetBrains Mono", ui-monospace, monospace`;
  fitText(c, info.line2, pad, y0 + 47 * u, lw);
  c.fillStyle = EXP_PAL.ink; fitText(c, info.line3, pad, y0 + 64 * u, lw);
  // colour bar with its scale
  c.fillStyle = EXP_PAL.muted; c.font = `500 ${11 * u}px "JetBrains Mono", ui-monospace, monospace`;
  const uw = s.unit ? Math.min(cbW * 0.4, c.measureText(s.unit).width) : 0;
  if (s.unit) fitText(c, s.unit, W - pad, y0 + 23 * u, cbW * 0.4, 'right');
  const short = { speed: 'Speed', vort: 'Vorticity', press: 'Pressure coefficient', smoke: 'Smoke', schl: 'Schlieren of heated model', piv: 'PIV vector field' }[S.field];
  const vw = G.play ? (G.play.view === 'mean' ? ' · time-averaged' : G.play.view === 'rms' ? ' · rms fluctuation' : ' · instantaneous') : '';
  c.fillStyle = EXP_PAL.ink; c.font = `600 ${13 * u}px "Barlow", system-ui, sans-serif`;
  fitText(c, short + vw, rx, y0 + 23 * u, cbW - uw - 10 * u);
  const by = y0 + 31 * u, bh = Math.max(4, Math.round(11 * u)), n = 256;
  for (let k = 0; k < n; k++) { const col = s.col(k / (n - 1)); c.fillStyle = `rgb(${col[0]},${col[1]},${col[2]})`; c.fillRect(rx + k * cbW / n, by, cbW / n + 1, bh); }
  c.strokeStyle = EXP_PAL.rule; c.lineWidth = Math.max(1, u); c.strokeRect(rx, by, cbW, bh);
  c.fillStyle = EXP_PAL.ink; c.font = `500 ${11 * u}px "JetBrains Mono", ui-monospace, monospace`; const ly = by + bh + 14 * u;
  if (s.ticks && s.ticks.length) {
    c.strokeStyle = EXP_PAL.muted; c.lineWidth = Math.max(1, u); c.beginPath();
    for (const [t] of s.ticks) { const x = rx + t * cbW; c.moveTo(x, by + bh); c.lineTo(x, by + bh + 3 * u); }
    c.stroke();
    for (const [t, lab] of s.ticks) fitText(c, lab, rx + t * cbW, ly, cbW / 4.2, t === 0 ? 'left' : t === 1 ? 'right' : 'center');
  } else { fitText(c, s.lo, rx, ly, cbW / 3, 'left'); fitText(c, s.hi, W - pad, ly, cbW / 3, 'right'); }
}
function composeFrame(c, L, info) {
  c.fillStyle = EXP_PAL.bg; c.fillRect(0, 0, L.W, L.H);
  wallBand(c, 0, L.W, L.wall, L.u); wallBand(c, L.wall + L.Hf, L.W, L.wall, L.u);
  if (MODE === 'gpu') c.drawImage(glc, 0, L.wall);
  c.drawImage(view, 0, L.wall);
  if (L.cap) drawCaption(c, L.Hf + 2 * L.wall, L.W, L.H - L.Hf - 2 * L.wall, L.u, info);
}
async function fontsReady() {
  try { await Promise.race([Promise.all(['600 20px "Barlow Condensed"', '600 13px "Barlow"', '500 11px "JetBrains Mono"'].map(f => document.fonts.load(f))), sleep(2500)]); } catch (e) { /* system fonts then */ }
}
function expName(ext, what) {
  const P = G.play, M = P && P.M;
  const base = M ? `${M.title.replace(/[^A-Za-z0-9.]+/g, '_')}_Re${M.Re}` : slug().replace(/[^A-Za-z0-9._-]+/g, '_');
  const view = what === 'video' ? 'video' : !P ? 'live' : P.view === 'mean' ? 'mean' : P.view === 'rms' ? 'rms' : `t${(Math.max(0, P.idx) * P.frameDt * 1000).toFixed(1)}ms`;
  return `${base}_${({ speed: 'speed', vort: 'vorticity', press: 'pressure', smoke: 'smoke', schl: 'schlieren', piv: 'piv' })[S.field]}_${view}.${ext}`;
}

// ----- delivery: the page's save prompt, with an in-page preview when saving is not possible here -----
async function deliverFile(filename, blob) {
  if (STANDALONE) { plainDownload(filename, blob); return true; }
  const dl = await Promise.race([dlP, sleep(1500).then(() => null)]);
  if (dl) {
    try { await dl.save({ filename, data: blob }); return true; }
    catch (e) {
      const c = e && e.code;
      if (c === 'declined') return false;
      if (c === 'rate_limited') { toast('Another save is already waiting for your answer.'); return false; }
      if (c === 'too_large') { toast('That file is too large to save here. Try a smaller size or a faster playback speed.'); return false; }
    }
  }
  showFilePreview(filename, blob); return false;
}
function showFilePreview(filename, blob) {
  const d = $('#fileDialog'), url = URL.createObjectURL(blob), isVid = blob.type.startsWith('video');
  $('#fileName').textContent = `${filename} · ${fmtBytes(blob.size)}`;
  $('#fileView').innerHTML = isVid ? `<video controls playsinline src="${url}"></video>` : `<img alt="Exported image" src="${url}">`;
  const a = $('#fileLink'); a.href = url; a.download = filename;
  d.hidden = false; $('#fileClose').focus();
  $('#fileClose').onclick = () => { d.hidden = true; $('#fileView').innerHTML = ''; setTimeout(() => URL.revokeObjectURL(url), 1000); };
}

// ----- image -----
async function exportImage(o) {
  await fontsReady();
  const L = expFit(o.width === 'screen' ? Math.max(640, view.width) : o.width, o.caption, false);
  const out = document.createElement('canvas'); out.width = L.W; out.height = L.H;
  expBegin(L, o);
  try { render(); composeFrame(out.getContext('2d'), L, captionInfo()); } finally { expEnd(); }
  const jpeg = o.fmt === 'jpeg', blob = await new Promise(r => out.toBlob(r, jpeg ? 'image/jpeg' : 'image/png', jpeg ? o.q : undefined));
  out.width = out.height = 0;
  if (!blob) throw new Error('The browser could not encode the image.');
  await deliverFile(expName(jpeg ? 'jpg' : 'png', 'image'), blob);
  return { W: L.W, H: L.H, bytes: blob.size };
}

// ----- video: WebCodecs encoder and the built-in MP4 writer -----
const H264_LEVELS = [[0x1f, 3600, 108000], [0x28, 8192, 245760], [0x2a, 8704, 522240], [0x32, 22080, 589824], [0x33, 36864, 983040], [0x34, 36864, 2073600]];
const VP9_LEVELS = [[31, 983040, 36864000], [40, 2228224, 83558400], [41, 2228224, 160432128], [50, 8912896, 311951360], [51, 8912896, 588251136], [52, 8912896, 1176502272]];
const vidBitrate = (W, H, fps) => Math.round(Math.min(25e6, Math.max(3e6, W * H * fps * 0.15)));
const codecCache = new Map();
async function pickCodec(W, H, fps) {
  if (typeof VideoEncoder === 'undefined' || typeof VideoFrame === 'undefined') return null;
  const key = `${W}x${H}@${fps}`; if (codecCache.has(key)) return codecCache.get(key);
  const bitrate = vidBitrate(W, H, fps), mbs = Math.ceil(W / 16) * Math.ceil(H / 16);
  const lv = (H264_LEVELS.find(([, fs, mr]) => mbs <= fs && mbs * fps <= mr) || H264_LEVELS[H264_LEVELS.length - 1])[0].toString(16).padStart(2, '0');
  const vl = (VP9_LEVELS.find(([, ps, sr]) => W * H <= ps && W * H * fps <= sr) || VP9_LEVELS[VP9_LEVELS.length - 1])[0];
  const base = { width: W, height: H, bitrate, framerate: fps, latencyMode: 'quality' };
  const cands = [
    ...['6400', '4d00', '4200'].map(p => ({ kind: 'avc', cfg: { ...base, codec: `avc1.${p}${lv}`, avc: { format: 'avc' } } })),
    { kind: 'vp9', vp9: { profile: 0, level: vl, bitDepth: 8 }, cfg: { ...base, codec: `vp09.00.${vl}.08` } },
  ];
  let pick = null;
  for (const c of cands) { try { if ((await VideoEncoder.isConfigSupported(c.cfg)).supported) { pick = c; break; } } catch (e) { /* try the next */ } }
  codecCache.set(key, pick); return pick;
}
// smooth movies end on the last recorded frame; held-frame movies show every frame for one frame interval
function videoFrames(M, P, o) { return o.interp && M.frames > 1 ? Math.round((M.frames - 1) * P.frameDt / o.speed * o.fps) + 1 : Math.max(1, Math.round(M.frames * P.frameDt / o.speed * o.fps)); }
function colourCodes(cs) { // WebCodecs colour space -> ISO/IEC 23091-2 code points for the 'colr' box (BT.709 when unknown)
  const P = { bt709: 1, bt470bg: 5, smpte170m: 6, bt2020: 9, smpte432: 12 }, T = { bt709: 1, smpte170m: 6, linear: 8, 'iec61966-2-1': 13, pq: 16, hlg: 18 }, Mx = { rgb: 0, bt709: 1, bt470bg: 5, smpte170m: 6, 'bt2020-ncl': 9 };
  return { p: (cs && P[cs.primaries]) || 1, t: (cs && T[cs.transfer]) || 1, m: (cs && cs.matrix in Mx) ? Mx[cs.matrix] : 1, full: !!(cs && cs.fullRange) };
}
async function ensureFrame(i) {
  const P = G.play;
  for (let k = 0; k < 300 && !P.frames.has(i); k++) { P.want = i; await loadFrame(i); if (!P.frames.has(i)) await sleep(5); }
}
async function exportVideo(o, progress, signal) {
  const P = G.play, M = P.M;
  await fontsReady();
  const L = expFit(o.width, o.caption, true), codec = await pickCodec(L.W, L.H, o.fps);
  if (!codec) throw new Error('This browser cannot encode video.');
  const nOut = videoFrames(M, P, o), TS = 90000, dur = TS / o.fps;
  const chunks = []; let desc = null, encErr = null, colour = null;
  const enc = new VideoEncoder({
    output: (ch, meta) => {
      const b = new Uint8Array(ch.byteLength); ch.copyTo(b); chunks.push({ data: b, key: ch.type === 'key', ts: ch.timestamp });
      const dc = meta && meta.decoderConfig, d = dc && dc.description;
      if (d && !desc) desc = d instanceof ArrayBuffer ? new Uint8Array(d.slice(0)) : new Uint8Array(d.buffer.slice(d.byteOffset, d.byteOffset + d.byteLength));
      if (dc && dc.colorSpace && !colour) colour = dc.colorSpace;
    },
    error: (e) => { encErr = e; },
  });
  enc.configure(codec.cfg);
  const out = document.createElement('canvas'); out.width = L.W; out.height = L.H; const octx = out.getContext('2d');
  const keep = { idx: P.idx, view: P.view, playing: P.playing, time: P.time };
  P.playing = false; if (P.view !== 'inst') { P.view = 'inst'; P.mv = null; $('#pbView').value = 'inst'; }
  expBegin(L, o); wake(true);
  const t0 = performance.now();
  try {
    let lastX = -1;
    const raw = S.field === 'piv' && S.pivView === 'raw', animated = raw || S.field === 'smoke' || S.field === 'schl'; // film grain and particles move even on a held frame
    for (let j = 0; j < nOut; j++) {
      if (signal.aborted) throw new DOMException('cancelled', 'AbortError');
      if (ctxLost) throw new Error('the graphics processor was reset, so the export stopped');
      if (encErr) throw encErr;
      // position in recorded frames; between two frames the picture is interpolated along the flow, or the earlier one is held
      const xf = Math.min(M.frames - 1, j / o.fps * o.speed / P.frameDt), i = Math.floor(xf + 1e-9), sb = xf - i;
      const blend = o.interp && sb > 0.02 && i + 1 < M.frames, x = blend ? xf : i, fresh = x !== lastX;
      if (fresh) {
        await ensureFrame(i); if (blend) await ensureFrame(i + 1);
        for (let k = 1; k <= 4; k++) if (i + k < M.frames) loadFrame(i + k);
        const A = P.frames.get(i), B = blend ? P.frames.get(i + 1) : null;
        if (A && B) applyFrame(blendFrames(A, B, sb), i); else showFrame(i);
        lastX = x; G.pivF = null;
      }
      if (raw) advectPIV(Math.min(40, o.speed / o.fps / M.dtStep));
      if (fresh || animated) { G.frame++; render(); composeFrame(octx, L, captionInfo({ x, video: o })); }
      const vf = new VideoFrame(out, { timestamp: Math.round(j * 1e6 / o.fps), duration: Math.round(1e6 / o.fps) });
      enc.encode(vf, { keyFrame: j % Math.round(2 * o.fps) === 0 }); vf.close();
      while (enc.encodeQueueSize > 2 && !encErr) await sleep(2); // let the encoder keep up, so frames do not pile up in memory
      if (j % 3 === 2) await sleep(0);
      progress(j + 1, nOut, (performance.now() - t0) / 1000);
    }
    await enc.flush();
    if (encErr) throw encErr;
  } finally {
    try { enc.close(); } catch (e) { /* already closed */ }
    expEnd(); wake(false); out.width = out.height = 0;
    Object.assign(P, { playing: keep.playing, time: keep.time }); P.idx = -1;
    if (keep.view !== 'inst') { P.view = keep.view; P.mv = null; $('#pbView').value = keep.view; showFrame(Math.max(0, keep.idx)); }
    else { await ensureFrame(Math.max(0, keep.idx)); showFrame(Math.max(0, keep.idx)); }
    syncPlayButton();
  }
  if (codec.kind === 'avc' && !desc) throw new Error('The video encoder did not describe its output.');
  const blob = mp4Mux({ codec: codec.kind, width: L.W, height: L.H, timescale: TS, samples: chunks.map(c => ({ data: c.data, dur, key: c.key, pts: Math.round(c.ts * TS / 1e6) })), avcC: desc, vp9: codec.vp9, colour: colourCodes(colour) });
  await deliverFile(expName('mp4', 'video'), blob);
  return { W: L.W, H: L.H, bytes: blob.size, codec: codec.kind, frames: nOut };
}

// ----- export dialog -----
const EXP_FIELDS = [['speed', 'Speed'], ['vort', 'Vorticity'], ['press', 'Pressure'], ['smoke', 'Smoke'], ['schl', 'Schlieren'], ['piv', 'PIV']];
function openExport(kind) {
  if (exporting) return;
  const P = G.play, d = $('#expDialog'), vid = kind === 'video';
  if (vid && !P) return;
  $('#expTitle').textContent = vid ? 'Save a video of this run' : P ? 'Save an image of this run' : 'Save an image of the tunnel';
  $('#exField').innerHTML = EXP_FIELDS.map(([v, t]) => `<option value="${v}">${t}</option>`).join(''); $('#exField').value = S.field;
  $('#exViewL').hidden = vid || !P;
  if (P && !vid) {
    const hasMean = !!(P.means && P.means.n);
    $('#exView').innerHTML = `<option value="mean"${hasMean ? '' : ' disabled'}>Time-averaged</option><option value="rms"${hasMean ? '' : ' disabled'}>Fluctuations (rms)</option><option value="inst">Instantaneous (frame on screen)</option>`;
    $('#exView').value = P.view;
  }
  $('#exSize').innerHTML = vid ? '<option value="1280">1280 px wide</option><option value="1920" selected>1920 px wide</option><option value="2560">2560 px wide</option><option value="3840">3840 px wide</option>'
    : `<option value="screen">As on screen</option><option value="1920" selected>1920 px wide</option><option value="3840">3840 px wide</option>`;
  $('#exFmtL').hidden = vid; $('#exQL').hidden = vid || $('#exFmt').value !== 'jpeg';
  $('#exFpsL').hidden = !vid; $('#exSpdL').hidden = !vid; $('#exInterpL').hidden = !vid;
  if (vid) { $('#exSpeed').innerHTML = SPEEDS.map(([v, t]) => `<option value="${v}">${t}</option>`).join(''); $('#exSpeed').value = String(P.speed); if (P.playing) { P.playing = false; syncPlayButton(); } }
  $('#exGo').textContent = vid ? 'Save video' : 'Save image'; $('#exGo').disabled = false;
  $('#exProg').hidden = true; $('#exStatus').textContent = '';
  d.dataset.kind = kind; d.hidden = false; setExpBusy(false);
  expInfo(); $('#exGo').focus();
}
function setPlayView(v) { const s = $('#pbView'); s.value = v; s.dispatchEvent(new Event('change')); }
function expOpts() {
  const vid = $('#expDialog').dataset.kind === 'video', sz = $('#exSize').value;
  return { vid, width: sz === 'screen' ? 'screen' : +sz, fmt: $('#exFmt').value, q: +$('#exQ').value, fps: +$('#exFps').value, speed: +$('#exSpeed').value, caption: $('#exCap').checked, probes: $('#exProbes').checked, interp: $('#exInterp').checked };
}
let expInfoSeq = 0;
async function expInfo() {
  const o = expOpts(), el = $('#exInfo'), seq = ++expInfoSeq;
  if (!o.vid) {
    const L = expFit(o.width === 'screen' ? Math.max(640, view.width) : o.width, o.caption, false);
    el.textContent = `${L.W.toLocaleString()} × ${L.H.toLocaleString()} px ${o.fmt === 'png' ? 'PNG' : 'JPEG'}. Uses the colour map, contrast and overlays set under Display${G.play && G.play.view !== 'inst' ? '; the averaged fields come from the run’s statistics on a grid up to 900 cells across' : ''}.`;
    return;
  }
  const P = G.play, M = P.M, L = expFit(o.width, o.caption, true), n = videoFrames(M, P, o), secs = n / o.fps;
  const per = P.frameDt / o.speed * o.fps, rep = per >= 1.5 ? (o.interp ? `${Math.round(per - 1)} in-between frame${Math.round(per - 1) === 1 ? '' : 's'} for each recorded one, interpolated along the flow` : `each recorded frame is held for about ${per.toFixed(1)} video frames`) : per < 0.7 ? `about ${Math.round(1 / per)} recorded frames per video frame, so some are skipped` : 'about one recorded frame per video frame';
  const bytes = vidBitrate(L.W, L.H, o.fps) * secs / 8 * 0.8, maxB = deviceBudget().mobile ? 300e6 : 1.2e9;
  let txt = `${L.W} × ${L.H} px at ${o.fps} frames/s: ${secs < 90 ? secs.toFixed(1) + ' s' : fmtDur(secs)} of video (${n.toLocaleString()} frames) from ${M.frames} recorded frames, ${speedLabel(o.speed)}; ${rep}. About ${fmtBytes(bytes)}.`;
  if (L.W < o.width) txt += ` Reduced from ${o.width} px wide to fit the encoder.`;
  if (P.shed && P.frameDt > 0.25 / P.shed.f) txt += ` The recorded frames are ${(P.frameDt * P.shed.f).toFixed(2)} shedding periods apart, too far for the in-between frames to follow the vortices faithfully; record more frames per flow-through for smooth shedding.`;
  const ok = secs <= 600 && bytes <= maxB;
  if (secs > 600) txt += ' That is longer than 10 minutes: choose a faster speed.';
  else if (!ok) txt += ' That is more than this device can hold while it builds the file: choose a smaller size, a lower frame rate or a faster speed.';
  el.textContent = txt; $('#exGo').disabled = !ok;
  const c = await pickCodec(L.W, L.H, o.fps); if (seq !== expInfoSeq) return;
  if (!c) { el.textContent = txt + ' This browser cannot encode video. Use a current version of Chrome, Edge or Safari, or Firefox on a computer.'; $('#exGo').disabled = true; return; }
  el.textContent = txt + (c.kind === 'avc' ? ' H.264 in MP4, which plays almost everywhere.' : ' This browser only encodes VP9, so the MP4 holds VP9 video: it plays in Chrome, Edge, Firefox, VLC and on Android, but not in QuickTime or on older iPhones.');
}
function setExpBusy(on) {
  document.querySelectorAll('#expDialog select, #expDialog input').forEach(el => { el.disabled = on; });
  $('#exGo').disabled = on;
}
function buildExportUI() {
  ['exField', 'exView', 'exSize', 'exFmt', 'exQ', 'exFps', 'exSpeed', 'exCap', 'exProbes', 'exInterp'].forEach(id => $('#' + id).addEventListener('change', () => {
    if (id === 'exField') { S.field = $('#exField').value; const c = ctlById('field'); c.sync(); onChange(c, true); }
    if (id === 'exView') setPlayView($('#exView').value);
    if (id === 'exFmt') $('#exQL').hidden = $('#exFmt').value !== 'jpeg';
    if (id === 'exQ') $('#exQv').textContent = (+$('#exQ').value).toFixed(2);
    dirty = true; expInfo();
  }));
  $('#exQ').addEventListener('input', () => { $('#exQv').textContent = (+$('#exQ').value).toFixed(2); });
  $('#exCancel').addEventListener('click', () => { if (expAbort) { expAbort.abort(); return; } $('#expDialog').hidden = true; });
  $('#expDialog').addEventListener('keydown', (e) => { if (e.key === 'Escape' && !expAbort) $('#expDialog').hidden = true; });
  $('#exGo').addEventListener('click', async () => {
    const o = expOpts(), st = $('#exStatus');
    setExpBusy(true); st.textContent = o.vid ? 'Preparing…' : 'Rendering…';
    try {
      if (!o.vid) { const r = await exportImage(o); st.textContent = `Saved ${r.W} × ${r.H} px, ${fmtBytes(r.bytes)}.`; $('#expDialog').hidden = true; toast(`Image ready: ${r.W} × ${r.H} px, ${fmtBytes(r.bytes)}.`); }
      else {
        expAbort = new AbortController(); $('#exProg').hidden = false; $('#exCancel').textContent = 'Stop';
        const r = await exportVideo(o, (k, n, el) => {
          $('#exBarFill').style.width = `${(100 * k / n).toFixed(1)}%`;
          if (k % 5 === 0 || k === n) st.textContent = `Rendering frame ${k.toLocaleString()} of ${n.toLocaleString()}${el > 3 && k < n ? ` · about ${fmtDur(el / k * (n - k))} left` : ''}. Keep this page in front until it finishes.`;
        }, expAbort.signal);
        $('#expDialog').hidden = true; toast(`Video ready: ${r.W} × ${r.H} px, ${r.frames.toLocaleString()} frames, ${fmtBytes(r.bytes)}${r.codec === 'vp9' ? ' (VP9)' : ''}.`);
      }
    } catch (e) {
      if (e && e.name === 'AbortError') { st.textContent = 'Stopped.'; $('#expDialog').hidden = true; }
      else { console.error(e); st.textContent = `Couldn't save: ${(e && e.message) || e}`; }
    } finally {
      expAbort = null; $('#exCancel').textContent = 'Close'; setExpBusy(false); $('#exProg').hidden = true; $('#exBarFill').style.width = '0';
      if ($('#expDialog').dataset.kind === 'video') expInfo();
    }
  });
}
