// ===== D2Q9 Lattice-Boltzmann core: BGK + Smagorinsky LES, Bouzidi interpolated bounce-back =====
const EX = [0, 1, 0, -1, 0, 1, -1, -1, 1];
const EY = [0, 0, 1, 0, -1, 1, 1, -1, -1];
const WQ = [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36];
const OPP = [0, 3, 4, 1, 2, 7, 8, 5, 6];
const MIRY = [0, 1, 4, 3, 2, 8, 7, 6, 5];

// ---- Geometry: shapes in chord units, reference point = quarter chord at (0.25, 0) ----
function nacaShape(m, p, t, n) {
  // m, p, t as fractions (e.g. 0.02, 0.4, 0.12). Closed trailing edge.
  n = n || 160;
  const up = [], lo = [];
  for (let k = 0; k <= n; k++) {
    const b = Math.PI * k / n;
    const x = 0.5 * (1 - Math.cos(b));
    const yt = 5 * t * (0.2969 * Math.sqrt(x) - 0.1260 * x - 0.3516 * x * x + 0.2843 * x * x * x - 0.1036 * x * x * x * x);
    let yc = 0, dy = 0;
    if (m > 0 && p > 0) {
      if (x < p) { yc = m / (p * p) * (2 * p * x - x * x); dy = 2 * m / (p * p) * (p - x); }
      else { yc = m / ((1 - p) * (1 - p)) * ((1 - 2 * p) + 2 * p * x - x * x); dy = 2 * m / ((1 - p) * (1 - p)) * (p - x); }
    }
    const th = Math.atan(dy);
    up.push({ x: x - yt * Math.sin(th), y: yc + yt * Math.cos(th), xc: x, s: 1 });
    lo.push({ x: x + yt * Math.sin(th), y: yc - yt * Math.cos(th), xc: x, s: -1 });
  }
  // CCW loop: TE -> upper -> LE -> lower -> TE
  const pts = [];
  for (let k = n; k >= 0; k--) pts.push(up[k]);
  for (let k = 1; k < n; k++) pts.push(lo[k]);
  return pts;
}
function cylinderShape(d, n) {
  n = n || 160;
  const pts = [];
  const r = d / 2;
  for (let k = 0; k < n; k++) {
    const a = 2 * Math.PI * k / n;
    const x = 0.25 + r * Math.cos(a), y = r * Math.sin(a);
    pts.push({ x, y, xc: (x - (0.25 - r)) / d, s: y >= 0 ? 1 : -1 });
  }
  return pts;
}


// ---- Trailing-edge flaps. A configuration is a list of elements (closed CCW loops, tagged e = 0, 1, ...), placed one
// after another in a single point array. Elements may overlap: the body is their union. Points carry xc (station along
// the retracted chord, for the pressure plot), s (+1 upper / -1 lower surface), e (element), te (thin region near a
// trailing edge, where the wall pressure is read from the first cell only) and hid (inside another element: not wetted).
function nacaSurfaces(m, p, t, n) {
  const up = [], lo = [], cam = [];
  for (let k = 0; k <= n; k++) {
    const x = 0.5 * (1 - Math.cos(Math.PI * k / n));
    const yt = 5 * t * (0.2969 * Math.sqrt(x) - 0.1260 * x - 0.3516 * x * x + 0.2843 * x * x * x - 0.1036 * x * x * x * x);
    let yc = 0, dy = 0;
    if (m > 0 && p > 0) {
      if (x < p) { yc = m / (p * p) * (2 * p * x - x * x); dy = 2 * m / (p * p) * (p - x); }
      else { yc = m / ((1 - p) * (1 - p)) * ((1 - 2 * p) + 2 * p * x - x * x); dy = 2 * m / ((1 - p) * (1 - p)) * (p - x); }
    }
    const th = Math.atan(dy);
    up.push({ x: x - yt * Math.sin(th), y: yc + yt * Math.cos(th) }); lo.push({ x: x + yt * Math.sin(th), y: yc - yt * Math.cos(th) }); cam.push({ x, y: yc });
  }
  return { up, lo, cam };
}
function yOn(curve, x) { // linear interpolation on a curve ordered by x
  if (x <= curve[0].x) return curve[0].y;
  for (let k = 1; k < curve.length; k++) if (curve[k].x >= x) { const a = curve[k - 1], b = curve[k]; return a.y + (b.y - a.y) * (x - a.x) / ((b.x - a.x) || 1); }
  return curve[curve.length - 1].y;
}
function rotTE(p, cx, cy, d) { // rotate about (cx, cy); positive d moves the trailing edge down
  const c = Math.cos(d), s = Math.sin(d), x = p.x - cx, y = p.y - cy;
  return { ...p, x: cx + c * x + s * y, y: cy - s * x + c * y };
}
function insidePoly(pts, x, y) {
  let c = false;
  for (let k = 0, j = pts.length - 1; k < pts.length; j = k++) { const a = pts[k], b = pts[j]; if ((a.y > y) !== (b.y > y) && x < a.x + (y - a.y) / (b.y - a.y) * (b.x - a.x)) c = !c; }
  return c;
}
function segDist(px, py, a, b) {
  const ex = b.x - a.x, ey = b.y - a.y, L = ex * ex + ey * ey || 1e-12, t = Math.max(0, Math.min(1, ((px - a.x) * ex + (py - a.y) * ey) / L));
  return Math.hypot(a.x + t * ex - px, a.y + t * ey - py);
}
function polyClearance(pts, x, y) { // distance from a point to a polygon, negative inside it
  let d = 1e9; for (let k = 0, j = pts.length - 1; k < pts.length; j = k++) d = Math.min(d, segDist(x, y, pts[j], pts[k]));
  return insidePoly(pts, x, y) ? -d : d;
}
function resample(pts, h) { // insert points so no edge is longer than h (straight plates need wall samples along them)
  const out = [];
  for (let k = 0; k < pts.length; k++) {
    const a = pts[k], b = pts[(k + 1) % pts.length], L = Math.hypot(b.x - a.x, b.y - a.y), m = Math.max(1, Math.ceil(L / h));
    for (let j = 0; j < m; j++) { const f = j / m; out.push({ ...a, x: a.x + f * (b.x - a.x), y: a.y + f * (b.y - a.y), xc: a.xc + f * (b.xc - a.xc) }); }
  }
  return out;
}
const FLAP_TYPES = {
  plain: { max: 40, presets: [0, 10, 30] },
  split: { max: 60, presets: [0, 20, 45] },
  slotted: { max: 40, presets: [0, 15, 35] },
  fowler: { max: 40, presets: [0, 15, 35] },
};
function flapShape(m, p, t, type, cf, deg, n) {
  n = n || 180;
  const { up, lo, cam } = nacaSurfaces(m, p, t, n), d = deg * Math.PI / 180, xf = 1 - cf;
  const tag = (pts, e, extra) => pts.map(q => ({ xc: q.x, ...q, e, ...(extra || {}) }));
  const els = [];
  if (type === 'plain' || type === 'split' && deg < 0.25) {
    if (type === 'split') return nacaShape(m, p, t, n);
    // fixed part and flap, each closed by a circular knuckle round the hinge on the camber line
    const hy = yOn(cam, xf), ru = yOn(up, xf) - hy, rl = hy - yOn(lo, xf), K = 24;
    const arc = (a0, a1, r0, r1) => { const o = []; for (let k = 0; k <= K; k++) { const f = k / K, a = a0 + (a1 - a0) * f, r = r0 + (r1 - r0) * f; o.push({ x: xf + r * Math.cos(a), y: hy + r * Math.sin(a), s: Math.sin(a) >= 0 ? 1 : -1 }); } return o; };
    const fixed = [...up.filter(q => q.x < xf).reverse().map(q => ({ ...q, s: 1 })), ...lo.filter(q => q.x < xf && q.x > 0).map(q => ({ ...q, s: -1 })), ...arc(-Math.PI / 2, Math.PI / 2, rl, ru)];
    const flap = [...arc(Math.PI / 2, 1.5 * Math.PI, ru, rl), ...lo.filter(q => q.x > xf).map(q => ({ ...q, s: -1 })), ...up.filter(q => q.x > xf && q.x < 1).reverse().map(q => ({ ...q, s: 1 }))];
    els.push(tag(fixed, 0), tag(flap, 1).map(q => { const r = rotTE(q, xf, hy, d); return { ...r, xc: r.x, te: q.xc > 1 - 0.35 * cf }; })); // xc: chordwise station of the deflected geometry, as for every flap type
  } else if (type === 'split') {
    // the whole section stays put; a thin plate hinged on the lower surface swings down
    const main = nacaShape(m, p, t, n).map(q => ({ ...q, e: 0 }));
    const hx = xf, hy = yOn(lo, xf), te = { x: 1, y: 0 }, L = Math.hypot(te.x - hx, te.y - hy), ux = (te.x - hx) / L, uy = (te.y - hy) / L, th = 0.011;
    const nxv = uy, nyv = -ux; // downward normal of the retracted plate
    const a = { x: hx - 0.012 * ux, y: hy - 0.012 * uy }, b = { x: hx + L * ux, y: hy + L * uy };
    // corners CCW; each corner's s labels the face that starts there: the outer (lower) face -1, the face toward the wing +1
    const corners = [{ x: a.x, y: a.y, s: -1, xc: hx }, { x: a.x + th * nxv, y: a.y + th * nyv, s: -1, xc: hx }, { x: b.x + th * nxv, y: b.y + th * nyv, s: 1, xc: 1 }, { x: b.x, y: b.y, s: 1, xc: 1 }];
    const plate = resample(corners.map(q => rotTE(q, hx, hy, d)), 0.008).map(q => ({ ...q, xc: q.x, e: 1, te: true }));
    els.push(main, plate);
  } else {
    // slotted and Fowler: the main element ends in a shroud over a cove; the flap is the rear of the section with a
    // rounded nose, carried out of the cove on its track and rotated, leaving a slot of about 2% chord
    const fowler = type === 'fowler', xs = xf + (fowler ? 0.8 : 0.2) * cf, tl = 0.006, ylip = yOn(up, xs) - tl;
    const yl0 = yOn(lo, xf), C = 14, cove = [];
    for (let k = 1; k < C; k++) { const ph = Math.PI / 2 * k / C; cove.push({ x: xs - (xs - xf) * Math.cos(ph), y: yl0 + (ylip - yl0) * Math.sin(ph), s: -1, te: true, cove: true }); }
    const main = [{ x: xs, y: yOn(up, xs), s: 1, te: true, cove: true }, ...up.filter(q => q.x < xs).reverse().map(q => ({ ...q, s: 1, te: q.x > xs - 0.06 })),
      ...lo.filter(q => q.x < xf && q.x > 0).map(q => ({ ...q, s: -1 })), { x: xf, y: yl0, s: -1 }, ...cove, { x: xs, y: ylip, s: -1, te: true, cove: true }];
    // flap: aft part of the section with an elliptical nose
    const a = 0.5 * (yOn(up, xf) - yOn(lo, xf)) * 0.75, xn = xf + a, ymid = 0.5 * (yOn(up, xn) + yOn(lo, xn)), b = 0.5 * (yOn(up, xn) - yOn(lo, xn));
    const nose = []; for (let k = 0; k <= 20; k++) { const ang = Math.PI / 2 + Math.PI * k / 20; nose.push({ x: xn + a * Math.cos(ang), y: ymid + b * Math.sin(ang), s: Math.sin(ang) >= 0 ? 1 : -1 }); }
    let flap = [...nose, ...lo.filter(q => q.x > xn).map(q => ({ ...q, s: -1 })), ...up.filter(q => q.x > xn && q.x < 1).reverse().map(q => ({ ...q, s: 1 }))].map(q => ({ ...q, te: q.x > 1 - 0.35 * cf }));
    if (deg > 0.25) {
      const N = { x: xf, y: ymid }, prog = fowler ? Math.sin(Math.PI / 2 * Math.min(1, deg / 30)) : Math.min(1, deg / 15), lex = xf + (xs - 0.012 - xf) * prog, gap = 0.018 * Math.min(1, deg / 12);
      const rot = flap.map(q => rotTE(q, N.x, N.y, d)); let minx = 1e9; for (const q of rot) minx = Math.min(minx, q.x);
      const place = (dy) => rot.map(q => ({ ...q, x: q.x + lex - minx, y: q.y + dy }));
      // lower the flap until it clears the shroud and cove everywhere by the slot width
      const under = main.filter(q => q.x > xf - 1e-6 && q.s === -1).concat([{ x: xs, y: ylip }]);
      const clear = (F) => { let c = 1e9; for (const q of under) c = Math.min(c, polyClearance(F, q.x, q.y)); for (const q of F) if (q.x < xs + 0.01) c = Math.min(c, polyClearance(main, q.x, q.y)); return c; };
      let lo2 = -0.4, hi2 = 0.1;
      for (let it = 0; it < 50; it++) { const mid = 0.5 * (lo2 + hi2); if (clear(place(mid)) < gap) hi2 = mid; else lo2 = mid; }
      flap = place(lo2);
    }
    els.push(tag(main, 0), tag(flap, 1).map((q, k) => ({ ...q, xc: q.x })));
  }
  // flag points that lie inside another element (not wetted)
  const out = [];
  els.forEach((E, i) => { for (const q of E) { let hid = false; els.forEach((F, j) => { if (j !== i && polyClearance(F, q.x, q.y) < -1e-4) hid = true; }); out.push({ ...q, e: i, hid }); } });
  return out;
}
// next/previous point within the same element, for loops of several elements in one array
function elementRing(poly) {
  const n = poly.length, next = new Int32Array(n), prev = new Int32Array(n), starts = [];
  let s = 0;
  for (let k = 1; k <= n; k++) if (k === n || (poly[k].e | 0) !== (poly[s].e | 0)) { starts.push([s, k]); for (let j = s; j < k; j++) { next[j] = j + 1 < k ? j + 1 : s; prev[j] = j > s ? j - 1 : k - 1; } s = k; }
  return { next, prev, ranges: starts };
}


// Rasterise a polygon (lattice coords, cell centres at integers) and find every fluid->solid link
// with its fractional wall distance, for interpolated bounce-back.
function bodyGeometry(poly, nx, ny, margin) {
  let x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9;
  for (const p of poly) { x0 = Math.min(x0, p.x); x1 = Math.max(x1, p.x); y0 = Math.min(y0, p.y); y1 = Math.max(y1, p.y); }
  const bx0 = Math.max(1, Math.floor(x0) - 1), bx1 = Math.min(nx - 2, Math.ceil(x1) + 1);
  const by0 = Math.max(1, Math.floor(y0) - 1), by1 = Math.min(ny - 2, Math.ceil(y1) + 1);
  const np = poly.length, list = [], ring = elementRing(poly), next = ring.next;
  const W = bx1 - bx0 + 3, H = by1 - by0 + 3, ox = bx0 - 1, oy = by0 - 1;
  const loc = new Uint8Array(W * H); // element id + 1 of the cell, 0 for fluid
  for (const [r0, r1] of ring.ranges) {
    const el = (poly[r0].e | 0) + 1;
    for (let y = by0; y <= by1; y++) {
      const xs = [];
      for (let k = r0; k < r1; k++) { const a = poly[k], b = poly[next[k]]; if ((a.y > y) !== (b.y > y)) xs.push(a.x + (y - a.y) / (b.y - a.y) * (b.x - a.x)); }
      xs.sort((a, b) => a - b);
      for (let k = 0; k + 1 < xs.length; k += 2) {
        const xa = Math.max(bx0, Math.ceil(xs[k])), xb = Math.min(bx1, Math.floor(xs[k + 1]));
        for (let x = xa; x <= xb; x++) { const o = (y - oy) * W + (x - ox); if (!loc[o]) { loc[o] = el; list.push(y * nx + x); } }
      }
    }
  }
  const isS = (x, y) => x >= ox && y >= oy && x < ox + W && y < oy + H && loc[(y - oy) * W + (x - ox)] !== 0;
  const elAt = (x, y) => loc[(y - oy) * W + (x - ox)] - 1;
  // bucket surface edges by cell so each link only tests nearby edges
  const B = new Map();
  for (let j = 0; j < np; j++) {
    const a = poly[j], b = poly[next[j]];
    const ex0 = Math.floor(Math.min(a.x, b.x)) - 1, ex1 = Math.ceil(Math.max(a.x, b.x)) + 1;
    const ey0 = Math.floor(Math.min(a.y, b.y)) - 1, ey1 = Math.ceil(Math.max(a.y, b.y)) + 1;
    for (let y = ey0; y <= ey1; y++) for (let x = ex0; x <= ex1; x++) { const key = y * 65536 + x; let l = B.get(key); if (!l) B.set(key, l = []); l.push(j); }
  }
  // A link exists wherever the segment from a fluid node to its neighbour crosses the surface,
  // even when the surface there is thinner than a cell (trailing edges): otherwise fluid leaks through.
  // With overlapping elements the first crossing is always on the outer surface of the union.
  const Li = [], Lq = [], Ld = [], Le = [];
  for (let y = oy; y < oy + H; y++) for (let x = ox; x < ox + W; x++) {
    if (x < 0 || y < 0 || x >= nx || y >= ny || isS(x, y)) continue;
    const i = y * nx + x, cand = B.get(y * 65536 + x);
    for (let q = 1; q < 9; q++) {
      const solidNb = isS(x + EX[q], y + EY[q]);
      if (!cand && !solidNb) continue;
      let tmin = 2, emin = 0;
      const dx = EX[q], dy = EY[q];
      if (cand) for (const j of cand) {
        const a = poly[j], b = poly[next[j]];
        const ex = b.x - a.x, ey = b.y - a.y;
        const den = dx * ey - dy * ex; if (Math.abs(den) < 1e-12) continue;
        const wx = a.x - x, wy = a.y - y;
        const t = (wx * ey - wy * ex) / den, s = (wx * dy - wy * dx) / den;
        if (t > 1e-9 && t <= 1 && s >= 0 && s <= 1 && t < tmin) { tmin = t; emin = a.e | 0; }
      }
      if (tmin > 1) { if (!solidNb) continue; tmin = 0.5; emin = Math.max(0, elAt(x + EX[q], y + EY[q])); }
      Li.push(i); Lq.push(q); Ld.push(Math.min(1, Math.max(0.02, tmin))); Le.push(emin);
    }
  }
  // distance from the surface (chamfer 3-4 transform on the rasterised mask) over the body box plus a margin
  let dist = null;
  if (margin > 0) {
    const M = Math.ceil(margin) + 1, dx0 = Math.max(0, ox - M), dy0 = Math.max(0, oy - M);
    const dx1 = Math.min(nx - 1, ox + W - 1 + M), dy1 = Math.min(ny - 1, oy + H - 1 + M), dw = dx1 - dx0 + 1, dh = dy1 - dy0 + 1;
    const D = new Float32Array(dw * dh);
    for (let y = 0; y < dh; y++) for (let x = 0; x < dw; x++) D[y * dw + x] = isS(x + dx0, y + dy0) ? 0 : 1e9;
    const a = 1, b = Math.SQRT2;
    for (let y = 0; y < dh; y++) for (let x = 0; x < dw; x++) {
      const i = y * dw + x; let v = D[i];
      if (x > 0) v = Math.min(v, D[i - 1] + a);
      if (y > 0) { v = Math.min(v, D[i - dw] + a); if (x > 0) v = Math.min(v, D[i - dw - 1] + b); if (x < dw - 1) v = Math.min(v, D[i - dw + 1] + b); }
      D[i] = v;
    }
    for (let y = dh - 1; y >= 0; y--) for (let x = dw - 1; x >= 0; x--) {
      const i = y * dw + x; let v = D[i];
      if (x < dw - 1) v = Math.min(v, D[i + 1] + a);
      if (y < dh - 1) { v = Math.min(v, D[i + dw] + a); if (x < dw - 1) v = Math.min(v, D[i + dw + 1] + b); if (x > 0) v = Math.min(v, D[i + dw - 1] + b); }
      D[i] = v;
    }
    dist = { x0: dx0, y0: dy0, w: dw, h: dh, d: D };
  }
  return { list, dist, box: { x0: ox, y0: oy, x1: ox + W - 1, y1: oy + H - 1 },
    links: { i: Int32Array.from(Li), q: Uint8Array.from(Lq), d: Float32Array.from(Ld), e: Uint8Array.from(Le), n: Li.length } };
}


// Inlet turbulence: random Fourier modes (divergence-free, frozen-turbulence convection), von Karman-like spectrum.
// Returns flat arrays so the GPU can take them as uniforms. Seeded, so CPU and GPU agree.
const TURB_MODES = 24;
function turbModes(ny, seed, u0) {
  u0 = u0 || 0.09;
  let st = (seed >>> 0) || 1; const rnd = () => { st = (st * 1664525 + 1013904223) >>> 0; return st / 4294967296; };
  const lmax = Math.max(40, ny / 2.2), lmin = Math.max(10, ny / 26), le = Math.max(20, ny / 8);
  const kmin = 6.2832 / lmax, kmax = 6.2832 / lmin, ke = 6.2832 / le;
  const K = [], E = [];
  for (let n = 0; n < TURB_MODES; n++) {
    const k = kmin * Math.pow(kmax / kmin, (n + 0.5) / TURB_MODES);
    const dk = k * Math.log(kmax / kmin) / TURB_MODES;
    const r = k / ke;
    E.push(Math.pow(r, 4) / Math.pow(1 + r * r, 17 / 6) * dk); K.push(k);
  }
  const sum = E.reduce((a, b) => a + b, 0);
  const kx = new Float32Array(TURB_MODES), ky = new Float32Array(TURB_MODES), am = new Float32Array(TURB_MODES), kk = new Float32Array(TURB_MODES), ph = new Float32Array(TURB_MODES), om = new Float32Array(TURB_MODES);
  for (let n = 0; n < TURB_MODES; n++) {
    const th = rnd() * 6.2832;
    am[n] = Math.sqrt(4 * E[n] / sum); // sum of a^2 = 4  ->  u_rms^2 + v_rms^2 = 2 per unit intensity
    kk[n] = K[n]; kx[n] = K[n] * Math.cos(th); ky[n] = K[n] * Math.sin(th);
    // each mode also evolves in the frame moving with the flow, so the eddies are not frozen plane waves
    ph[n] = rnd() * 6.2832; om[n] = (rnd() < 0.5 ? -1 : 1) * (0.08 + 0.12 * rnd()) * u0 * K[n];
  }
  return { kx, ky, k: kk, am, ph, om };
}



// probe samples for the CPU solver: [t, Fx, Fy, Mz, then rho, u, v, ok for each point]; pts = [x0, y0, x1, y1, ...] in cells
function sampleProbesLBM(sim, pts) {
  const n = pts.length / 2, row = new Float32Array(4 + 4 * n), nx = sim.nx, ny = sim.ny;
  row[0] = sim.t; row[1] = sim.Fx; row[2] = sim.Fy; row[3] = sim.Mz;
  for (let k = 0; k < n; k++) {
    const x = Math.min(nx - 1.001, Math.max(0, pts[2 * k])), y = Math.min(ny - 1.001, Math.max(0, pts[2 * k + 1]));
    const x0 = x | 0, y0 = y | 0, fx = x - x0, fy = y - y0; let r = 0, u = 0, v = 0, w = 0;
    for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++) {
      const i = (y0 + dy) * nx + x0 + dx; if (sim.solid[i]) continue;
      const wt = (dx ? fx : 1 - fx) * (dy ? fy : 1 - fy); r += wt * sim.rho[i]; u += wt * sim.ux[i]; v += wt * sim.uy[i]; w += wt;
    }
    const o = 4 + 4 * k; if (w > 1e-6) { row[o] = r / w; row[o + 1] = u / w; row[o + 2] = v / w; row[o + 3] = 1; } else row[o] = 1;
  }
  return row;
}
function packSamples(list) { if (!list.length) return null; const w = list[0].length, out = new Float32Array(list.length * w); list.forEach((r, i) => out.set(r, i * w)); return out; }

// ---- Simulation ----
class LBM {
  constructor(nx, ny) {
    this.nx = nx; this.ny = ny; const N = this.N = nx * ny;
    this.f = new Float32Array(9 * N); this.g = new Float32Array(9 * N);
    this.solid = new Uint8Array(N);
    this.rho = new Float32Array(N).fill(1); this.ux = new Float32Array(N); this.uy = new Float32Array(N);
    this.sponge = new Float32Array(nx); this.absorb = new Float32Array(nx);
    this.off = EX.map((e, q) => EY[q] * nx + e);
    this.u0 = 0.09; this.tau0 = 0.53; this.Cs = 0.16; this.setBulk(0.05); this.sigma = 0.95; this.cVreman = 2.5 * 0.25 * 0.25; this.smagC = 0.08; this.pux = new Float32Array(this.N); this.puy = new Float32Array(this.N); this.walls = 'solid'; this.turb = 0;
    this.solidList = []; this.links = { i: new Int32Array(0), q: new Uint8Array(0), d: new Float32Array(0), e: new Uint8Array(0), n: 0 };
    this.Fx = 0; this.Fy = 0; this.t = 0; this.rampSteps = 700;
    this.modes = turbModes(ny, 12345);
    this.sgsF = new Float32Array(this.N).fill(1); // sub-grid model factor (0 next to the model)
    this.fx0 = 24; this.fw = 5;
    this.initUniform(0);
    const sw = Math.max(8, Math.round(nx * 0.08));
    const iw = Math.max(6, Math.round(nx * 0.05));
    for (let x = 0; x < nx; x++) {
      const s = (x - (nx - sw)) / sw; this.sponge[x] = s > 0 ? 0.5 * s * s : 0;
      const si = 1 - x / iw;
      this.absorb[x] = (s > 0 ? 0.12 * s * s : 0) + (si > 0 ? 0.08 * si * si : 0);
    }
  }
  setBulk(nuB) { this.kBulk = 1 - 1 / (0.5 + 3 * nuB); }
  setFlow(u0, Re, Lref) {
    this.u0 = u0; this.Re = Re; this.Lref = Lref;
    this.sgsA = Math.max(1.5, 0.03 * Lref); this.sgsB = Math.max(4, 0.12 * Lref);
    this.fx0 = Math.round(0.3 * Lref); this.fw = Math.max(3, 0.06 * Lref);
    const nu = u0 * Lref / Re;
    this.tau0 = Math.max(0.5015, 3 * nu + 0.5);
    // near-wall sub-grid switch-off only where the grid resolves the laminar boundary layer (low cell Reynolds number)
    const reCell = u0 / ((this.tau0 - 0.5) / 3);
    this.sgsFloor = Math.min(1, Math.max(0, (reCell - 80) / 60));
  }
  initUniform(u) {
    const N = this.N, f = this.f;
    for (let i = 0; i < N; i++) {
      const ux = this.solid[i] ? 0 : u;
      for (let q = 0; q < 9; q++) {
        const cu = 3 * EX[q] * ux;
        f[q * N + i] = WQ[q] * (1 + cu + 0.5 * cu * cu - 1.5 * ux * ux);
      }
      this.rho[i] = 1; this.ux[i] = ux; this.uy[i] = 0;
    }
    this.t = 0;
  }
  inletU(y) {
    let u = this.u0;
    const r = Math.min(1, this.t / this.rampSteps); u *= r * r * (3 - 2 * r);
    let du = 0, dv = 0;
    if (this.walls === 'solid') { // thin inlet boundary layer avoids a corner singularity
      const d = Math.min(y, this.ny - 1 - y) + 0.5; const s = Math.tanh(d / 2.5);
      return [(u + du) * s, dv * s];
    }
    return [u + du, dv];
  }
  // body: polygon in lattice coords [{x,y}]
  setBody(poly) {
    const { nx, N, solid, f } = this;
    const prev = this.solidList;
    for (const i of prev) solid[i] = 0;
    const geo = bodyGeometry(poly, this.nx, this.ny, this.sgsB || 8);
    // sub-grid factor: zero within sgsA of the surface, ramping to one at sgsB
    if (this.sgsBox) { const B = this.sgsBox; for (let y = 0; y < B.h; y++) this.sgsF.fill(1, (B.y0 + y) * nx + B.x0, (B.y0 + y) * nx + B.x0 + B.w); }
    if (geo.dist) {
      const D = geo.dist, A = this.sgsA || 2, Bv = this.sgsB || 8;
      for (let y = 0; y < D.h; y++) for (let x = 0; x < D.w; x++) {
        const v = Math.min(1, Math.max(0, (D.d[y * D.w + x] - A) / (Bv - A)));
        this.sgsF[(D.y0 + y) * nx + D.x0 + x] = Math.round(255 * v) / 255;
      }
      this.sgsBox = D;
    }
    for (const i of geo.list) solid[i] = 1;
    for (const i of prev) if (!solid[i]) {
      for (let q = 0; q < 9; q++) f[q * N + i] = WQ[q];
      this.rho[i] = 1; this.ux[i] = 0; this.uy[i] = 0;
    }
    for (const i of geo.list) { this.ux[i] = 0; this.uy[i] = 0; }
    this.solidList = geo.list;
    this.links = geo.links;
    void nx;
  }
  step() {
    const { nx, ny, N, solid, rho, ux: UX, uy: UY, sponge, absorb } = this;
    let f = this.f, g = this.g;
    const tau0 = this.tau0, C = 18 * Math.SQRT2 * this.Cs * this.Cs;
    const sig = this.sigma, PX = this.pux, PY = this.puy, cV = this.cVreman;
    const sgsF = this.sgsF, solidWalls = this.walls === 'solid', sgsA = this.sgsA || 2, sgsB = this.sgsB || 8, sgsFloor = this.sgsFloor || 0;
    // divergence-free forcing strip (a turbulence grid ahead of the test section)
    const fOn = this.turb > 0, FM = this.modes, fx0 = this.fx0, fw = this.fw;
    const ramp = Math.min(1, this.t / this.rampSteps), rampS = ramp * ramp * (3 - 2 * ramp);
    const fAmp = (this.fGain || 0.9) * this.turb * this.u0 * this.u0 * rampS / (1.7725 * fw), Uc = this.u0, tt = this.t;
    const fxa = Math.max(1, Math.floor(fx0 - 3 * fw)), fxb = Math.ceil(fx0 + 3 * fw);
    PX.set(UX); PY.set(UY);
    // ---- collide (in place, f -> post-collision) ----
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) {
        const i = y * nx + x; if (solid[i]) continue;
        const f0 = f[i], f1 = f[N + i], f2 = f[2 * N + i], f3 = f[3 * N + i], f4 = f[4 * N + i],
          f5 = f[5 * N + i], f6 = f[6 * N + i], f7 = f[7 * N + i], f8 = f[8 * N + i];
        const r = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8;
        const ir = 1 / r;
        const u = (f1 + f5 + f8 - f3 - f6 - f7) * ir, v = (f2 + f5 + f6 - f4 - f7 - f8) * ir;
        rho[i] = r; UX[i] = u; UY[i] = v;
        const usq = 1.5 * (u * u + v * v);
        const r9 = r / 9, r36 = r / 36;
        const e0 = 4 * r9 * (1 - usq);
        const e1 = r9 * (1 + 3 * u + 4.5 * u * u - usq), e3 = r9 * (1 - 3 * u + 4.5 * u * u - usq);
        const e2 = r9 * (1 + 3 * v + 4.5 * v * v - usq), e4 = r9 * (1 - 3 * v + 4.5 * v * v - usq);
        const a = u + v, b = -u + v;
        const e5 = r36 * (1 + 3 * a + 4.5 * a * a - usq), e7 = r36 * (1 - 3 * a + 4.5 * a * a - usq);
        const e6 = r36 * (1 + 3 * b + 4.5 * b * b - usq), e8 = r36 * (1 - 3 * b + 4.5 * b * b - usq);
        const n1 = f1 - e1, n2 = f2 - e2, n3 = f3 - e3, n4 = f4 - e4, n5 = f5 - e5, n6 = f6 - e6, n7 = f7 - e7, n8 = f8 - e8;
        let pxx = n1 + n3 + n5 + n6 + n7 + n8, pyy = n2 + n4 + n5 + n6 + n7 + n8, pxy = n5 - n6 + n7 - n8;
        const t0 = tau0 + sponge[x];
        let tau = t0;
        if (x > 0 && x < nx - 1 && y > 0 && y < ny - 1) {
          // velocity gradients from the previous step (central differences)
          const il = i - 1, ir2 = i + 1, id = i - nx, iu = i + nx;
          const dudx = 0.5 * (PX[ir2] - PX[il]), dvdx = 0.5 * (PY[ir2] - PY[il]);
          const dudy = 0.5 * (PX[iu] - PX[id]), dvdy = 0.5 * (PY[iu] - PY[id]);
          // Vreman sub-grid model (2-D form): vanishes in pure shear, so laminar boundary layers are left alone
          const aa = dudx * dudx + dvdx * dvdx + dudy * dudy + dvdy * dvdy;
          const det = dudx * dvdy - dvdx * dudy;
          // sub-grid models are switched off next to the model and the walls, where the grid resolves the laminar boundary layer
          let fs = sgsF[i];
          if (solidWalls) { const dw = Math.min(y, ny - 1 - y) + 0.5; fs = Math.min(fs, Math.min(1, Math.max(0, (dw - sgsA) / (sgsB - sgsA)))); }
          fs = Math.max(fs, sgsFloor);
          if (aa > 1e-14) tau += fs * 3 * cV * Math.abs(det) / Math.sqrt(aa);
          if (this.smagC > 0) tau = 0.5 * (tau + Math.sqrt(tau * tau + fs * 18 * Math.SQRT2 * this.smagC * this.smagC * Math.sqrt(pxx * pxx + pyy * pyy + 2 * pxy * pxy) * ir));
          if (sig < 1) {
            // hybrid recursive regularization: blend in the stress from the finite-difference strain rate
            const fc = -r * tau / 3;
            pxx = sig * pxx + (1 - sig) * fc * 2 * dudx;
            pyy = sig * pyy + (1 - sig) * fc * 2 * dvdy;
            pxy = sig * pxy + (1 - sig) * fc * (dudy + dvdx);
          }
        }
        const k = 1 - 1 / tau, kb = this.kBulk;
        const hT = 0.5 * (pxx + pyy);
        const qxx = k * (pxx - hT) + kb * hT, qyy = k * (pyy - hT) + kb * hT, qxy = k * pxy;
        const tr = (qxx + qyy) / 3;
        const ra = 0.5 * (qxx - tr), rb = 0.5 * (qyy - tr);
        const rd = 0.25 * tr, rx = 0.25 * qxy;
        f[i] = e0 - 2 * tr;
        f[N + i] = e1 + ra; f[3 * N + i] = e3 + ra; f[2 * N + i] = e2 + rb; f[4 * N + i] = e4 + rb;
        f[5 * N + i] = e5 + rd + rx; f[7 * N + i] = e7 + rd + rx; f[6 * N + i] = e6 + rd - rx; f[8 * N + i] = e8 + rd - rx;
        if (fOn && x >= fxa && x <= fxb) {
          const xr = x - fx0, g = Math.exp(-xr * xr / (fw * fw)), gp = -2 * xr / (fw * fw) * g;
          let Fx = 0, Fy = 0;
          for (let n = 0; n < TURB_MODES; n++) {
            const th = FM.kx[n] * (x - Uc * tt) + FM.ky[n] * y + FM.ph[n] + FM.om[n] * tt;
            const b = FM.am[n] / FM.k[n], c = Math.cos(th), sn = Math.sin(th);
            Fx += g * b * FM.ky[n] * c;
            Fy -= gp * b * sn + g * b * FM.kx[n] * c;
          }
          Fx *= fAmp; Fy *= fAmp;
          // add momentum F (first-order forcing)
          f[N + i] += Fx / 3; f[3 * N + i] -= Fx / 3; f[2 * N + i] += Fy / 3; f[4 * N + i] -= Fy / 3;
          f[5 * N + i] += (Fx + Fy) / 12; f[7 * N + i] -= (Fx + Fy) / 12; f[6 * N + i] += (-Fx + Fy) / 12; f[8 * N + i] -= (-Fx + Fy) / 12;
        }
        const sg = absorb[x];
        if (sg > 0) { // absorbing layer: relax pressure toward far field
          const rr = sg * ir;
          f[i] += rr * e0 - sg * f[i]; f[N + i] += rr * e1 - sg * f[N + i]; f[2 * N + i] += rr * e2 - sg * f[2 * N + i];
          f[3 * N + i] += rr * e3 - sg * f[3 * N + i]; f[4 * N + i] += rr * e4 - sg * f[4 * N + i]; f[5 * N + i] += rr * e5 - sg * f[5 * N + i];
          f[6 * N + i] += rr * e6 - sg * f[6 * N + i]; f[7 * N + i] += rr * e7 - sg * f[7 * N + i]; f[8 * N + i] += rr * e8 - sg * f[8 * N + i];
        }
      }
    }
    // ---- stream (pull) ----
    const off = this.off;
    for (let y = 1; y < ny - 1; y++) {
      const row = y * nx;
      for (let x = 1; x < nx - 1; x++) {
        const i = row + x; if (solid[i]) continue;
        g[i] = f[i];
        for (let q = 1; q < 9; q++) g[q * N + i] = f[q * N + i - off[q]];
      }
    }
    const free = this.walls !== 'solid';
    const edgeNode = (x, y) => {
      const i = y * nx + x; if (solid[i]) return;
      for (let q = 0; q < 9; q++) {
        const sx = x - EX[q], sy = y - EY[q];
        if (sy < 0 || sy >= ny) {
          if (free) { const xs = Math.min(nx - 1, Math.max(0, sx)); g[q * N + i] = f[MIRY[q] * N + y * nx + xs]; }
          else g[q * N + i] = f[OPP[q] * N + i];
        } else if (sx < 0 || sx >= nx) g[q * N + i] = f[q * N + i];
        else g[q * N + i] = f[q * N + sy * nx + sx];
      }
    };
    for (let x = 0; x < nx; x++) { edgeNode(x, 0); edgeNode(x, ny - 1); }
    for (let y = 1; y < ny - 1; y++) { edgeNode(0, y); edgeNode(nx - 1, y); }
    // ---- curved boundary (Bouzidi) + momentum exchange ----
    const L = this.links; let Fx = 0, Fy = 0, Mz = 0, FxF = 0, FyF = 0, MzF = 0; const rx0 = this.refX || 0, ry0 = this.refY || 0, LE = L.e;
    // the flap's share is taken against the free-stream static pressure, read where the tunnel's pitot-static sits
    let rB = 1;
    if (LE && L.n) { const Lr = this.Lref || 50, xr = Math.min(nx - 1, Math.round(0.7 * Lr)), yb = Math.round(0.25 * Lr); let sr = 0, cr = 0;
      for (let y = yb; y < ny - yb; y++) { const i = y * nx + xr; if (!solid[i]) { sr += this.rho[i]; cr++; } } if (cr) rB = sr / cr; }
    for (let k = 0; k < L.n; k++) {
      const i = L.i[k], q = L.q[k], d = L.d[k], o = OPP[q];
      const fq = f[q * N + i];
      let val;
      if (d < 0.5) {
        const j = i - off[q];
        val = (j >= 0 && j < N && !solid[j]) ? 2 * d * fq + (1 - 2 * d) * f[q * N + j] : fq;
      } else {
        val = (fq + (2 * d - 1) * f[o * N + i]) / (2 * d);
      }
      g[o * N + i] = val;
      const fx = EX[q] * (fq + val), fy = EY[q] * (fq + val);
      Fx += fx; Fy += fy;
      const px = (i % nx) + d * EX[q] - rx0, py = ((i / nx) | 0) + d * EY[q] - ry0;
      Mz += px * fy - py * fx;
      // the flap's share, against the free-stream static pressure: a flap part whose outline is partly
      // buried in the wing is not a closed surface, so a uniform background would not cancel round it
      if (LE && LE[k]) { const gx = fx - 2 * WQ[q] * EX[q] * rB, gy = fy - 2 * WQ[q] * EY[q] * rB; FxF += gx; FyF += gy; MzF += px * gy - py * gx; }
    }
    this.Fx = Fx; this.Fy = Fy; this.Mz = Mz; this.FxF = FxF; this.FyF = FyF; this.MzF = MzF; // F: the flap's share
    // ---- outlet: zero-gradient for incoming populations ----
    for (let y = 0; y < ny; y++) {
      const i = y * nx + nx - 1, j = i - 1;
      g[3 * N + i] = g[3 * N + j]; g[6 * N + i] = g[6 * N + j]; g[7 * N + i] = g[7 * N + j];
    }
    // ---- inlet: equilibrium with prescribed velocity, density extrapolated ----
    for (let y = 0; y < ny; y++) {
      const i = y * nx; const r = rho[i + 1] || 1;
      const [u, v] = this.inletU(y);
      const usq = 1.5 * (u * u + v * v);
      for (let q = 0; q < 9; q++) {
        const cu = 3 * (EX[q] * u + EY[q] * v);
        g[q * N + i] = WQ[q] * r * (1 + cu + 0.5 * cu * cu - usq);
      }
    }
    this.f = g; this.g = f; this.t++;
  }
  // macroscopic fields are from the collide pass (one step behind) - fine for display
  isHealthy() {
    const { rho, N } = this; for (let k = 0; k < 64; k++) { const r = rho[(Math.random() * N) | 0]; if (!(r > 0.5 && r < 1.6)) return false; }
    return true;
  }
}
if (typeof module !== 'undefined') module.exports = { sampleProbesLBM, LBM, nacaShape, cylinderShape, bodyGeometry, turbModes, TURB_MODES, EX, EY };
