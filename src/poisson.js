// ===== Incompressible (hydrodynamic) pressure from the velocity field: lap(p) = -(du_i/dx_j)(du_j/dx_i) =====
// Cell-centred multigrid V-cycles, red-black Gauss-Seidel. Neumann on the model, the walls and the inlet; p = 0 at the outlet.
class PoissonMG {
  constructor(nx, ny, mask) {
    this.levels = [];
    let w = nx, h = ny, m = mask;
    for (let l = 0; ; l++) {
      const L = { w, h, m, p: new Float32Array(w * h), f: new Float32Array(w * h), r: new Float32Array(w * h), h2: Math.pow(4, l) };
      this.levels.push(L);
      if (w <= 16 || h <= 12) break;
      const W = Math.ceil(w / 2), H = Math.ceil(h / 2), M = new Uint8Array(W * H);
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        let s = 0, c = 0;
        for (let j = 0; j < 2; j++) for (let i = 0; i < 2; i++) { const X = 2 * x + i, Y = 2 * y + j; if (X < w && Y < h) { c++; s += m[Y * w + X]; } }
        M[y * W + x] = s * 2 > c ? 1 : 0;
      }
      w = W; h = H; m = M;
    }
  }
  smooth(L, sweeps) {
    const { w, h, m, p, f, h2 } = L;
    for (let s = 0; s < sweeps; s++) for (let color = 0; color < 2; color++) {
      for (let y = 0; y < h; y++) for (let x = (y + color) & 1; x < w; x += 2) {
        const i = y * w + x; if (m[i]) continue;
        let sum = 0, d = 0;
        if (x > 0 && !m[i - 1]) { sum += p[i - 1]; d++; }
        if (x < w - 1) { if (!m[i + 1]) { sum += p[i + 1]; d++; } } else d += 2; // outlet face, p = 0
        if (y > 0 && !m[i - w]) { sum += p[i - w]; d++; }
        if (y < h - 1 && !m[i + w]) { sum += p[i + w]; d++; }
        if (d > 0) p[i] = (sum - h2 * f[i]) / d;
      }
    }
  }
  residual(L) {
    const { w, h, m, p, f, r, h2 } = L;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = y * w + x; if (m[i]) { r[i] = 0; continue; }
      let lap = 0;
      if (x > 0 && !m[i - 1]) lap += p[i - 1] - p[i];
      if (x < w - 1) { if (!m[i + 1]) lap += p[i + 1] - p[i]; } else lap -= 2 * p[i];
      if (y > 0 && !m[i - w]) lap += p[i - w] - p[i];
      if (y < h - 1 && !m[i + w]) lap += p[i + w] - p[i];
      r[i] = f[i] - lap / h2;
    }
  }
  vcycle(l) {
    const L = this.levels[l];
    if (l === this.levels.length - 1) { this.smooth(L, 40); return; }
    this.smooth(L, 3); this.residual(L);
    const C = this.levels[l + 1];
    C.p.fill(0); C.f.fill(0);
    const cnt = new Uint8Array(C.w * C.h);
    for (let y = 0; y < L.h; y++) for (let x = 0; x < L.w; x++) {
      const i = y * L.w + x; if (L.m[i]) continue;
      const ci = (y >> 1) * C.w + (x >> 1); C.f[ci] += L.r[i]; cnt[ci]++;
    }
    for (let k = 0; k < C.f.length; k++) if (cnt[k]) C.f[k] /= cnt[k];
    this.vcycle(l + 1);
    // bilinear prolongation of the correction (cell-centred weights 3/4, 1/4), fluid coarse cells only
    for (let y = 0; y < L.h; y++) for (let x = 0; x < L.w; x++) {
      const i = y * L.w + x; if (L.m[i]) continue;
      const X0 = x >> 1, Y0 = y >> 1, X1 = (x & 1) ? X0 + 1 : X0 - 1, Y1 = (y & 1) ? Y0 + 1 : Y0 - 1;
      let s = 0, ws = 0;
      const add = (X, Y, wt) => { if (X < 0 || Y < 0 || X >= C.w || Y >= C.h) return; const k = Y * C.w + X; if (C.m[k]) return; s += wt * C.p[k]; ws += wt; };
      add(X0, Y0, 9); add(X1, Y0, 3); add(X0, Y1, 3); add(X1, Y1, 1);
      if (ws > 0) L.p[i] += s / ws;
    }
    this.smooth(L, 3);
  }
  // ux, uy on the same grid as the mask; returns pressure (velocity^2 units, density 1)
  solve(ux, uy, cycles) {
    const L = this.levels[0], { w, h, m, f } = L;
    const AX = this.ax || (this.ax = new Float32Array(w * h)), AY = this.ay || (this.ay = new Float32Array(w * h));
    // convective acceleration a = (u . grad) u in every fluid cell
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = y * w + x; if (m[i]) { AX[i] = AY[i] = 0; continue; }
      const l = x > 0 && !m[i - 1], r = x < w - 1 && !m[i + 1], d = y > 0 && !m[i - w], u = y < h - 1 && !m[i + w];
      const sx = (l && r) ? 0.5 : (l || r) ? 1 : 0, sy = (d && u) ? 0.5 : (d || u) ? 1 : 0;
      const il = l ? i - 1 : i, ir = r ? i + 1 : i, id = d ? i - w : i, iu = u ? i + w : i;
      const ux_x = (ux[ir] - ux[il]) * sx, uy_x = (uy[ir] - uy[il]) * sx, ux_y = (ux[iu] - ux[id]) * sy, uy_y = (uy[iu] - uy[id]) * sy;
      AX[i] = ux[i] * ux_x + uy[i] * ux_y; AY[i] = ux[i] * uy_x + uy[i] * uy_y;
    }
    // lap p = -div a, written as face fluxes between fluid cells only (wall faces carry the momentum balance exactly and drop out)
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = y * w + x; if (m[i]) { f[i] = 0; continue; }
      let rhs = 0;
      if (x > 0 && !m[i - 1]) rhs += 0.5 * (AX[i] + AX[i - 1]);
      if (x < w - 1) { if (!m[i + 1]) rhs -= 0.5 * (AX[i] + AX[i + 1]); } else rhs -= AX[i];
      if (y > 0 && !m[i - w]) rhs += 0.5 * (AY[i] + AY[i - w]);
      if (y < h - 1 && !m[i + w]) rhs -= 0.5 * (AY[i] + AY[i + w]);
      f[i] = rhs;
    }
    for (let c = 0; c < cycles; c++) this.vcycle(0);
    return L.p;
  }
}
