// ===== WebGL2 lattice-Boltzmann tunnel: same model as the CPU core, run as fragment-shader passes =====
const GLSL_HEAD = `#version 300 es
precision highp float; precision highp int; precision highp sampler2D;
`;
const VS = `#version 300 es
void main(){ vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2)); gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0); }`;
const LBM_CONST = `
const ivec2 CQ[9] = ivec2[9](ivec2(0,0),ivec2(1,0),ivec2(0,1),ivec2(-1,0),ivec2(0,-1),ivec2(1,1),ivec2(-1,1),ivec2(-1,-1),ivec2(1,-1));
const int OPPQ[9] = int[9](0,3,4,1,2,7,8,5,6);
const int MIRQ[9] = int[9](0,1,4,3,2,8,7,6,5);
const float WQ[9] = float[9](4./9.,1./9.,1./9.,1./9.,1./9.,1./36.,1./36.,1./36.,1./36.);
`;
const FS_INIT = GLSL_HEAD + LBM_CONST + `
layout(location=0) out vec4 o0; layout(location=1) out vec4 o1; layout(location=2) out vec4 o2;
void main(){ o0 = vec4(WQ[0],WQ[1],WQ[2],WQ[3]); o1 = vec4(WQ[4],WQ[5],WQ[6],WQ[7]); o2 = vec4(WQ[8],0.,0.,0.); }`;

const FS_COLLIDE = GLSL_HEAD + `
uniform sampler2D F0, F1, F2, SOL, MACP, SGSF;
uniform float tau0, cV, Csm, nx, spW, inW, kb, sig, sgsA, sgsB, sgsFloor;
uniform int ny, solidWalls;
uniform float fOn, fx0, fw, fAmp, Uc, tt;
uniform float mkx[24], mky[24], mk[24], mam[24], mph[24], mom[24];
layout(location=0) out vec4 o0; layout(location=1) out vec4 o1; layout(location=2) out vec4 o2; layout(location=3) out vec4 oM;
void main(){
  ivec2 p = ivec2(gl_FragCoord.xy);
  if (texelFetch(SOL, p, 0).r > 0.5) { o0 = vec4(0.); o1 = vec4(0.); o2 = vec4(0.); oM = vec4(1.,0.,0.,0.); return; }
  vec4 a = texelFetch(F0, p, 0), b = texelFetch(F1, p, 0), c = texelFetch(F2, p, 0);
  float f0=a.x,f1=a.y,f2=a.z,f3=a.w,f4=b.x,f5=b.y,f6=b.z,f7=b.w,f8=c.x;
  float r = f0+f1+f2+f3+f4+f5+f6+f7+f8, ir = 1.0/r;
  float u = (f1+f5+f8-f3-f6-f7)*ir, v = (f2+f5+f6-f4-f7-f8)*ir;
  float usq = 1.5*(u*u+v*v), r9 = r/9.0, r36 = r/36.0;
  float e0 = 4.0*r9*(1.0-usq);
  float e1 = r9*(1.0+3.0*u+4.5*u*u-usq), e3 = r9*(1.0-3.0*u+4.5*u*u-usq);
  float e2 = r9*(1.0+3.0*v+4.5*v*v-usq), e4 = r9*(1.0-3.0*v+4.5*v*v-usq);
  float aa = u+v, bb = -u+v;
  float e5 = r36*(1.0+3.0*aa+4.5*aa*aa-usq), e7 = r36*(1.0-3.0*aa+4.5*aa*aa-usq);
  float e6 = r36*(1.0+3.0*bb+4.5*bb*bb-usq), e8 = r36*(1.0-3.0*bb+4.5*bb*bb-usq);
  float n1=f1-e1,n2=f2-e2,n3=f3-e3,n4=f4-e4,n5=f5-e5,n6=f6-e6,n7=f7-e7,n8=f8-e8;
  float pxx = n1+n3+n5+n6+n7+n8, pyy = n2+n4+n5+n6+n7+n8, pxy = n5-n6+n7-n8;
  float x = float(p.x);
  bool interior = p.x > 0 && float(p.x) < nx - 1.0 && p.y > 0 && p.y < ny - 1;
  float s = (x-(nx-spW))/spW, sp = s > 0.0 ? 0.5*s*s : 0.0;
  float si = 1.0 - x/inW;
  float sg = (s > 0.0 ? 0.12*s*s : 0.0) + (si > 0.0 ? 0.08*si*si : 0.0);
  float tau = tau0 + sp;
  if (interior) {
    vec4 ml = texelFetch(MACP, p - ivec2(1,0), 0), mr = texelFetch(MACP, p + ivec2(1,0), 0);
    vec4 md = texelFetch(MACP, p - ivec2(0,1), 0), mu = texelFetch(MACP, p + ivec2(0,1), 0);
    float dudx = 0.5*(mr.y - ml.y), dvdx = 0.5*(mr.z - ml.z), dudy = 0.5*(mu.y - md.y), dvdy = 0.5*(mu.z - md.z);
    // Vreman sub-grid model (2-D form): zero in pure shear, so laminar boundary layers are untouched
    float ga = dudx*dudx + dvdx*dvdx + dudy*dudy + dvdy*dvdy;
    float det = dudx*dvdy - dvdx*dudy;
    // sub-grid models are switched off next to the model and the walls, where the laminar boundary layer is resolved
    float fs = texelFetch(SGSF, p, 0).r;
    if (solidWalls == 1) { float dw = min(float(p.y), float(ny-1-p.y)) + 0.5; fs = min(fs, clamp((dw - sgsA)/(sgsB - sgsA), 0.0, 1.0)); }
    fs = max(fs, sgsFloor);
    if (ga > 1e-14) tau += fs*3.0*cV*abs(det)/sqrt(ga);
    // weak Smagorinsky floor for robustness under strong inflow turbulence
    float Qn = sqrt(pxx*pxx + pyy*pyy + 2.0*pxy*pxy);
    tau = 0.5*(tau + sqrt(tau*tau + fs*Csm*Qn*ir));
    if (sig < 1.0) { // hybrid recursive regularization (finite-difference strain blend)
      float fc = -r*tau/3.0;
      pxx = sig*pxx + (1.0-sig)*fc*2.0*dudx;
      pyy = sig*pyy + (1.0-sig)*fc*2.0*dvdy;
      pxy = sig*pxy + (1.0-sig)*fc*(dudy + dvdx);
    }
  }
  float k = 1.0 - 1.0/tau;
  float hT = 0.5*(pxx+pyy);
  float qxx = k*(pxx-hT) + kb*hT, qyy = k*(pyy-hT) + kb*hT, qxy = k*pxy;
  float tr = (qxx+qyy)/3.0;
  float ra = 0.5*(qxx-tr), rb = 0.5*(qyy-tr), rd = 0.25*tr, rx = 0.25*qxy;
  float g0 = e0-2.0*tr, g1 = e1+ra, g3 = e3+ra, g2 = e2+rb, g4 = e4+rb;
  float g5 = e5+rd+rx, g7 = e7+rd+rx, g6 = e6+rd-rx, g8 = e8+rd-rx;
  if (fOn > 0.5 && abs(x - fx0) <= 3.0*fw && p.x >= 1) { // divergence-free forcing strip (turbulence grid)
    float xr = x - fx0, gg = exp(-xr*xr/(fw*fw)), gp = -2.0*xr/(fw*fw)*gg, y = float(p.y);
    float Fx = 0.0, Fy = 0.0;
    for (int n = 0; n < 24; n++) {
      float th = mkx[n]*(x - Uc*tt) + mky[n]*y + mph[n] + mom[n]*tt;
      float bb = mam[n]/mk[n], cth = cos(th), sth = sin(th);
      Fx += gg*bb*mky[n]*cth;
      Fy -= gp*bb*sth + gg*bb*mkx[n]*cth;
    }
    Fx *= fAmp; Fy *= fAmp;
    g1 += Fx/3.0; g3 -= Fx/3.0; g2 += Fy/3.0; g4 -= Fy/3.0;
    g5 += (Fx+Fy)/12.0; g7 -= (Fx+Fy)/12.0; g6 += (-Fx+Fy)/12.0; g8 -= (-Fx+Fy)/12.0;
  }
  if (sg > 0.0) { float rr = sg*ir;
    g0 += rr*e0-sg*g0; g1 += rr*e1-sg*g1; g2 += rr*e2-sg*g2; g3 += rr*e3-sg*g3; g4 += rr*e4-sg*g4;
    g5 += rr*e5-sg*g5; g6 += rr*e6-sg*g6; g7 += rr*e7-sg*g7; g8 += rr*e8-sg*g8; }
  o0 = vec4(g0,g1,g2,g3); o1 = vec4(g4,g5,g6,g7); o2 = vec4(g8,0.,0.,0.); oM = vec4(r,u,v,0.);
}`;

const FS_STREAM = GLSL_HEAD + LBM_CONST + `
uniform sampler2D P0, P1, P2, SOL, MAC, D1, D2;
uniform int nx, ny, freeSlip, solidWalls;
uniform float uIn;
layout(location=0) out vec4 o0; layout(location=1) out vec4 o1; layout(location=2) out vec4 o2;
float Pq(int q, ivec2 p){ if (q < 4) return texelFetch(P0,p,0)[q]; if (q < 8) return texelFetch(P1,p,0)[q-4]; return texelFetch(P2,p,0).x; }
bool solidAt(ivec2 p){ return texelFetch(SOL,p,0).r > 0.5; }
void main(){
  ivec2 p = ivec2(gl_FragCoord.xy);
  float f[9];
  if (solidAt(p)) { o0 = vec4(WQ[0],WQ[1],WQ[2],WQ[3]); o1 = vec4(WQ[4],WQ[5],WQ[6],WQ[7]); o2 = vec4(WQ[8],0.,0.,0.); return; }
  if (p.x == 0) {
    float y = float(p.y), u = uIn, dv = 0.0;
    if (solidWalls == 1) { float d = min(y, float(ny)-1.0-y) + 0.5; float sc = tanh(d/2.5); u *= sc; dv *= sc; }
    float r = texelFetch(MAC, ivec2(1,p.y), 0).x; if (r <= 0.0) r = 1.0;
    float usq = 1.5*(u*u+dv*dv);
    for (int q = 0; q < 9; q++) { float cu = 3.0*(float(CQ[q].x)*u + float(CQ[q].y)*dv); f[q] = WQ[q]*r*(1.0+cu+0.5*cu*cu-usq); }
  } else {
    vec4 d1 = texelFetch(D1,p,0), d2 = texelFetch(D2,p,0);
    float dd[9] = float[9](0., d1.x, d1.y, d1.z, d1.w, d2.x, d2.y, d2.z, d2.w);
    bool outlet = p.x == nx-1;
    for (int q = 0; q < 9; q++) {
      ivec2 bq = (outlet && CQ[q].x < 0) ? ivec2(nx-2, p.y) : p;
      ivec2 s = bq - CQ[q];
      float val;
      if (s.y < 0 || s.y >= ny) {
        if (freeSlip == 1) val = Pq(MIRQ[q], ivec2(clamp(bq.x - CQ[q].x, 0, nx-1), bq.y));
        else val = Pq(OPPQ[q], bq);
      } else if (s.x < 0 || s.x >= nx) {
        val = Pq(q, bq);
      } else if (dd[OPPQ[q]] > 0.0 || solidAt(s)) { // wall link (also across surfaces thinner than a cell)
        int a = OPPQ[q]; float d = dd[a]; float fa = Pq(a, bq);
        if (d <= 0.0) val = fa;
        else if (d < 0.5) { ivec2 j = bq - CQ[a]; val = solidAt(j) ? fa : 2.0*d*fa + (1.0-2.0*d)*Pq(a, j); }
        else val = (fa + (2.0*d-1.0)*Pq(q, bq)) / (2.0*d);
      } else val = Pq(q, s);
      f[q] = val;
    }
  }
  o0 = vec4(f[0],f[1],f[2],f[3]); o1 = vec4(f[4],f[5],f[6],f[7]); o2 = vec4(f[8],0.,0.,0.);
}`;

// momentum exchange on each boundary node (after the stream pass)
const FS_FORCE = GLSL_HEAD + LBM_CONST + `
uniform sampler2D P0, P1, P2, F0, F1, F2, SOL, D1, D2, FM;
uniform vec2 ref;
uniform int flapOnly; // 1: only the links that end on the flap (bit mask in FM), against the free-stream static pressure
uniform float rhoB;
out vec4 o;
float Pq(int q, ivec2 p){ if (q < 4) return texelFetch(P0,p,0)[q]; if (q < 8) return texelFetch(P1,p,0)[q-4]; return texelFetch(P2,p,0).x; }
float Fq(int q, ivec2 p){ if (q < 4) return texelFetch(F0,p,0)[q]; if (q < 8) return texelFetch(F1,p,0)[q-4]; return texelFetch(F2,p,0).x; }
void main(){
  ivec2 p = ivec2(gl_FragCoord.xy);
  o = vec4(0.);
  if (texelFetch(SOL,p,0).r > 0.5) return;
  vec4 d1 = texelFetch(D1,p,0), d2 = texelFetch(D2,p,0);
  float dd[9] = float[9](0., d1.x, d1.y, d1.z, d1.w, d2.x, d2.y, d2.z, d2.w);
  vec3 acc = vec3(0.);
  int fm = flapOnly == 1 ? int(texelFetch(FM,p,0).r * 255.0 + 0.5) : 0;
  if (flapOnly == 1 && fm == 0) return;
  for (int a = 1; a < 9; a++) {
    float d = dd[a]; if (d <= 0.0) continue;
    if (flapOnly == 1 && ((fm >> (a - 1)) & 1) == 0) continue;
    float m = Pq(a, p) + Fq(OPPQ[a], p);
    if (flapOnly == 1) m -= 2.0 * WQ[a] * rhoB;
    vec2 fc = vec2(CQ[a]) * m;
    vec2 r = vec2(p) + d*vec2(CQ[a]) - ref;
    acc += vec3(fc, r.x*fc.y - r.y*fc.x);
  }
  o = vec4(acc, 1.0);
}`;

// derived fields: velocity (filterable) and the scalar being shown, with a validity flag
const FS_SCALAR = GLSL_HEAD + `
uniform sampler2D MAC, SOL;
uniform int mode, nx, ny;
uniform float invU, cpq, rhoRef, vortS;
layout(location=0) out vec4 oV; layout(location=1) out vec4 oS;
bool sol(ivec2 p){ return texelFetch(SOL,p,0).r > 0.5; }
void main(){
  ivec2 p = ivec2(gl_FragCoord.xy);
  vec4 m = texelFetch(MAC,p,0);
  bool s = sol(p);
  oV = s ? vec4(0.,0.,1.,1.) : vec4(m.y, m.z, m.x, 0.);
  if (s) { oS = vec4(0.,0.,0.,0.); return; }
  float val;
  if (mode == 1) {
    ivec2 xl = (p.x > 0 && !sol(p-ivec2(1,0))) ? p-ivec2(1,0) : p, xr = (p.x < nx-1 && !sol(p+ivec2(1,0))) ? p+ivec2(1,0) : p;
    ivec2 yd = (p.y > 0 && !sol(p-ivec2(0,1))) ? p-ivec2(0,1) : p, yu = (p.y < ny-1 && !sol(p+ivec2(0,1))) ? p+ivec2(0,1) : p;
    float hx = max(1.0, float(xr.x-xl.x)), hy = max(1.0, float(yu.y-yd.y));
    val = ((texelFetch(MAC,xr,0).z - texelFetch(MAC,xl,0).z)/hx - (texelFetch(MAC,yu,0).y - texelFetch(MAC,yd,0).y)/hy) * vortS;
  } else if (mode == 2) {
    val = (m.x - rhoRef) * cpq;
  } else {
    val = length(m.yz) * invU;
  }
  oS = vec4(val, 1., 0., 0.);
}`;
// fill solid cells from fluid neighbours so upscaling does not bleed a false value into the wall
const FS_FILL = GLSL_HEAD + `
uniform sampler2D SC; uniform int nx, ny, smooth3;
out vec4 o;
void main(){
  ivec2 p = ivec2(gl_FragCoord.xy);
  vec4 c = texelFetch(SC,p,0);
  if (c.y > 0.5) {
    if (smooth3 == 0) { o = c; return; }
    float s = 0., w = 0.;
    for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++) {
      vec4 t = texelFetch(SC, clamp(p + ivec2(i,j), ivec2(0), ivec2(nx-1,ny-1)), 0);
      float wt = t.y * (i == 0 ? 2.0 : 1.0) * (j == 0 ? 2.0 : 1.0);
      s += t.x*wt; w += wt;
    }
    o = vec4(s/w, 1., 0., 0.); return;
  }
  float s = 0., w = 0.;
  for (int j = -2; j <= 2; j++) for (int i = -2; i <= 2; i++) {
    ivec2 q = clamp(p + ivec2(i,j), ivec2(0), ivec2(nx-1,ny-1));
    vec4 t = texelFetch(SC,q,0);
    float wt = t.y / (1.0 + float(i*i + j*j));
    s += t.x*wt; w += wt;
  }
  o = vec4(w > 0.0 ? s/w : 0.0, 0., 0., 0.);
}`;

// passive scalars (R: smoke, G: heat) on a finer grid, MacCormack advection
const FS_ADV = GLSL_HEAD + `
uniform sampler2D Q, VEL; uniform vec2 dtUV;
out vec4 o;
void main(){
  vec2 uv = gl_FragCoord.xy / vec2(textureSize(Q,0));
  vec2 v1 = texture(VEL, uv).xy;
  vec2 mid = uv - 0.5*v1*dtUV;
  vec2 v2 = texture(VEL, mid).xy;
  o = texture(Q, uv - v2*dtUV);
}`;
const FS_CORR = GLSL_HEAD + `
uniform sampler2D Q, FWD, BWD, VEL, SOLF;
uniform vec2 dtUV;
uniform float rakeSp, rakeOff, rakeX0, rakeX1, rakeY0, rakeY1, heatOn, decay, lineW;
uniform vec2 kap;
out vec4 o;
void main(){
  ivec2 p = ivec2(gl_FragCoord.xy);
  vec2 sz = vec2(textureSize(Q,0));
  vec2 uv = gl_FragCoord.xy / sz;
  vec4 q = texelFetch(Q,p,0), f = texelFetch(FWD,p,0), b = texelFetch(BWD,p,0);
  vec4 r = f + 0.5*(q - b);
  // clamp to the range of the four cells around the departure point (keeps it monotone)
  vec2 v1 = texture(VEL, uv).xy; vec2 v2 = texture(VEL, uv - 0.5*v1*dtUV).xy;
  vec2 dp = (uv - v2*dtUV)*sz - 0.5;
  ivec2 i0 = ivec2(floor(dp)); ivec2 mx = ivec2(sz) - 1;
  vec4 a = texelFetch(Q, clamp(i0,ivec2(0),mx),0), c = texelFetch(Q, clamp(i0+ivec2(1,0),ivec2(0),mx),0);
  vec4 d = texelFetch(Q, clamp(i0+ivec2(0,1),ivec2(0),mx),0), e = texelFetch(Q, clamp(i0+ivec2(1,1),ivec2(0),mx),0);
  r = clamp(r, min(min(a,c),min(d,e)), max(max(a,c),max(d,e)));
  // diffusion of the advected field: Laplacian taken at the departure point (stable for kap < 0.25)
  vec2 dep = uv - v2*dtUV, tx = 1.0/sz;
  vec4 Lp = texture(Q, dep + vec2(tx.x,0.)) + texture(Q, dep - vec2(tx.x,0.)) + texture(Q, dep + vec2(0.,tx.y)) + texture(Q, dep - vec2(0.,tx.y)) - 4.0*texture(Q, dep);
  r.rg += kap*Lp.rg;
  r.rg = max(r.rg, vec2(0.0));
  r *= decay;
  float solid = texture(SOLF, uv).r;
  // smoke rake: thin filaments released just downstream of the inlet
  float px = gl_FragCoord.x;
  float py = gl_FragCoord.y;
  if (px < rakeX1 && px > max(1.0, rakeX0) && py > rakeY0 && py < rakeY1) {
    float y = py - rakeY0 - rakeOff; float k = mod(y, rakeSp); float dist = min(k, rakeSp - k);
    r.r = max(r.r, exp(-dist*dist/(lineW*lineW)));
  }
  r.r *= (1.0 - smoothstep(0.35, 0.6, solid));
  if (heatOn > 0.5) r.g = max(r.g, smoothstep(0.05, 0.6, solid));
  r.g = clamp(r.g, 0.0, 1.0);
  o = r;
}`;

// knife-edge input: smoothed temperature gradient at dye resolution (Sobel operator)
const FS_GRAD = GLSL_HEAD + `
uniform sampler2D Q;
out vec4 o;
void main(){
  ivec2 p = ivec2(gl_FragCoord.xy); ivec2 mx = textureSize(Q,0) - 1;
  // derivative-of-Gaussian filter, sigma = 1.5 px: smooth, isotropic gradient of the heat field
  vec2 g = vec2(0.); float t0 = 0., w0 = 0.;
  for (int j = -4; j <= 4; j++) for (int i = -4; i <= 4; i++) {
    float r2 = float(i*i + j*j); if (r2 > 18.0) continue;
    float w = exp(-r2 / 4.5);
    float T = texelFetch(Q, clamp(p + ivec2(i,j), ivec2(0), mx), 0).g;
    g += vec2(float(i), float(j)) * w * T; t0 += w*T; w0 += w;
  }
  o = vec4(g / (w0 * 2.25), t0 / w0, 0.0);
}`;


// probes: bilinear sample of (rho, u, v) at each probe position (fine-grid cell coordinates), solid nodes excluded
const FS_PROBE = GLSL_HEAD + `
uniform sampler2D MAC, SOL, PRB; uniform int nx, ny;
out vec4 o;
void main(){
  vec2 p = texelFetch(PRB, ivec2(int(gl_FragCoord.x), 0), 0).xy;
  vec2 f = floor(p); vec2 t = p - f; ivec2 b = ivec2(f);
  vec4 s = vec4(0.); float w = 0.;
  for (int j = 0; j < 2; j++) for (int i = 0; i < 2; i++) {
    ivec2 q = clamp(b + ivec2(i, j), ivec2(0), ivec2(nx - 1, ny - 1));
    if (texelFetch(SOL, q, 0).r > 0.5) continue;
    float wt = (i == 0 ? 1.0 - t.x : t.x) * (j == 0 ? 1.0 - t.y : t.y);
    s += wt * texelFetch(MAC, q, 0); w += wt;
  }
  o = w > 1e-6 ? vec4(s.xyz / w, 1.0) : vec4(1.0, 0.0, 0.0, 0.0);
}`;
// frame capture for recordings: vorticity as 16-bit fixed point (R high byte, G low byte), smoke in B, heat in A
const FS_CAPTURE = GLSL_HEAD + `
uniform sampler2D SCV, DYE; uniform vec2 outSize; uniform float vScale;
out vec4 o;
void main(){
  vec2 uv = gl_FragCoord.xy / outSize;
  float w = texture(SCV, uv).x;
  float v = clamp(floor(w * vScale + 32768.5), 0.0, 65535.0);
  float hi = floor(v / 256.0), lo = v - hi * 256.0;
  vec2 h = 0.25 / outSize;
  vec4 d = 0.25 * (texture(DYE, uv + vec2(-h.x, -h.y)) + texture(DYE, uv + vec2(h.x, -h.y)) + texture(DYE, uv + vec2(-h.x, h.y)) + texture(DYE, uv + vec2(h.x, h.y)));
  o = vec4(hi / 255.0, lo / 255.0, clamp(d.r, 0.0, 1.0), clamp(d.g, 0.0, 1.0));
}`;


// field statistics while recording: running sums at the probe cadence (shifted values keep float32 sums accurate)
const FS_ACC = GLSL_HEAD + `
uniform sampler2D MAC, SOL, A0, A1, A2; uniform int nx, ny; uniform float u0, wScale;
layout(location=0) out vec4 o0; layout(location=1) out vec4 o1; layout(location=2) out vec4 o2;
bool sol(ivec2 p){ return texelFetch(SOL,p,0).r > 0.5; }
void main(){
  ivec2 p = ivec2(gl_FragCoord.xy);
  vec4 a0 = texelFetch(A0,p,0), a1 = texelFetch(A1,p,0), a2 = texelFetch(A2,p,0);
  if (sol(p)) { o0 = a0; o1 = a1; o2 = a2; return; }
  vec4 m = texelFetch(MAC,p,0);
  ivec2 xl = (p.x > 0 && !sol(p-ivec2(1,0))) ? p-ivec2(1,0) : p, xr = (p.x < nx-1 && !sol(p+ivec2(1,0))) ? p+ivec2(1,0) : p;
  ivec2 yd = (p.y > 0 && !sol(p-ivec2(0,1))) ? p-ivec2(0,1) : p, yu = (p.y < ny-1 && !sol(p+ivec2(0,1))) ? p+ivec2(0,1) : p;
  float hx = max(1.0, float(xr.x-xl.x)), hy = max(1.0, float(yu.y-yd.y));
  float w = ((texelFetch(MAC,xr,0).z - texelFetch(MAC,xl,0).z)/hx - (texelFetch(MAC,yu,0).y - texelFetch(MAC,yd,0).y)/hy) * wScale;
  float du = m.y - u0, v = m.z, dr = m.x - 1.0;
  o0 = a0 + vec4(du, v, du*du, v*v);
  o1 = a1 + vec4(dr, dr*dr, w, w*w);
  o2 = a2 + vec4(du*v, 1.0, 0.0, 0.0);
}`;

// readback grid: block average of the macroscopic fields over k x k fluid cells
const FS_DOWN = GLSL_HEAD + `
uniform sampler2D MAC, SOL; uniform int k;
out vec4 o;
void main(){
  ivec2 p = ivec2(gl_FragCoord.xy) * k;
  vec4 s = vec4(0.); float n = 0.;
  for (int j = 0; j < 4; j++) for (int i = 0; i < 4; i++) {
    if (i >= k || j >= k) continue;
    ivec2 q = p + ivec2(i,j);
    if (texelFetch(SOL,q,0).r > 0.5) continue;
    s += texelFetch(MAC,q,0); n += 1.;
  }
  o = n > 0. ? vec4(s.xyz/n, n/float(k*k)) : vec4(1.,0.,0.,0.);
}`;

// final image
const FS_DISPLAY = GLSL_HEAD + `
uniform sampler2D SC, DYE, LUT, SOLF, VEL, GRD, PSN;
uniform int usePsn;
uniform float cpCap;
uniform int mode, contours, divMap, deadZero;
uniform vec2 canvas, grid;
uniform float lo, hi, mid, cstep, contrast, frame, schK;
uniform vec2 aperC; uniform float aperR;
uniform vec2 laser, bodyY;
out vec4 o;
float hash(vec2 p){ p = fract(p*vec2(443.897, 441.423)); p += dot(p, p.yx+19.19); return fract((p.x+p.y)*p.x); }
float cr(float t, float a, float b, float c, float d){ return b + 0.5*t*(c-a + t*(2.*a-5.*b+4.*c-d + t*(3.*(b-c)+d-a))); }
float scAt(ivec2 q){ return texelFetch(SC, clamp(q, ivec2(0), ivec2(grid)-1), 0).x; }
float bicubic(vec2 g){
  vec2 f = floor(g); vec2 t = g - f; ivec2 b = ivec2(f);
  float r[4];
  for (int j = 0; j < 4; j++) {
    ivec2 q = b + ivec2(-1, j-1);
    r[j] = cr(t.x, scAt(q), scAt(q+ivec2(1,0)), scAt(q+ivec2(2,0)), scAt(q+ivec2(3,0)));
  }
  return cr(t.y, r[0], r[1], r[2], r[3]);
}
vec3 lut(float t){ return texture(LUT, vec2(clamp(t,0.,1.)*0.996+0.002, 0.5)).rgb; }
void main(){
  vec2 uv = gl_FragCoord.xy / canvas;
  vec2 g = uv*grid - 0.5;
  float grain = (hash(gl_FragCoord.xy + frame*17.13) - 0.5);
  if (mode == 3) { // smoke in a dark tunnel, lit by a light sheet
    float d = texture(DYE, uv).r * contrast;
    float dens = 1.0 - exp(-2.6*d);
    float light = 0.82 + 0.18*smoothstep(0.0, 1.0, uv.y);
    vec3 bg = vec3(0.020, 0.024, 0.031) + 0.02*vec3(uv.x*0.4);
    vec3 col = mix(bg, vec3(0.93, 0.95, 0.97)*light, dens);
    col += grain*0.025;
    float vig = 1.0 - 0.28*pow(length((uv-0.5)*vec2(1.0, 1.3)), 2.0);
    o = vec4(col*vig, 1.0); return;
  }
  if (mode == 4) { // knife-edge schlieren of the heated model through a circular mirror
    vec2 px = gl_FragCoord.xy;
    float r = length(px - aperC) / aperR;
    vec3 gt = texture(GRD, uv).xyz;
    float scaleG = float(textureSize(GRD,0).y) / 640.0;
    float beta = 0.35, den = 1.0 + beta*gt.z;
    float drho = -beta * gt.y * scaleG / (den*den);
    float I = clamp(0.5 - schK*contrast*drho*4.0, 0.0, 1.0);
    // real optics: slightly uneven knife-edge cutoff, faint fringes near the mirror rim, dust on the mirror
    vec2 q = (px - aperC) / aperR;
    I *= 1.0 + 0.045*q.y + 0.02*q.x + 0.015*sin(3.1*q.x + 1.3) * cos(2.3*q.y);
    I *= 1.0 - 0.035*smoothstep(0.82, 0.99, r) * (0.5 + 0.5*cos(r*95.0));
    for (int k = 0; k < 9; k++) {
      vec2 c = vec2(hash(vec2(float(k), 3.7)), hash(vec2(float(k), 9.1))) * 1.4 - 0.7;
      float rad = 0.006 + 0.012*hash(vec2(float(k), 5.3));
      float dd = length(q - c);
      I *= 1.0 - 0.28*exp(-dd*dd/(rad*rad)) + 0.1*exp(-(dd-rad*1.6)*(dd-rad*1.6)/(rad*rad*0.3));
    }
    I *= 1.0 - 0.35*r*r;
    vec3 col = vec3(I)*vec3(0.98, 1.0, 0.97) + grain*0.03;
    float edge = smoothstep(1.0, 0.985, r);
    o = vec4(mix(vec3(0.035,0.038,0.042), col, edge), 1.0); return;
  }
  if (mode == 5) { // PIV raw-image background, laser sheet from above with the model's shadow
    // march toward the light source only through the band of heights the model occupies
    float sh = 0.0;
    if (uv.y < bodyY.y) {
      vec2 dir = laser - uv;
      float t0 = max(0.0, (bodyY.x - uv.y) / dir.y), t1 = (bodyY.y - uv.y) / dir.y;
      for (int s = 0; s <= 48; s++) {
        vec2 q = uv + dir * mix(t0, t1, float(s)/48.0);
        if (texture(SOLF, q).r > 0.5) { sh = 1.0; break; }
      }
    }
    float sheet = 0.55 + 0.45*smoothstep(0.0, 1.0, uv.y);
    uvec2 ip = uvec2(gl_FragCoord.xy);
    uint hh = ip.x * 1973u + ip.y * 9277u + uint(frame * 0.5) * 26699u;
    hh = (hh ^ (hh >> 13u)) * 1274126177u; hh ^= hh >> 16u; hh *= 668265263u; hh ^= hh >> 15u;
    float n = float(hh & 65535u) / 65535.0;
    float spk = n > 0.997 ? 0.16 : n > 0.97 ? 0.035 : 0.0; // sensor hot pixels and read noise
    float b = (0.035 + spk) * sheet * (1.0 - 0.8*sh);
    o = vec4(vec3(b*0.92, b, b*0.96), 1.0); return;
  }
  float v = bicubic(g);
  if (mode == 2 && usePsn == 1) { vec2 ps = texture(PSN, uv).rg; v = mix(ps.r, v, ps.g); } // hydrodynamic pressure away from the model
  if (mode == 2) v = min(v, cpCap);
  float t;
  if (divMap == 1) t = v < mid ? 0.5*(v - lo)/(mid - lo) : 0.5 + 0.5*(v - mid)/(hi - mid);
  else t = (v - lo)/(hi - lo);
  vec3 col = lut(t);
  if (contours == 1) {
    float f = v / cstep;
    float w = fwidth(f);
    float dl = abs(fract(f + 0.5) - 0.5) / max(w, 1e-5);
    float a = 1.0 - smoothstep(0.4, 1.3, dl);
    if (deadZero == 1 && abs(v) < 0.5*cstep) a = 0.0;
    float lum = dot(col, vec3(0.3,0.59,0.11));
    col = mix(col, lum > 0.55 ? vec3(0.08,0.1,0.12) : vec3(0.95), a*0.55);
  }
  o = vec4(col, 1.0);
}`;

class GPUTunnel {
  static probe() {
    try {
      const c = document.createElement('canvas');
      const gl = c.getContext('webgl2');
      if (!gl) return false;
      if (!gl.getExtension('EXT_color_buffer_float')) return false;
      return true;
    } catch (e) { return false; }
  }
  constructor(canvas) {
    const gl = this.gl = canvas.getContext('webgl2', { antialias: false, alpha: false, depth: false, stencil: false, premultipliedAlpha: false, preserveDrawingBuffer: true, powerPreference: 'high-performance' });
    if (!gl) throw new Error('no webgl2');
    if (!gl.getExtension('EXT_color_buffer_float')) throw new Error('no float render targets');
    this.canvas = canvas;
    const mk = (fs) => this.program(fs);
    this.P = { init: mk(FS_INIT), collide: mk(FS_COLLIDE), stream: mk(FS_STREAM), force: mk(FS_FORCE), scalar: mk(FS_SCALAR), fill: mk(FS_FILL), adv: mk(FS_ADV), corr: mk(FS_CORR), down: mk(FS_DOWN), display: mk(FS_DISPLAY), grad: mk(FS_GRAD), probe: mk(FS_PROBE), capture: mk(FS_CAPTURE), acc: mk(FS_ACC) };
    this.vao = gl.createVertexArray();
    this.t = 0; this.u0 = 0.09; this.tau0 = 0.52; this.Cs = 0.16; this.sigma = 0.95; this.cVreman = 2.5 * 0.25 * 0.25; this.smagC = 0.08; this.walls = 'solid'; this.turb = 0; this.rampSteps = 1000;
    this.sgsA = 2.4; this.sgsB = 9.6; this.fx0 = 24; this.fw = 5;
    this.lutTex = this.tex(256, 1, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, gl.LINEAR);
    this.lutName = '';
    this.dyeAcc = 0;
  }
  program(fs) {
    const gl = this.gl;
    const sh = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s); if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) + '\n' + src.split('\n').slice(0, 3).join(' ')); return s; };
    const p = gl.createProgram(); gl.attachShader(p, sh(gl.VERTEX_SHADER, VS)); gl.attachShader(p, sh(gl.FRAGMENT_SHADER, fs)); gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    const u = {}; const n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < n; i++) { const info = gl.getActiveUniform(p, i); const name = info.name.replace(/\[0\]$/, ''); u[name] = gl.getUniformLocation(p, info.name); }
    return { p, u };
  }
  tex(w, h, ifmt, fmt, type, filter, data) {
    const gl = this.gl, t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texImage2D(gl.TEXTURE_2D, 0, ifmt, w, h, 0, fmt, type, data || null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return t;
  }
  fbo(texs) {
    const gl = this.gl, f = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, f);
    texs.forEach((t, i) => gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i, gl.TEXTURE_2D, t, 0));
    gl.drawBuffers(texs.map((_, i) => gl.COLOR_ATTACHMENT0 + i));
    const st = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (st !== gl.FRAMEBUFFER_COMPLETE) throw new Error('framebuffer incomplete ' + st);
    return f;
  }
  configure(nx, ny, k) {
    const gl = this.gl;
    this.dispose();
    this.nx = nx; this.ny = ny; this.k = k; this.N = nx * ny;
    const F32 = [gl.RGBA32F, gl.RGBA, gl.FLOAT, gl.NEAREST];
    const T = () => this.tex(nx, ny, ...F32);
    this.F = [T(), T(), T()]; this.Pt = [T(), T(), T()]; this.MACs = [T(), T()]; this.mi = 0;
    this.FORCE = T();
    this.VEL = this.tex(nx, ny, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT, gl.LINEAR);
    this.SC = T(); this.SC2 = T();
    this.SOL = this.tex(nx, ny, gl.R8, gl.RED, gl.UNSIGNED_BYTE, gl.LINEAR);
    this.FM = this.tex(nx, ny, gl.R8, gl.RED, gl.UNSIGNED_BYTE, gl.NEAREST); // per cell: which of its wall links end on the flap
    this.SGSF = this.tex(nx, ny, gl.R8, gl.RED, gl.UNSIGNED_BYTE, gl.NEAREST, new Uint8Array(nx * ny).fill(255));
    this.sgs8 = new Uint8Array(nx * ny).fill(255); this.sgsBox = null;
    this.modes = turbModes(ny, 12345);
    this.D1 = T(); this.D2 = T();
    this.solid = new Uint8Array(nx * ny); this.dist = [new Float32Array(nx * ny * 4), new Float32Array(nx * ny * 4)];
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.bindTexture(gl.TEXTURE_2D, this.SOL); gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, nx, ny, gl.RED, gl.UNSIGNED_BYTE, this.solid);
    for (const [t, d] of [[this.D1, this.dist[0]], [this.D2, this.dist[1]]]) { gl.bindTexture(gl.TEXTURE_2D, t); gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, nx, ny, gl.RGBA, gl.FLOAT, d); }
    this.fbF = this.fbo(this.F); this.fbPs = this.MACs.map(m => this.fbo([...this.Pt, m]));
    this.fbForce = this.fbo([this.FORCE]); this.fbScalar = this.fbo([this.VEL, this.SC]); this.fbFill = this.fbo([this.SC2]);
    this.fbMacs = this.MACs.map(m => this.fbo([m]));
    // dye grid: up to 2x finer than the lattice
    const ds = nx * ny * 4 <= 1.5e6 ? 2 : nx * ny * 2.25 <= 1.5e6 ? 1.5 : 1;
    this.dw = Math.round(nx * ds); this.dh = Math.round(ny * ds); this.ds = ds;
    const D = () => this.tex(this.dw, this.dh, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT, gl.LINEAR);
    this.DYE = [D(), D()]; this.DF = D(); this.DB = D(); this.GRD = D(); this.fbGrd = this.fbo([this.GRD]);
    this.fbDye = this.DYE.map(t => this.fbo([t])); this.fbDF = this.fbo([this.DF]); this.fbDB = this.fbo([this.DB]);
    this.dyeI = 0;
    this.nxr = nx / k; this.nyr = ny / k;
    this.DOWN = this.tex(this.nxr, this.nyr, ...F32); this.fbDown = this.fbo([this.DOWN]);
    this.down = new Float32Array(this.nxr * this.nyr * 4);
    this.box = null; this.links = 0;
    this.reset();
  }
  dispose() {
    const gl = this.gl;
    for (const k of ['F', 'Pt', 'DYE', 'MACs']) if (this[k]) this[k].forEach(t => gl.deleteTexture(t));
    for (const k of ['SGSF', 'GRD', 'FORCE', 'VEL', 'SC', 'SC2', 'SOL', 'FM', 'D1', 'D2', 'DF', 'DB', 'DOWN']) if (this[k]) gl.deleteTexture(this[k]);
    for (const k of ['fbGrd', 'fbF', 'fbForce', 'fbScalar', 'fbFill', 'fbDF', 'fbDB', 'fbDown']) if (this[k]) gl.deleteFramebuffer(this[k]);
    for (const k of ['fbDye', 'fbPs', 'fbMacs']) if (this[k]) this[k].forEach(f => gl.deleteFramebuffer(f));
  }
  reset() {
    const gl = this.gl;
    this.draw(this.P.init, this.fbF, this.nx, this.ny, {});
    for (const f of this.fbMacs) { gl.bindFramebuffer(gl.FRAMEBUFFER, f); gl.viewport(0, 0, this.nx, this.ny); gl.clearColor(1, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT); }
    for (const f of [...this.fbDye, this.fbDF, this.fbDB]) { gl.bindFramebuffer(gl.FRAMEBUFFER, f); gl.viewport(0, 0, this.dw, this.dh); gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT); }
    this.t = 0;
  }
  // test/diagnostic hook: load populations laid out as f[q * N + i]
  loadState(f, t) {
    const gl = this.gl, N = this.N, A = [new Float32Array(N * 4), new Float32Array(N * 4), new Float32Array(N * 4)];
    for (let q = 0; q < 9; q++) { const a = A[q >> 2], c = q & 3; for (let i = 0; i < N; i++) a[i * 4 + c] = f[q * N + i]; }
    A.forEach((a, k) => { gl.bindTexture(gl.TEXTURE_2D, this.F[k]); gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, this.nx, this.ny, gl.RGBA, gl.FLOAT, a); });
    this.t = t;
  }
  readDye() { const gl = this.gl, b = new Float32Array(this.dw * this.dh * 4); gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbDye[this.dyeI]); gl.readPixels(0, 0, this.dw, this.dh, gl.RGBA, gl.FLOAT, b); return b; }
  writeDye(b) { const gl = this.gl; for (const t of this.DYE) { gl.bindTexture(gl.TEXTURE_2D, t); gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, this.dw, this.dh, gl.RGBA, gl.FLOAT, b); } }
  clearDye() { const gl = this.gl; for (const f of this.fbDye) { gl.bindFramebuffer(gl.FRAMEBUFFER, f); gl.viewport(0, 0, this.dw, this.dh); gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT); } }
  setFlow(u0, Re, Lref, turb) {
    this.u0 = u0; this.Re = Re; this.tau0 = Math.max(0.5015, 3 * u0 * Lref / Re + 0.5); this.turb = turb; this.nu = (this.tau0 - 0.5) / 3;
    this.sgsA = Math.max(1.5, 0.03 * Lref); this.sgsB = Math.max(4, 0.12 * Lref);
    this.fx0 = Math.round(0.3 * Lref); this.fw = Math.max(3, 0.06 * Lref);
    this.sgsFloor = Math.min(1, Math.max(0, (u0 / this.nu - 80) / 60));
  }
  setBody(poly, refX, refY) {
    const gl = this.gl, { nx, ny } = this;
    const geo = bodyGeometry(poly, nx, ny, this.sgsB);
    // sub-grid factor next to the model
    const S8 = this.sgs8, A = this.sgsA, Bv = this.sgsB;
    let ux0 = 1e9, uy0 = 1e9, ux1 = -1, uy1 = -1;
    const grow = (B) => { ux0 = Math.min(ux0, B.x0); uy0 = Math.min(uy0, B.y0); ux1 = Math.max(ux1, B.x0 + B.w - 1); uy1 = Math.max(uy1, B.y0 + B.h - 1); };
    if (this.sgsBox) { const B = this.sgsBox; for (let y = 0; y < B.h; y++) S8.fill(255, (B.y0 + y) * nx + B.x0, (B.y0 + y) * nx + B.x0 + B.w); grow(B); }
    if (geo.dist) {
      const D = geo.dist;
      for (let y = 0; y < D.h; y++) for (let x = 0; x < D.w; x++) S8[(D.y0 + y) * nx + D.x0 + x] = Math.round(255 * Math.min(1, Math.max(0, (D.d[y * D.w + x] - A) / (Bv - A))));
      this.sgsBox = D; grow(D);
    }
    if (ux1 >= 0) {
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      gl.pixelStorei(gl.UNPACK_ROW_LENGTH, nx); gl.pixelStorei(gl.UNPACK_SKIP_PIXELS, ux0); gl.pixelStorei(gl.UNPACK_SKIP_ROWS, uy0);
      gl.bindTexture(gl.TEXTURE_2D, this.SGSF); gl.texSubImage2D(gl.TEXTURE_2D, 0, ux0, uy0, ux1 - ux0 + 1, uy1 - uy0 + 1, gl.RED, gl.UNSIGNED_BYTE, S8);
      gl.pixelStorei(gl.UNPACK_ROW_LENGTH, 0); gl.pixelStorei(gl.UNPACK_SKIP_PIXELS, 0); gl.pixelStorei(gl.UNPACK_SKIP_ROWS, 0);
    }
    const old = this.box;
    // clear previous geometry
    if (old) for (let y = old.y0; y <= old.y1; y++) for (let x = old.x0; x <= old.x1; x++) {
      const i = y * nx + x; this.solid[i] = 0; this.dist[0].fill(0, i * 4, i * 4 + 4); this.dist[1].fill(0, i * 4, i * 4 + 4);
    }
    for (const i of geo.list) this.solid[i] = 255;
    const L = geo.links;
    for (let n = 0; n < L.n; n++) { const i = L.i[n], q = L.q[n]; const a = q <= 4 ? 0 : 1, c = (q - 1) % 4; this.dist[a][i * 4 + c] = L.d[n]; }
    const b = geo.box;
    const u = old ? { x0: Math.min(old.x0, b.x0), y0: Math.min(old.y0, b.y0), x1: Math.max(old.x1, b.x1), y1: Math.max(old.y1, b.y1) } : b;
    const x0 = Math.max(0, u.x0), y0 = Math.max(0, u.y0), w = Math.min(nx - 1, u.x1) - x0 + 1, h = Math.min(ny - 1, u.y1) - y0 + 1;
    gl.pixelStorei(gl.UNPACK_ROW_LENGTH, nx); gl.pixelStorei(gl.UNPACK_SKIP_PIXELS, x0); gl.pixelStorei(gl.UNPACK_SKIP_ROWS, y0);
    gl.bindTexture(gl.TEXTURE_2D, this.SOL); gl.texSubImage2D(gl.TEXTURE_2D, 0, x0, y0, w, h, gl.RED, gl.UNSIGNED_BYTE, this.solid);
    gl.bindTexture(gl.TEXTURE_2D, this.D1); gl.texSubImage2D(gl.TEXTURE_2D, 0, x0, y0, w, h, gl.RGBA, gl.FLOAT, this.dist[0]);
    gl.bindTexture(gl.TEXTURE_2D, this.D2); gl.texSubImage2D(gl.TEXTURE_2D, 0, x0, y0, w, h, gl.RGBA, gl.FLOAT, this.dist[1]);
    gl.pixelStorei(gl.UNPACK_ROW_LENGTH, 0); gl.pixelStorei(gl.UNPACK_SKIP_PIXELS, 0); gl.pixelStorei(gl.UNPACK_SKIP_ROWS, 0);
    this.box = b; this.links = L.n; this.ref = [refX, refY];
    // the flap's share: a bit per wall link that ends on the flap, so a second force pass can sum exactly those links
    // (a slot cell has links to both the shroud and the flap, with large opposing forces)
    let anyF = false; for (let k = 0; k < L.n; k++) if (L.e[k]) { anyF = true; break; }
    if (anyF || this.flapOn) {
      const M = new Uint8Array(w * h);
      if (anyF) for (let k = 0; k < L.n; k++) if (L.e[k]) {
        const i = L.i[k], x = i % nx - x0, y = ((i / nx) | 0) - y0; if (x >= 0 && y >= 0 && x < w && y < h) M[y * w + x] |= 1 << (L.q[k] - 1);
      }
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      gl.bindTexture(gl.TEXTURE_2D, this.FM); gl.texSubImage2D(gl.TEXTURE_2D, 0, x0, y0, w, h, gl.RED, gl.UNSIGNED_BYTE, M);
    }
    this.flapOn = anyF;
  }
  bind(prog, unis, texs) {
    const gl = this.gl; gl.useProgram(prog.p);
    let unit = 0;
    for (const [name, t] of Object.entries(texs)) { if (prog.u[name] == null) continue; gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, t); gl.uniform1i(prog.u[name], unit); unit++; }
    for (const [name, v] of Object.entries(unis)) {
      const loc = prog.u[name]; if (loc == null) continue;
      if (typeof v === 'number') gl.uniform1f(loc, v);
      else if (v.i !== undefined) gl.uniform1i(loc, v.i);
      else if (v.length === 2 && !(v instanceof Float32Array)) gl.uniform2f(loc, v[0], v[1]);
      else gl.uniform1fv(loc, v);
    }
  }
  draw(prog, fb, w, h, unis, texs, vp) {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    if (vp) { gl.viewport(0, 0, w, h); gl.enable(gl.SCISSOR_TEST); gl.scissor(vp[0], vp[1], vp[2], vp[3]); } else { gl.viewport(0, 0, w, h); gl.disable(gl.SCISSOR_TEST); }
    this.bind(prog, unis || {}, texs || {});
    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.disable(gl.SCISSOR_TEST);
  }
  step(n) {
    const { nx, ny } = this;
    const C = 18 * Math.SQRT2 * this.Cs * this.Cs;
    const spW = Math.max(8, Math.round(nx * 0.08)), inW = Math.max(6, Math.round(nx * 0.05));
    const I = (v) => ({ i: v });
    for (let s = 0; s < n; s++) {
      const prev = this.MACs[this.mi]; this.mi = 1 - this.mi;
      const rr = Math.min(1, this.t / this.rampSteps), rampS = rr * rr * (3 - 2 * rr);
      const M = this.modes;
      this.draw(this.P.collide, this.fbPs[this.mi], nx, ny, { tau0: this.tau0, cV: this.cVreman, Csm: 18 * Math.SQRT2 * this.smagC * this.smagC, nx, ny: I(ny), spW, inW, kb: 1 - 1 / (0.5 + 3 * 0.05), sig: this.sigma,
        sgsA: this.sgsA, sgsB: this.sgsB, sgsFloor: this.sgsFloor || 0, solidWalls: I(this.walls === 'solid' ? 1 : 0),
        fOn: this.turb > 0 ? 1 : 0, fx0: this.fx0, fw: this.fw, fAmp: 0.9 * this.turb * this.u0 * this.u0 * rampS / (1.7725 * this.fw), Uc: this.u0, tt: this.t,
        mkx: M.kx, mky: M.ky, mk: M.k, mam: M.am, mph: M.ph, mom: M.om },
        { F0: this.F[0], F1: this.F[1], F2: this.F[2], SOL: this.SOL, MACP: prev, SGSF: this.SGSF });
      const r = Math.min(1, this.t / this.rampSteps), uIn = this.u0 * r * r * (3 - 2 * r);
      this.draw(this.P.stream, this.fbF, nx, ny, { nx: I(nx), ny: I(ny), freeSlip: I(this.walls === 'solid' ? 0 : 1), solidWalls: I(this.walls === 'solid' ? 1 : 0), uIn },
        { P0: this.Pt[0], P1: this.Pt[1], P2: this.Pt[2], SOL: this.SOL, MAC: this.MACs[this.mi], D1: this.D1, D2: this.D2 });
      this.t++;
    }
  }
  forces(rhoB) {
    const gl = this.gl, b = this.box; if (!b) return [0, 0, 0, 0, 0, 0];
    const x0 = Math.max(0, b.x0), y0 = Math.max(0, b.y0), w = Math.min(this.nx - 1, b.x1) - x0 + 1, h = Math.min(this.ny - 1, b.y1) - y0 + 1;
    this.draw(this.P.force, this.fbForce, this.nx, this.ny, { ref: this.ref, flapOnly: { i: 0 } },
      { P0: this.Pt[0], P1: this.Pt[1], P2: this.Pt[2], F0: this.F[0], F1: this.F[1], F2: this.F[2], SOL: this.SOL, D1: this.D1, D2: this.D2 }, [x0, y0, w, h]);
    const buf = this.fbuf && this.fbuf.length >= w * h * 4 ? this.fbuf : (this.fbuf = new Float32Array(w * h * 4));
    const sum = () => {
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbForce);
      gl.readPixels(x0, y0, w, h, gl.RGBA, gl.FLOAT, buf);
      let a = 0, c = 0, m = 0; for (let i = 0; i < w * h * 4; i += 4) { a += buf[i]; c += buf[i + 1]; m += buf[i + 2]; }
      return [a, c, m];
    };
    const [fx, fy, mz] = sum();
    let fxF = 0, fyF = 0, mzF = 0;
    if (this.flapOn) {
      this.draw(this.P.force, this.fbForce, this.nx, this.ny, { ref: this.ref, flapOnly: { i: 1 }, rhoB: rhoB || 1 },
        { P0: this.Pt[0], P1: this.Pt[1], P2: this.Pt[2], F0: this.F[0], F1: this.F[1], F2: this.F[2], SOL: this.SOL, D1: this.D1, D2: this.D2, FM: this.FM }, [x0, y0, w, h]);
      [fxF, fyF, mzF] = sum();
    }
    return [fx, fy, mz, fxF, fyF, mzF];
  }
  readDown() {
    const gl = this.gl;
    this.draw(this.P.down, this.fbDown, this.nxr, this.nyr, { k: { i: this.k } }, { MAC: this.MACs[this.mi], SOL: this.SOL });
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbDown);
    gl.readPixels(0, 0, this.nxr, this.nyr, gl.RGBA, gl.FLOAT, this.down);
    return this.down;
  }
  readMacBox(x0, y0, w, h) {
    const gl = this.gl;
    x0 = Math.max(0, x0); y0 = Math.max(0, y0); w = Math.min(this.nx - x0, w); h = Math.min(this.ny - y0, h);
    const buf = new Float32Array(w * h * 4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbMacs[this.mi]);
    gl.readPixels(x0, y0, w, h, gl.RGBA, gl.FLOAT, buf);
    return { x0, y0, w, h, buf };
  }
  fields(mode, rhoRef, Nc, uRef) {
    uRef = uRef || this.u0;
    const { nx, ny } = this, I = (v) => ({ i: v });
    this.draw(this.P.scalar, this.fbScalar, nx, ny, { mode: I(mode), nx: I(nx), ny: I(ny), invU: 1 / this.u0, cpq: 2 / (3 * uRef * uRef), rhoRef, vortS: Nc / this.u0 * 0.5 * 2 }, { MAC: this.MACs[this.mi], SOL: this.SOL });
    this.draw(this.P.fill, this.fbFill, nx, ny, { nx: I(nx), ny: I(ny), smooth3: I(mode === 1 || mode === 2 ? 1 : 0) }, { SC: this.SC });
  }
  advect(steps, opt) {
    if (steps <= 0) return;
    const kmax = Math.max(opt.kHeat, opt.kSmoke) * this.ds * this.ds;
    const sub = Math.max(1, Math.ceil(steps * this.u0 * 1.6 / 3), Math.ceil(steps * kmax / 0.22));
    const dt = steps / sub;
    const dtUV = [dt / this.nx, dt / this.ny];
    const kap = [opt.kSmoke * dt * this.ds * this.ds, opt.kHeat * dt * this.ds * this.ds];
    for (let s = 0; s < sub; s++) {
      const Q = this.DYE[this.dyeI], out = this.fbDye[1 - this.dyeI];
      this.draw(this.P.adv, this.fbDF, this.dw, this.dh, { dtUV }, { Q, VEL: this.VEL });
      this.draw(this.P.adv, this.fbDB, this.dw, this.dh, { dtUV: [-dtUV[0], -dtUV[1]] }, { Q: this.DF, VEL: this.VEL });
      const R = opt.rake, d = this.ds;
      this.draw(this.P.corr, out, this.dw, this.dh, { dtUV, rakeSp: R.sp * d, rakeOff: R.off * d, rakeX0: R.x0 * d, rakeX1: R.x1 * d, rakeY0: R.y0 * d, rakeY1: R.y1 * d, lineW: Math.max(0.5, R.w * d), heatOn: 1, kap, decay: Math.pow(0.99985, dt) },
        { Q, FWD: this.DF, BWD: this.DB, VEL: this.VEL, SOLF: this.SOL });
      this.dyeI = 1 - this.dyeI;
    }
  }

  // ----- probes -----
  setProbes(pts) { // pts: [[x, y], ...] in fine-grid cell coordinates
    const gl = this.gl, n = Math.max(1, pts.length);
    if (this.PRB) { gl.deleteTexture(this.PRB); gl.deleteTexture(this.PHIST); gl.deleteFramebuffer(this.fbPH); }
    const d = new Float32Array(n * 4); pts.forEach((p, i) => { d[i * 4] = p[0]; d[i * 4 + 1] = p[1]; });
    this.PRB = this.tex(n, 1, gl.RGBA32F, gl.RGBA, gl.FLOAT, gl.NEAREST, d);
    this.PROWS = 64; this.PHIST = this.tex(n, this.PROWS, gl.RGBA32F, gl.RGBA, gl.FLOAT, gl.NEAREST); this.fbPH = this.fbo([this.PHIST]);
    this.nProbes = pts.length; this.probeW = n;
  }
  sampleProbes(row) { // record the current probe values into one row of the history texture (no GPU stall)
    if (!this.PRB) return;
    this.draw(this.P.probe, this.fbPH, this.probeW, this.PROWS, { nx: { i: this.nx }, ny: { i: this.ny } }, { MAC: this.MACs[this.mi], SOL: this.SOL, PRB: this.PRB }, [0, row, this.probeW, 1]);
  }
  readProbes(rows) {
    const gl = this.gl, b = new Float32Array(this.probeW * rows * 4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbPH); gl.readPixels(0, 0, this.probeW, rows, gl.RGBA, gl.FLOAT, b);
    return b;
  }
  // ----- field statistics -----
  accReset() {
    const gl = this.gl;
    if (!this.ACC || this.accW !== this.nx || this.accH !== this.ny) {
      if (this.ACC) { this.ACC.flat().forEach(t => gl.deleteTexture(t)); this.fbAcc.forEach(f => gl.deleteFramebuffer(f)); }
      const T = () => this.tex(this.nx, this.ny, gl.RGBA32F, gl.RGBA, gl.FLOAT, gl.NEAREST);
      this.ACC = [[T(), T(), T()], [T(), T(), T()]]; this.fbAcc = this.ACC.map(s => this.fbo(s)); this.accW = this.nx; this.accH = this.ny;
    }
    for (const f of this.fbAcc) { gl.bindFramebuffer(gl.FRAMEBUFFER, f); gl.viewport(0, 0, this.nx, this.ny); gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT); }
    this.accI = 0;
  }
  accumulate(wScale) {
    if (!this.ACC) this.accReset();
    const src = this.ACC[this.accI], out = 1 - this.accI;
    this.draw(this.P.acc, this.fbAcc[out], this.nx, this.ny, { nx: { i: this.nx }, ny: { i: this.ny }, u0: this.u0, wScale },
      { MAC: this.MACs[this.mi], SOL: this.SOL, A0: src[0], A1: src[1], A2: src[2] });
    this.accI = out;
  }
  readAccRows(cb) { // accumulator in horizontal bands, so the read-back never needs the whole grid in memory at once
    const gl = this.gl, nx = this.nx, band = Math.max(1, Math.min(this.ny, Math.floor((1 << 18) / Math.max(1, nx))));
    const bufs = [0, 1, 2].map(() => new Float32Array(nx * band * 4));
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbAcc[this.accI]);
    for (let y0 = 0; y0 < this.ny; y0 += band) {
      const h = Math.min(band, this.ny - y0);
      for (let k = 0; k < 3; k++) { gl.readBuffer(gl.COLOR_ATTACHMENT0 + k); gl.readPixels(0, y0, nx, h, gl.RGBA, gl.FLOAT, bufs[k].subarray(0, nx * h * 4)); }
      cb(bufs[0], bufs[1], bufs[2], y0, h);
    }
    gl.readBuffer(gl.COLOR_ATTACHMENT0);
  }
  readAcc() {
    const gl = this.gl, N = this.nx * this.ny, out = [];
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbAcc[this.accI]);
    for (let k = 0; k < 3; k++) { const b = new Float32Array(4 * N); gl.readBuffer(gl.COLOR_ATTACHMENT0 + k); gl.readPixels(0, 0, this.nx, this.ny, gl.RGBA, gl.FLOAT, b); out.push(b); }
    gl.readBuffer(gl.COLOR_ATTACHMENT0);
    return out;
  }
  // ----- recording -----
  capture(cw, ch, vScale) { // call after fields(1, ...) so SC2 holds vorticity
    const gl = this.gl;
    if (!this.CAP || this.capW !== cw || this.capH !== ch) {
      if (this.CAP) { gl.deleteTexture(this.CAP); gl.deleteFramebuffer(this.fbCap); }
      this.CAP = this.tex(cw, ch, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, gl.NEAREST); this.fbCap = this.fbo([this.CAP]); this.capW = cw; this.capH = ch;
    }
    this.draw(this.P.capture, this.fbCap, cw, ch, { outSize: [cw, ch], vScale }, { SCV: this.SC2, DYE: this.DYE[this.dyeI] });
    const b = new Uint8Array(cw * ch * 4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbCap); gl.readPixels(0, 0, cw, ch, gl.RGBA, gl.UNSIGNED_BYTE, b);
    return b;
  }
  readPopulations() { // inverse of loadState: f[q * N + i], read in bands to keep the temporary buffer small
    const gl = this.gl, N = this.N, nx = this.nx, out = new Float32Array(9 * N);
    const band = Math.max(1, Math.min(this.ny, Math.floor((1 << 19) / Math.max(1, nx)))), tmp = new Float32Array(nx * band * 4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbF);
    for (let k = 0; k < 3; k++) {
      gl.readBuffer(gl.COLOR_ATTACHMENT0 + k);
      for (let y0 = 0; y0 < this.ny; y0 += band) {
        const h = Math.min(band, this.ny - y0), t = tmp.subarray(0, nx * h * 4), cells = nx * h;
        gl.readPixels(0, y0, nx, h, gl.RGBA, gl.FLOAT, t);
        for (let c = 0; c < 4; c++) { const q = 4 * k + c; if (q > 8) break; const o = q * N + y0 * nx; for (let i = 0; i < cells; i++) out[o + i] = t[i * 4 + c]; }
      }
    }
    gl.readBuffer(gl.COLOR_ATTACHMENT0);
    return out;
  }
  readDye8() { // smoke and heat as 8-bit pairs, for checkpoints
    const f = this.readDye(), n = this.dw * this.dh, b = new Uint8Array(n * 2);
    for (let i = 0; i < n; i++) { b[2 * i] = Math.max(0, Math.min(255, Math.round(f[4 * i] * 255))); b[2 * i + 1] = Math.max(0, Math.min(255, Math.round(f[4 * i + 1] * 255))); }
    return b;
  }
  writeDye8(b) {
    const n = this.dw * this.dh, f = new Float32Array(n * 4);
    for (let i = 0; i < n; i++) { f[4 * i] = b[2 * i] / 255; f[4 * i + 1] = b[2 * i + 1] / 255; }
    this.writeDye(f);
  }
  // ----- playback -----
  pbScalar(data, nx, ny) {
    const gl = this.gl;
    if (!this.PB_SC || this.pbSW !== nx || this.pbSH !== ny) {
      if (this.PB_SC) gl.deleteTexture(this.PB_SC);
      this.PB_SC = this.tex(nx, ny, gl.R32F, gl.RED, gl.FLOAT, gl.NEAREST); this.pbSW = nx; this.pbSH = ny;
    }
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.bindTexture(gl.TEXTURE_2D, this.PB_SC); gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, nx, ny, gl.RED, gl.FLOAT, data);
  }
  pbDye(data, w, h) { // two 8-bit channels: smoke, heat
    const gl = this.gl;
    if (!this.PB_DYE || this.pbDW !== w || this.pbDH !== h) {
      if (this.PB_DYE) { gl.deleteTexture(this.PB_DYE); gl.deleteTexture(this.PB_GRD); gl.deleteFramebuffer(this.fbPbGrd); }
      this.PB_DYE = this.tex(w, h, gl.RG8, gl.RG, gl.UNSIGNED_BYTE, gl.LINEAR);
      this.PB_GRD = this.tex(w, h, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT, gl.LINEAR); this.fbPbGrd = this.fbo([this.PB_GRD]);
      this.pbDW = w; this.pbDH = h;
    }
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.bindTexture(gl.TEXTURE_2D, this.PB_DYE); gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, w, h, gl.RG, gl.UNSIGNED_BYTE, data);
  }
  setPoisson(cp, w, nx, ny) {
    const gl = this.gl;
    if (!this.PSN || this.psnW !== nx || this.psnH !== ny) {
      if (this.PSN) gl.deleteTexture(this.PSN);
      this.PSN = this.tex(nx, ny, gl.RG16F, gl.RG, gl.HALF_FLOAT, gl.LINEAR); this.psnW = nx; this.psnH = ny; this.psnBuf = new Float32Array(nx * ny * 2);
    }
    const b = this.psnBuf; for (let i = 0; i < nx * ny; i++) { b[2 * i] = cp[i]; b[2 * i + 1] = w[i]; }
    gl.bindTexture(gl.TEXTURE_2D, this.PSN); gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, nx, ny, gl.RG, gl.FLOAT, b);
    this.psnOK = true;
  }
  setLUT(name, colors) {
    if (this.lutName === name) return;
    const gl = this.gl, d = new Uint8Array(256 * 4);
    colors.forEach((c, i) => { d[i * 4] = c[0]; d[i * 4 + 1] = c[1]; d[i * 4 + 2] = c[2]; d[i * 4 + 3] = 255; });
    gl.bindTexture(gl.TEXTURE_2D, this.lutTex); gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 256, 1, gl.RGBA, gl.UNSIGNED_BYTE, d);
    this.lutName = name;
  }
  display(o) {
    const gl = this.gl, I = (v) => ({ i: v });
    const W = this.canvas.width, H = this.canvas.height;
    const pb = o.pb; // playback: scalar and dye come from a recorded frame
    const SCt = pb ? this.PB_SC : this.SC2, grid = pb ? [this.pbSW, this.pbSH] : [this.nx, this.ny];
    const DYEt = pb && this.PB_DYE ? this.PB_DYE : this.DYE[this.dyeI], GRDt = pb && this.PB_DYE ? this.PB_GRD : this.GRD;
    if (o.mode === 4) {
      if (pb && this.PB_DYE) this.draw(this.P.grad, this.fbPbGrd, this.pbDW, this.pbDH, {}, { Q: this.PB_DYE });
      else this.draw(this.P.grad, this.fbGrd, this.dw, this.dh, {}, { Q: this.DYE[this.dyeI] });
    }
    this.draw(this.P.display, null, W, H, {
      mode: I(o.mode), contours: I(o.contours ? 1 : 0), divMap: I(o.div ? 1 : 0), deadZero: I(o.mode === 1 ? 1 : 0), canvas: [W, H], grid,
      lo: o.lo, hi: o.hi, mid: o.mid, cstep: o.cstep, contrast: o.contrast, frame: o.frame, schK: o.schK,
      aperC: o.aperC, aperR: o.aperR, laser: o.laser, bodyY: o.bodyY || [0, 0],
      usePsn: { i: !pb && this.psnOK && o.psn ? 1 : 0 }, cpCap: o.cpCap || 1.1,
    }, { SC: SCt, DYE: DYEt, LUT: this.lutTex, SOLF: this.SOL, VEL: this.VEL, GRD: GRDt, PSN: this.PSN || this.SOL });
  }
}
