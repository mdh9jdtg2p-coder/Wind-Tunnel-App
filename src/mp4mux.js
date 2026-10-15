// ===== Minimal MP4 (ISO BMFF) writer for one video track: H.264 ('avc1' + avcC) or VP9 ('vp09' + vpcC) =====
// Samples are encoded chunks as WebCodecs produces them (H.264 in AVCC length-prefixed form). The movie box is
// written before the media data so the file can start playing before it has fully loaded.
function mp4Mux(opt) {
  // opt: { codec: 'avc' | 'vp9', width, height, timescale, samples: [{ data: Uint8Array, dur, key, pts? }], avcC?: Uint8Array,
  //        vp9?: { profile, level, bitDepth }, colour?: { p, t, m, full } (ISO/IEC 23091-2 code points; BT.709 limited range by default) }
  const enc = new TextEncoder();
  const u8 = (n) => new Uint8Array([n & 255]);
  const u16 = (n) => new Uint8Array([(n >> 8) & 255, n & 255]);
  const u32 = (n) => new Uint8Array([(n >>> 24) & 255, (n >>> 16) & 255, (n >>> 8) & 255, n & 255]);
  const str = (s) => enc.encode(s);
  const zeros = (k) => new Uint8Array(k);
  const cat = (parts) => { let n = 0; for (const p of parts) n += p.length; const o = new Uint8Array(n); let k = 0; for (const p of parts) { o.set(p, k); k += p.length; } return o; };
  const box = (type, ...parts) => { const body = cat(parts); return cat([u32(body.length + 8), str(type), body]); };
  const full = (type, version, flags, ...parts) => box(type, u8(version), new Uint8Array([(flags >> 16) & 255, (flags >> 8) & 255, flags & 255]), ...parts);
  const matrix = cat([u32(0x00010000), u32(0), u32(0), u32(0), u32(0x00010000), u32(0), u32(0), u32(0), u32(0x40000000)]);

  const { width: W, height: H, timescale: TS, samples } = opt;
  const total = samples.reduce((a, s) => a + s.dur, 0), durMs = Math.round(total * 1000 / TS);

  // stts: run-length encoded sample durations
  const stts = []; for (const s of samples) { const last = stts[stts.length - 1]; if (last && last[1] === s.dur) last[0]++; else stts.push([1, s.dur]); }
  const keys = []; samples.forEach((s, i) => { if (s.key) keys.push(i + 1); });
  const col = opt.colour || { p: 1, t: 1, m: 1, full: false };
  // composition offsets, only if the encoder reordered frames (decode time is the running sum of durations)
  let dts = 0; const cto = samples.map(s => { const o = s.pts === undefined ? 0 : s.pts - dts; dts += s.dur; return o; });
  const shift = Math.max(0, -Math.min(0, ...cto)), reordered = cto.some(o => o !== 0);
  const ctts = []; if (reordered) for (const o of cto) { const v = o + shift, last = ctts[ctts.length - 1]; if (last && last[1] === v) last[0]++; else ctts.push([1, v]); }

  const compressor = zeros(32); const cname = opt.codec === 'avc' ? 'AVC Coding' : 'VP9 Coding'; compressor[0] = cname.length; compressor.set(str(cname), 1);
  const visualEntry = (type, config) => box(type, zeros(6), u16(1), u16(0), u16(0), zeros(12), u16(W), u16(H),
    u32(0x00480000), u32(0x00480000), u32(0), u16(1), compressor, u16(0x0018), u16(0xffff), config,
    box('colr', str('nclx'), u16(col.p), u16(col.t), u16(col.m), u8(col.full ? 0x80 : 0)), box('pasp', u32(1), u32(1)));
  let entry;
  if (opt.codec === 'avc') entry = visualEntry('avc1', box('avcC', opt.avcC));
  else {
    const v = opt.vp9 || { profile: 0, level: 40, bitDepth: 8 };
    // vpcC v1: profile, level, bitDepth(4) | chroma subsampling(3) = 1 (4:2:0 colocated) | full range(1) = 0, primaries, transfer, matrix (BT.709), no init data
    entry = visualEntry('vp09', full('vpcC', 1, 0, u8(v.profile), u8(v.level), u8((v.bitDepth << 4) | (1 << 1) | (col.full ? 1 : 0)), u8(col.p), u8(col.t), u8(col.m), u16(0)));
  }

  const moov = (mdatOffset) => box('moov',
    full('mvhd', 0, 0, u32(0), u32(0), u32(1000), u32(durMs), u32(0x00010000), u16(0x0100), zeros(10), matrix, zeros(24), u32(2)),
    box('trak',
      full('tkhd', 0, 3, u32(0), u32(0), u32(1), u32(0), u32(durMs), zeros(8), u16(0), u16(0), u16(0), u16(0), matrix, u32(W << 16), u32(H << 16)),
      ...(shift ? [box('edts', full('elst', 0, 0, u32(1), u32(durMs), u32(shift), u16(1), u16(0)))] : []),
      box('mdia',
        full('mdhd', 0, 0, u32(0), u32(0), u32(TS), u32(total), u16(0x55c4), u16(0)),
        full('hdlr', 0, 0, u32(0), str('vide'), zeros(12), str('VideoHandler'), u8(0)),
        box('minf',
          full('vmhd', 0, 1, u16(0), u16(0), u16(0), u16(0)),
          box('dinf', full('dref', 0, 0, u32(1), full('url ', 0, 1))),
          box('stbl',
            full('stsd', 0, 0, u32(1), entry),
            full('stts', 0, 0, u32(stts.length), ...stts.map(([c, d]) => cat([u32(c), u32(d)]))),
            full('stss', 0, 0, u32(keys.length), ...keys.map(k => u32(k))),
            ...(reordered ? [full('ctts', 0, 0, u32(ctts.length), ...ctts.map(([c, o]) => cat([u32(c), u32(o)])))] : []),
            full('stsc', 0, 0, u32(1), u32(1), u32(samples.length), u32(1)),
            full('stsz', 0, 0, u32(0), u32(samples.length), ...samples.map(s => u32(s.data.length))),
            full('stco', 0, 0, u32(1), u32(mdatOffset)))))));

  const ftyp = box('ftyp', str('isom'), u32(512), str('isom'), str('iso2'), str(opt.codec === 'avc' ? 'avc1' : 'iso6'), str('mp41'));
  const dataLen = samples.reduce((a, s) => a + s.data.length, 0);
  const probe = moov(0), offset = ftyp.length + probe.length + 8; // moov size does not depend on the offset value
  const head = cat([ftyp, moov(offset), u32(dataLen + 8), str('mdat')]);
  return new Blob([head, ...samples.map(s => s.data)], { type: 'video/mp4' });
}
if (typeof module !== 'undefined') module.exports = { mp4Mux };
