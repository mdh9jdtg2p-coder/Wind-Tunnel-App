#!/usr/bin/env python3
"""Build the installable web app from the sources in this folder.

Run from anywhere:  python3 src/build_webapp.py
It writes index.html, sw.js and manifest.webmanifest one level up (the site root), next to icons/.
The offline cache version is a hash of everything it caches, so each change you publish reaches
installed copies the next time they are opened online.
"""
import hashlib, json, os

SRC = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(SRC)
read = lambda f: open(os.path.join(SRC, f), encoding='utf-8').read()

ICONS = ['icons/icon-192.png', 'icons/icon-512.png', 'icons/maskable-192.png', 'icons/maskable-512.png', 'icons/apple-touch-icon.png', 'icons/favicon-32.png']

# ---- page: the same sources as the Claude version, as a complete document ----
features = read('features.js') + '\n' + read('exports.js') + '\n' + read('mp4mux.js')
page = read('app.html').replace('/*CORE*/', read('core.js')).replace('/*GPU*/', read('gpu.js')).replace(
    '/*MAIN*/', read('poisson.js') + '\n' + read('main.js').replace('/*FEATURES*/', features))
head, body = page.split('</style>', 1)
head = head.replace('<meta name="claude:viewport-fit-sides" content="cover">\n', '')

TOP = '''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="description" content="Two-dimensional wind tunnel: lattice-Boltzmann CFD of aerofoils, with probes, recordings and image and video export">
<meta name="theme-color" media="(prefers-color-scheme: light)" content="#e9edf0">
<meta name="theme-color" media="(prefers-color-scheme: dark)" content="#0f1418">
<link rel="manifest" href="manifest.webmanifest">
<link rel="icon" type="image/png" sizes="32x32" href="icons/favicon-32.png">
<link rel="apple-touch-icon" href="icons/apple-touch-icon.png">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-title" content="Wind Tunnel">
<meta name="apple-mobile-web-app-status-bar-style" content="default">
'''
STYLE = '''
/* installed app: clear the status bar and notch, and the update notice */
@media (display-mode: standalone) { body { padding-top: max(14px, env(safe-area-inset-top, 0px)); } }
.updbar { position: fixed; left: 50%; bottom: max(14px, env(safe-area-inset-bottom, 0px)); transform: translateX(-50%); z-index: 20;
  display: flex; align-items: center; gap: 10px; background: var(--ink); color: var(--bg); padding: 8px 8px 8px 14px; border-radius: var(--r);
  font-size: 13.5px; box-shadow: 0 6px 24px rgba(0,0,0,.25); max-width: calc(100% - 32px); }
.updbar button { min-height: 32px; padding: 4px 12px; }
'''
INSTALL_BTN = '\n      <button id="btnInstall" hidden>Install app</button>'
FOOT = '''  <p><b>App.</b> This page installs as an app and then works offline. In Chrome or Edge choose Install app, or the install icon in the address bar; on an iPhone or iPad tap Share, then Add to Home Screen; on Android use the browser menu. Recordings stay on the device and browser that made them. Like any web app, the tunnel pauses while it is not on screen.</p>
</footer>'''
SCRIPT = '''
<div class="updbar" id="updBar" hidden role="status"><span>A new version is ready.</span><button type="button" id="updGo">Reload</button></div>
<script>
// Offline support, the install button and the update notice (only when served from a web address, not opened as a file)
(function () {
  if (!('serviceWorker' in navigator) || location.protocol === 'file:') return;
  var btn = document.getElementById('btnInstall'), deferred = null;
  window.addEventListener('beforeinstallprompt', function (e) { e.preventDefault(); deferred = e; btn.hidden = false; });
  btn.addEventListener('click', function () { if (!deferred) return; btn.hidden = true; deferred.prompt(); deferred = null; });
  window.addEventListener('appinstalled', function () { btn.hidden = true; });
  var hadController = !!navigator.serviceWorker.controller;
  navigator.serviceWorker.register('sw.js').then(function (reg) {
    setInterval(function () { reg.update().catch(function () {}); }, 3600 * 1000);
  }).catch(function (e) { console.warn('Offline support is unavailable here:', e); });
  navigator.serviceWorker.addEventListener('controllerchange', function () { if (hadController) document.getElementById('updBar').hidden = false; });
  document.getElementById('updGo').addEventListener('click', function () {
    var w = window.__wt, rec = w && w.G && w.G.rec;
    if (rec && !confirm('A recording is running. Reload now? It can resume from its last checkpoint.')) return;
    location.reload();
  });
})();
</script>
'''
assert '<button id="btnImg">Save image</button>' in body and '</footer>' in body
body = body.replace('<button id="btnImg">Save image</button>', '<button id="btnImg">Save image</button>' + INSTALL_BTN, 1)
body = body.replace('</footer>', FOOT, 1)
html = TOP + head + STYLE + '</style>\n</head>\n<body>\n' + body + SCRIPT + '</body>\n</html>\n'
open(os.path.join(OUT, 'index.html'), 'w', encoding='utf-8').write(html)

# ---- manifest ----
manifest = {
    'name': 'Aerofoil Wind Tunnel', 'short_name': 'Wind Tunnel',
    'description': 'Two-dimensional wind tunnel: lattice-Boltzmann CFD of aerofoils, with probes, recordings and image and video export.',
    'id': './', 'start_url': './', 'scope': './', 'display': 'standalone', 'orientation': 'any',
    'background_color': '#0c1015', 'theme_color': '#0f1418', 'categories': ['education', 'science', 'utilities'],
    'icons': [{'src': 'icons/icon-192.png', 'sizes': '192x192', 'type': 'image/png', 'purpose': 'any'},
              {'src': 'icons/icon-512.png', 'sizes': '512x512', 'type': 'image/png', 'purpose': 'any'},
              {'src': 'icons/maskable-192.png', 'sizes': '192x192', 'type': 'image/png', 'purpose': 'maskable'},
              {'src': 'icons/maskable-512.png', 'sizes': '512x512', 'type': 'image/png', 'purpose': 'maskable'}],
}
mtext = json.dumps(manifest, indent=2, ensure_ascii=False) + '\n'
open(os.path.join(OUT, 'manifest.webmanifest'), 'w', encoding='utf-8').write(mtext)

# ---- offline script, versioned by what it caches ----
h = hashlib.sha256(html.encode() + mtext.encode())
for f in ICONS: h.update(open(os.path.join(OUT, f), 'rb').read())
version = h.hexdigest()[:12]
sw = '''// Offline cache for the Aerofoil Wind Tunnel. Generated by src/build_webapp.py; version changes with the content.
const VERSION = '%s';
const SHELL = ['./', 'index.html', 'manifest.webmanifest', %s];
const CACHE = 'wt-shell-' + VERSION, FONTS = 'wt-fonts';

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});
self.addEventListener('activate', (e) => {
  e.waitUntil(caches.keys()
    .then((keys) => Promise.all(keys.filter((k) => k.startsWith('wt-shell-') && k !== CACHE).map((k) => caches.delete(k))))
    .then(() => self.clients.claim()));
});
self.addEventListener('fetch', (e) => {
  const req = e.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  if (url.origin === self.location.origin) {
    // the app itself: from the cache, so it opens instantly and offline; a new version arrives with a new offline script
    const key = req.mode === 'navigate' ? 'index.html' : req;
    e.respondWith(caches.open(CACHE).then((c) => c.match(key, { ignoreSearch: true })).then((hit) => hit || fetch(req)));
    return;
  }
  if (url.hostname === 'fonts.googleapis.com' || url.hostname === 'fonts.gstatic.com') {
    // web fonts: kept once seen, refreshed in the background
    e.respondWith(caches.open(FONTS).then(async (c) => {
      const hit = await c.match(req);
      const net = fetch(req).then((res) => { if (res.ok || res.type === 'opaque') c.put(req, res.clone()); return res; }).catch(() => hit);
      return hit || net;
    }));
  }
});
''' % (version, ', '.join("'%s'" % f for f in ICONS))
open(os.path.join(OUT, 'sw.js'), 'w', encoding='utf-8').write(sw)
print('built web app', version, '-', len(html) // 1024, 'KB page')
