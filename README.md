# Aerofoil Wind Tunnel

A two-dimensional wind tunnel that runs in the browser: lattice-Boltzmann CFD of aerofoils on the graphics card, with probes, recordings, and image and video export. This folder is an installable web app. Once installed it has its own icon and window, and it works offline.

## Put it online

The app has to be served from a web address before browsers will offer to install it. Opening `index.html` as a file still runs the tunnel, but without install or offline support.

**Netlify Drop (no account needed to try it)**
1. Go to https://app.netlify.com/drop.
2. Drag this whole folder onto the page.
3. Open the link it gives you.

**GitHub Pages**
1. Create a public repository, for example `wind-tunnel`, and upload the contents of this folder to it.
2. In the repository open Settings → Pages. Choose "Deploy from a branch", the `main` branch and the `/ (root)` folder, then Save.
3. After a minute or two the app is at `https://<your-username>.github.io/wind-tunnel/`.

## Install it

- **Chrome or Edge on a computer:** click Install app in the toolbar, or the install icon in the address bar.
- **Android:** open the browser menu and choose Install app or Add to Home screen.
- **iPhone or iPad:** in Safari tap Share, then Add to Home Screen.

Recordings are stored on the device and in the browser that made them. Like any web app, the tunnel pauses while it isn't on screen, so keep it in front during long recordings.

## Updating

The sources are in `src/`. After changing them, rebuild the site files and publish the folder again:

```
python3 src/build_webapp.py      # writes index.html, sw.js and manifest.webmanifest
python3 src/make_icons.py        # only if you change the icon (needs Pillow)
```

Installed copies pick up the new version the next time they are opened online. A "new version is ready" notice offers to reload.

## Files

| File | What it is |
|---|---|
| `index.html` | The whole app in one page |
| `manifest.webmanifest` | App name, colours and icons, used when installing |
| `sw.js` | Offline cache; its version changes whenever the app does |
| `icons/` | App icons for computers, Android and iOS |
| `src/` | Sources and build scripts |
| `.nojekyll` | Tells GitHub Pages to serve the files as they are |
