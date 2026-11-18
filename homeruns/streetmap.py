"""A flat, north-up street map with street names and the heat map between the roads and the labels, so
names stay readable on top of the routes.

The map is OpenFreeMap's vector tiles of OpenStreetMap data (no API key), drawn by MapLibre GL in a
headless Chrome or Chromium. The runs are drawn into an image here and served to the browser from
127.0.0.1 with the page; only map tiles, fonts and the MapLibre library come from the internet.
"""
import http.server
import json
import math
import os
import shutil
import subprocess
import tempfile
import threading

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from . import overlay, tracks

STYLES = ("liberty", "bright", "positron")
MAPLIBRE = "https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js"
DPR = 2                 # device pixels per CSS pixel: crisp labels in a big image
NAMES_ZOOM = 14         # the vector tiles carry the names of small streets from this zoom
MAX_PX = 14000          # longest side we'll grow to for NAMES_ZOOM (WebGL allows 16384)
LINE_W = 6              # route width in device px
INK = [(0.0, (242, 147, 95)), (0.12, (226, 72, 43)), (0.35, (184, 19, 47)), (0.7, (114, 9, 43)), (1.0, (60, 4, 24))]
CREDIT = "\u00a9 OpenFreeMap \u00a9 OpenMapTiles \u00a9 OpenStreetMap contributors"
CHROMES = ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
           "/Applications/Chromium.app/Contents/MacOS/Chromium",
           "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
           "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
           "google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "microsoft-edge",
           "C:/Program Files/Google/Chrome/Application/chrome.exe",
           "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"]


def find_chrome(path=None):
    for cand in ([path] if path else CHROMES):
        found = cand if os.path.exists(cand) else shutil.which(cand)
        if found:
            return found
    raise SystemExit("--street-map needs Chrome or Chromium; install one or pass --chrome /path/to/chrome.")


def merc(lat, lon):
    """Normalised Web Mercator, 0..1 with y pointing down."""
    lat = np.asarray(lat, float)
    return (np.asarray(lon, float) + 180) / 360, (1 - np.arcsinh(np.tan(np.radians(lat))) / math.pi) / 2


def frame(runs, long_px=8000):
    """Map extent (all runs plus a margin) and the zoom at which it is `long_px` device pixels on its long
    side, grown up to MAX_PX if that is what it takes to reach NAMES_ZOOM."""
    allp = np.concatenate([r.loc for r in runs])
    S, N, W, E = allp[:, 0].min(), allp[:, 0].max(), allp[:, 1].min(), allp[:, 1].max()
    ky, kx = tracks.metres_per_degree((S + N) / 2)
    pad = max(0.06 * max((N - S) * ky, (E - W) * kx), 300)
    S, N, W, E = S - pad / ky, N + pad / ky, W - pad / kx, E + pad / kx
    x0, y0 = (float(v) for v in merc(N, W))
    x1, y1 = (float(v) for v in merc(S, E))
    span = max(x1 - x0, y1 - y0) * 512
    zoom = math.log2(long_px / DPR / span)
    if zoom < NAMES_ZOOM:
        zoom = min(NAMES_ZOOM, math.log2(MAX_PX / DPR / span))
    # round up: rounding down could land just below a whole zoom and load the coarser tiles
    css_w, css_h = math.ceil((x1 - x0) * 512 * 2 ** zoom), math.ceil((y1 - y0) * 512 * 2 ** zoom)
    zoom = math.log2(css_w / ((x1 - x0) * 512))      # exact for the rounded width
    cy = (y0 + y1) / 2
    return dict(S=S, N=N, W=W, E=E, x0=x0, y0=y0, zoom=zoom, css_w=css_w, css_h=css_h,
                center=[(x0 + x1) / 2 * 360 - 180, math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * cy))))])


def heat_image(runs, fr):
    """RGBA image of the runs, pixel-aligned with the map canvas: more runs on a street, darker ink."""
    w, h = fr["css_w"] * DPR, fr["css_h"] * DPR
    scale = 512 * 2 ** fr["zoom"] * DPR
    acc = np.zeros((h, w), np.uint16)
    for r in runs:
        mx, my = merc(r.loc[:, 0], r.loc[:, 1])
        px, py = (mx - fr["x0"]) * scale, (my - fr["y0"]) * scale
        bx0, by0 = max(int(px.min()) - LINE_W, 0), max(int(py.min()) - LINE_W, 0)
        bx1, by1 = min(int(px.max()) + LINE_W + 1, w), min(int(py.max()) + LINE_W + 1, h)
        m = Image.new("L", (bx1 - bx0, by1 - by0), 0)
        ImageDraw.Draw(m).line(list(zip(px - bx0, py - by0)), fill=1, width=LINE_W, joint="curve")
        acc[by0:by1, bx0:bx1] += np.asarray(m, np.uint16)
    top = max(float(np.percentile(acc[acc > 0], 99.5)) if acc.any() else 2.0, 2.0)
    t = np.clip(np.log1p(acc) / math.log1p(top), 0, 1)
    pp = np.array([p for p, _ in INK]); cc = np.array([c for _, c in INK], float)
    rgb = np.stack([np.interp(t, pp, cc[:, i]) for i in range(3)], -1)
    alpha = np.where(acc > 0, 0.6 + 0.4 * t, 0.0)
    im = Image.fromarray(np.dstack([rgb, alpha * 255]).astype(np.uint8), "RGBA")
    return im.filter(ImageFilter.GaussianBlur(0.7))   # soften the pixel steps of the lines


def page_html(fr, style):
    cfg = dict(style=f"https://tiles.openfreemap.org/styles/{style}", center=fr["center"], zoom=fr["zoom"],
               dpr=DPR, corners=[[fr["W"], fr["N"]], [fr["E"], fr["N"]], [fr["E"], fr["S"]], [fr["W"], fr["S"]]])
    return f"""<!doctype html><html><head><meta charset="utf-8">
<script src="{MAPLIBRE}"></script>
<style>body{{margin:0}} #map{{width:{fr["css_w"]}px;height:{fr["css_h"]}px}}</style></head>
<body><div id="map"></div><script>
const cfg = {json.dumps(cfg)};
const say = m => fetch('/log', {{method: 'POST', body: String(m)}});
window.onerror = m => say('page error: ' + m);
const map = new maplibregl.Map({{container: 'map', style: cfg.style, center: cfg.center, zoom: cfg.zoom,
  pixelRatio: cfg.dpr, maxCanvasSize: [16384, 16384], preserveDrawingBuffer: true, interactive: false,
  fadeDuration: 0, attributionControl: false}});
map.on('error', e => say('map error: ' + (e.error && e.error.message)));
map.on('load', () => {{
  const layers = map.getStyle().layers;
  map.addSource('heat', {{type: 'image', url: 'heat.png', coordinates: cfg.corners}});
  map.addLayer({{id: 'heat', type: 'raster', source: 'heat',
    paint: {{'raster-resampling': 'linear', 'raster-fade-duration': 0}}}}, layers.find(l => l.type === 'symbol').id);
  // names of small streets normally appear from zoom 15; show them at this zoom too
  for (const l of layers)
    if (l.type === 'symbol' && l['source-layer'] === 'transportation_name')
      map.setLayerZoomRange(l.id, Math.min(l.minzoom ?? 0, 12), l.maxzoom ?? 24);
  map.once('idle', () => setTimeout(() => map.getCanvas().toBlob(
    b => fetch('/save', {{method: 'POST', body: b}})), 1000));
}});
</script></body></html>"""


def render(runs, out, style="liberty", chrome=None, long_px=8000, log=print):
    chrome = chrome or find_chrome()
    fr = frame(runs, long_px)
    log(f"street map: {fr['css_w'] * DPR}x{fr['css_h'] * DPR} px at zoom {fr['zoom']:.1f} "
        f"(OpenFreeMap {style}, drawn in headless Chrome)")
    with tempfile.TemporaryDirectory(prefix="homeruns_map_", ignore_cleanup_errors=True) as page:
        heat_image(runs, fr).save(os.path.join(page, "heat.png"))
        open(os.path.join(page, "index.html"), "w").write(page_html(fr, style))
        done = threading.Event()

        class Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *a, **k):
                super().__init__(*a, directory=page, **k)

            def log_message(self, *a):
                pass

            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                if self.path == "/save":
                    open(os.path.join(page, "map.png"), "wb").write(body)
                    done.set()
                else:
                    log(f"street map: {body.decode(errors='replace')}")
                self.send_response(204)
                self.end_headers()

        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        proc = subprocess.Popen([chrome, "--headless=new", "--no-first-run", "--no-default-browser-check",
                                 f"--user-data-dir={os.path.join(page, 'profile')}", "--window-size=1600,1200",
                                 f"http://127.0.0.1:{srv.server_address[1]}/"],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            ok = done.wait(300)
        finally:
            proc.terminate()
            proc.wait()
            srv.shutdown()
        if not ok:
            raise SystemExit("street map: timed out waiting for the browser (no internet, or the tiles didn't load)")
        img = Image.open(os.path.join(page, "map.png")).convert("RGB")
    d = ImageDraw.Draw(img)
    f = overlay.load_font("text", 13 * DPR)
    tw = d.textlength(CREDIT, font=f)
    d.rectangle((img.width - tw - 16 * DPR, img.height - 22 * DPR, img.width, img.height), fill=(255, 255, 255))
    d.text((img.width - 8 * DPR, img.height - 6 * DPR), CREDIT, font=f, fill=(80, 80, 80), anchor="rs")
    img.save(out)
    log(f"wrote {out}")
