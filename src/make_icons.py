# App icons: streamlines of potential flow round a cambered Joukowski aerofoil at 8 degrees (Kutta condition),
# drawn in the tunnel's dark test-section colours. Free stream horizontal, aerofoil nose-up.
import cmath, math, os
from PIL import Image, ImageDraw

BG = (12, 16, 21); LINE = (196, 210, 222); ACC = (238, 132, 66)
S = 1536                                  # drawing size before downsampling
ALPHA = math.radians(8)
MU = complex(-0.09, 0.07)                 # circle centre: thickness and camber
R = abs(1 - MU)                           # circle through z = 1, so the trailing edge is sharp
BETA = math.atan2(MU.imag, 1 - MU.real)
GAMMA = 4 * math.pi * R * math.sin(ALPHA + BETA)

def zeta_of(z):  # inverse Joukowski map, root outside the circle
    s = cmath.sqrt(z * z - 4)
    a, b = (z + s) / 2, (z - s) / 2
    return a if abs(a - MU) >= abs(b - MU) else b

def vel(P):  # velocity at P in the drawing frame (free stream along +x)
    z = P * cmath.exp(1j * ALPHA)         # aerofoil frame: chord along x, flow arriving at +alpha
    zt = zeta_of(z)
    if abs(zt - MU) < R * 1.0005: return None
    d = zt - MU
    dW = cmath.exp(-1j * ALPHA) - R * R * cmath.exp(1j * ALPHA) / (d * d) + 1j * GAMMA / (2 * math.pi * d)
    w = dW / (1 - 1 / (zt * zt))          # conjugate velocity u - iv in the aerofoil frame
    u = w.conjugate() * cmath.exp(-1j * ALPHA)  # back to the drawing frame, where the free stream is horizontal
    return u

def streamline(y0, x0=-4.2, x1=4.2, h=0.01):
    P = complex(x0, y0); pts = [P]
    for _ in range(4000):
        v = vel(P)
        if v is None or abs(v) < 1e-6: break
        k1 = v / abs(v); Pm = P + 0.5 * h * k1; v2 = vel(Pm)
        if v2 is None: break
        P = P + h * v2 / abs(v2); pts.append(P)
        if P.real > x1: break
    return pts

def body():
    pts = []
    for i in range(400):
        th = 2 * math.pi * i / 400
        zt = MU + R * cmath.exp(1j * th); z = zt + 1 / zt
        pts.append(z * cmath.exp(-1j * ALPHA))
    return pts

LINES = [streamline(y) for y in (-2.6, -1.95, -1.3, -0.68, 0.62, 1.25, 1.9, 2.55, 3.2)]
BODY = body()

def art(size, extent, rounded, lines=True):
    im = Image.new('RGBA', (size, size), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.rounded_rectangle([0, 0, size - 1, size - 1], radius=int(0.22 * size) if rounded else 0, fill=BG + (255,))
    k = size / extent; cx, cy = size / 2, size / 2 + 0.06 * size
    to = lambda P: (cx + k * (P.real - 0.05), cy - k * (P.imag - 0.3))
    mask = Image.new('L', (size, size), 0); ImageDraw.Draw(mask).rounded_rectangle([0, 0, size - 1, size - 1], radius=int(0.22 * size) if rounded else 0, fill=255)
    if lines:
        layer = Image.new('RGBA', (size, size), (0, 0, 0, 0)); ld = ImageDraw.Draw(layer)
        for j, ln in enumerate(LINES):
            dist = min(abs(j - 3.5), 4) / 4
            ld.line([to(P) for P in ln[::5]] + [to(ln[-1])], fill=LINE + (int(235 - 120 * dist),), width=max(2, int(size * 0.016)), joint='curve')
        im.alpha_composite(Image.composite(layer, Image.new('RGBA', (size, size), (0, 0, 0, 0)), mask))
    d = ImageDraw.Draw(im)
    d.polygon([to(P) for P in BODY], fill=ACC + (255,))
    return im

def save(size, extent, rounded, path, lines=True):
    art(S, extent, rounded, lines).resize((size, size), Image.LANCZOS).save(path)

ICON_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'icons')  # site root/icons
os.makedirs(ICON_DIR, exist_ok=True)
save(512, 6.2, True, ICON_DIR + '/icon-512.png')
save(192, 6.2, True, ICON_DIR + '/icon-192.png')
save(512, 7.9, False, ICON_DIR + '/maskable-512.png')   # artwork kept inside the central safe circle
save(192, 7.9, False, ICON_DIR + '/maskable-192.png')
save(180, 6.6, False, ICON_DIR + '/apple-touch-icon.png')
save(32, 5.2, True, ICON_DIR + '/favicon-32.png', lines=False)
print('icons written')
