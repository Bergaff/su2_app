import math
import os
import numpy as np


def generate_naca4_section(chord: float, code: str = "0012", twist: float = 0.0,
                           n: int = 40):
    """Сечение NACA 4-значного профиля в плоскости XZ.
    chord — хорда сечения (м), twist — локальная крутка (град, вокруг 25% хорды).
    Возвращает списки (x, z) длиной 2n-1: верх от ЗК к ХВ + низ к ЗК."""
    code = "".join(ch for ch in str(code) if ch.isdigit())[:4] or "0012"
    code = code.rjust(4, "0")
    m = int(code[0]) / 100.0
    p = int(code[1]) / 10.0
    t = int(code[2:]) / 100.0

    beta = np.linspace(0.0, math.pi, n)
    x = 0.5 * (1.0 - np.cos(beta))
    yt = 5.0 * t * (0.2969 * np.sqrt(np.clip(x, 1e-12, None))
                    - 0.1260 * x - 0.3516 * x ** 2
                    + 0.2843 * x ** 3 - 0.1015 * x ** 4)
    if p > 0.0:
        yc = np.where(x < p,
                      m / p ** 2 * (2.0 * p * x - x ** 2),
                      m / (1.0 - p) ** 2 * ((1.0 - 2.0 * p) + 2.0 * p * x - x ** 2))
    else:
        yc = np.zeros_like(x)

    xu, yu = x, yc + yt
    xl, yl = x, yc - yt
    # контур: верхняя дуга от ХВ к ЗК, затем нижняя обратно к ХВ
    xs = np.concatenate([xu[::-1], xl[1:]])
    ys = np.concatenate([yu[::-1], yl[1:]])

    tw = math.radians(twist)
    px = (xs - 0.25) * math.cos(tw) + ys * math.sin(tw)
    py = -(xs - 0.25) * math.sin(tw) + ys * math.cos(tw)

    xs_out = ((px + 0.25) * chord).tolist()
    ys_out = (py * chord).tolist()
    return xs_out, ys_out


class AirfoilManager:
    """Менеджер профилей: NACA 4-значные + файлы Selig (.dat)."""

    def __init__(self):
        self.cache = {}
        default_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "airfoils")
        self.airfoil_dir = default_dir if os.path.isdir(default_dir) else "./airfoils"

    def get_airfoil(self, name: str, n: int = 120):
        key = name.strip()
        cache_key = (key.upper(), n)
        if cache_key in self.cache:
            return self.cache[cache_key]

        coords = None
        if key.lower().startswith("naca"):
            x, y = generate_naca4_section(1.0, "".join(ch for ch in key if ch.isdigit()),
                                          0.0, n // 2 + 1)
            coords = np.column_stack([x, y])
        else:
            path = os.path.join(self.airfoil_dir, f"{key}.dat")
            if os.path.exists(path):
                coords = self._load_selig(path, n)

        self.cache[cache_key] = coords
        return coords

    def _load_selig(self, path: str, n: int):
        pts = []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.replace(",", " ").split()
                if len(parts) >= 2:
                    try:
                        pts.append([float(parts[0]), float(parts[1])])
                    except ValueError:
                        continue
        if len(pts) < 10:
            return None
        pts = np.array(pts, dtype=float)
        xmin, xmax = pts[:, 0].min(), pts[:, 0].max()
        if xmax > xmin:
            pts[:, 0] = (pts[:, 0] - xmin) / (xmax - xmin)
            pts[:, 1] /= (xmax - xmin)
        idx = np.linspace(0, len(pts) - 1, n).astype(int)
        return pts[idx]

    def list_available(self):
        out = ["NACA0012", "NACA2412", "NACA4415", "NACA6409"]
        if os.path.isdir(self.airfoil_dir):
            for f in sorted(os.listdir(self.airfoil_dir)):
                if f.lower().endswith(".dat"):
                    out.append(os.path.splitext(f)[0])
        return out
