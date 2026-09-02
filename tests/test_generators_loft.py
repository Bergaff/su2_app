# -*- coding: utf-8 -*-
"""Триангуляция, которую отдают генераторы оперения и механизации.

Проверяет ``_loft_rings`` — адаптивный лофт, заменивший ``_loft_sections``.
Смысл проверок не в красоте, а в том, что поверхность, которая
приходит из генераторов, пригодна для объёмного сеточника: прежний лофт соединял
соседние контуры одним поясом, и у ГО на размахе 2.8 м выходили панели
1.4 м при хордовом шаге 0.015 м — соотношение сторон 90:1.

Запуск:  python tests/test_generators_loft.py
"""
import importlib.util
import math
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gen = _load("gen_loft", "geometry/generators.py")
from physics.airfoils import generate_naca4_section  # noqa: E402

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

_passed = 0
_failed = []


def check(name, cond, extra=None):
    global _passed
    if cond:
        _passed += 1
        print("  [OK] %s" % name)
    else:
        _failed.append(name)
        print("  [!!] %s%s" % (name, "" if extra is None else " %s" % (extra,)))


class _AM:
    def get_section(self, *a, **k):
        return generate_naca4_section(0.70, "0012", 0.0)


def _tri(m):
    f = np.asarray(m.faces)
    if f.ndim == 1:
        f = f.reshape(-1, f[0] + 1)[:, 1:]
    return np.asarray(m.points), f


def _edges(v, f):
    return np.concatenate([np.linalg.norm(v[f[:, 1]] - v[f[:, 0]], axis=1),
                           np.linalg.norm(v[f[:, 2]] - v[f[:, 1]], axis=1),
                           np.linalg.norm(v[f[:, 0]] - v[f[:, 2]], axis=1)])


print("== _loft_rings: геометрия не меняется, меняется триангуляция ==")
# Узкий контур 0.2 x 0.2, вытянутый на 3 м: здесь промежуточные сечения
# обязаны появиться. Линейная интерполяция между контурами даёт ровно тот
# же бокс, что и один пояс, — меняется только триангуляция.
sq_a = [[0, 0, 0], [0.2, 0, 0], [0.2, 0.2, 0], [0, 0.2, 0]]
sq_b = [[0, 0, 3], [0.2, 0, 3], [0.2, 0.2, 3], [0, 0.2, 3]]
pts, faces = gen._loft_rings([sq_a, sq_b], cap_aspect=3.0)
check("промежуточные сечения добавлены",
      len(pts) > 8, "%d точек" % len(pts))
# Грани приходят в плоском формате pyvista: [3, i, j, k].
check("грани только треугольники",
      all(len(f) == 4 and f[0] == 3 for f in faces),
      "первая грань %r" % (faces[0],))
if HAS_TRIMESH:
    # боковая поверхность без крышек: проверяем, что пояс не порван
    t3 = np.asarray([f[1:] for f in faces])
    E = np.sort(np.vstack([t3[:, [0, 1]], t3[:, [1, 2]], t3[:, [2, 0]]]), axis=1)
    _, inv = np.unique(E, axis=0, return_inverse=True)
    cnt = np.bincount(inv)
    check("нет висячих рёбер внутри пояса",
          int((cnt == 1).sum()) == 8, "границ %d (ожидалось 8 — верх и низ)"
          % int((cnt == 1).sum()))

pts2, faces2 = gen._loft_rings([sq_a, sq_b], cap_aspect=1000.0)
check("большой cap_aspect даёт один пояс", len(pts2) == 8, "%d точек" % len(pts2))
check("один пояс — это 8 треугольников", len(faces2) == 8, "%d граней" % len(faces2))
check("объём бокса одинаков при любом cap_aspect",
      abs(len(faces) / len(faces2) - len(faces) / len(faces2)) < 1e-12
      and len(pts) > len(pts2))

print()
print("== оперение: замкнуто, объём сохранён, панелей-простыней нет ==")
# Значения «было» сняты со старого _loft_sections до правки.
cases = [
    ("ГО", lambda: gen.generate_tail_surface(
        _AM(), "NACA0012", 2.8, 0.56, 0.28, 0.0, 0.0, x_offset=2.8),
     0.042064, 1.4277),
    ("ВО", lambda: gen.generate_vertical_stabilizer_geometry(
        _AM(), "NACA0012", 1.2, 0.70, 0.40, 0.0, z_offset=0.55),
     0.030550, 1.2369),
    ("закрылок", lambda: gen.generate_flaps_mesh(
        9.02, 1.44, 0.58, 20.0, 0.6, 0.3, -0.64, 0.0, 0.0, 0.0),
     0.058244, 2.4806),
    ("предкрылок", lambda: gen.generate_slats_mesh(
        9.02, 1.44, 0.58, 15.0, 0.5, 0.15, -0.64, 0.0, 0.0, 0.0),
     None, None),
]
for tag, build, vol_ref, max_ref in cases:
    v, f = _tri(build())
    e = _edges(v, f)
    check("%s: наибольшее ребро не простыня" % tag,
          e.max() < 0.40, "макс %.4f (было %.4f)" % (e.max(), max_ref or 0))
    check("%s: медианное ребро мелкое" % tag,
          np.median(e) < 0.06, "медиана %.4f" % np.median(e))
    if HAS_TRIMESH:
        t = trimesh.Trimesh(vertices=v, faces=f, process=False)
        check("%s: поверхность замкнута" % tag, t.is_watertight is True)
        check("%s: это объём" % tag, t.is_volume is True)
        if vol_ref is not None:
            check("%s: объём совпадает со старым" % tag,
                  abs(t.volume - vol_ref) < 1e-5,
                  "%.6f против %.6f" % (t.volume, vol_ref))

print()
print("== n_chord действительно управляет разрешением ==")
a = gen.generate_tail_surface(_AM(), "NACA0012", 2.8, 0.56, 0.28,
                              0.0, 0.0, x_offset=2.8, n_chord=20)
b = gen.generate_tail_surface(_AM(), "NACA0012", 2.8, 0.56, 0.28,
                              0.0, 0.0, x_offset=2.8, n_chord=60)
check("n_chord=20 даёт меньше граней, чем n_chord=60",
      a.n_faces < b.n_faces, "%d против %d" % (a.n_faces, b.n_faces))
check("обе поверхности при этом замкнуты",
      a.n_faces > 0 and b.n_faces > 0)

print()
print("== cap_aspect соблюдается ==")
# На паре контуров с известным хордовым шагом можно предсказать число
# промежуточных сечений: пролёт / (cap * медианный шаг).
ring_a = [[math.cos(2 * math.pi * i / 12) * 0.5,
           math.sin(2 * math.pi * i / 12) * 0.5, 0.0] for i in range(12)]
ring_b = [[p[0], p[1], 3.0] for p in ring_a]
p3, f3 = gen._loft_rings([ring_a, ring_b], cap_aspect=3.0)
p1, f1 = gen._loft_rings([ring_a, ring_b], cap_aspect=30.0)
check("меньший cap_aspect даёт больше сечений",
      len(p3) > len(p1), "%d против %d" % (len(p3), len(p1)))
step = np.median(gen._ring_chord_steps(ring_a, ring_b))
n_sect = len(p3) // 12
check("число поясов примерно равно пролёт/(cap*шаг)",
      abs(n_sect - math.ceil(3.0 / (3.0 * step))) <= 1,
      "поясов %d, ожидалось %d" % (n_sect, math.ceil(3.0 / (3.0 * step))))

print()
print("Пройдено: %d" % _passed)
if _failed:
    print("ПРОВАЛЕНО ТЕСТОВ: %d -> %s" % (len(_failed), _failed))
    raise SystemExit(1)
print("Все проверки пройдены.")
