# -*- coding: utf-8 -*-
"""
tests/test_surface_remesh.py — изотропная ремешка поверхности
(mesh/surface_remesh.py).

Без gmsh проверяется чистая математика: многоугольность рёбер, статистика
вытянутости, выбор целевого шага, знаковый объём, согласование ориентации.
С gmsh — сама ремешка на вырожденно вытянутой пластине: лофт из 3 сечений
(та же беда, что у ГО/руля генератора) должен перестроиться в
равносторонние треугольники с сохранением замкнутости и объёма.

Запуск:  python tests/test_surface_remesh.py
"""
import importlib.util
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

FAIL = []


def check(name, cond, extra=""):
    if cond:
        print("  OK %s" % name)
    else:
        FAIL.append(name)
        print("  FAIL %s %s" % (name, extra))


def _load():
    path = os.path.join(ROOT, "mesh", "surface_remesh.py")
    spec = importlib.util.spec_from_file_location("surface_remesh_standalone",
                                                  path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sr = _load()


def box_pts_faces(x0, y0, z0, x1, y1, z1):
    """Замкнутый бокс из 12 треугольников (наружные нормали)."""
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])
    f = np.array([
        [0, 2, 1], [0, 3, 2],            # низ
        [4, 5, 6], [4, 6, 7],            # верх
        [0, 1, 5], [0, 5, 4],            # y0
        [2, 3, 7], [2, 7, 6],            # y1
        [1, 2, 6], [1, 6, 5],            # x1
        [3, 0, 4], [3, 4, 7],            # x0
    ], dtype=np.int64)
    return v, f


def thin_plate_loft(n_span=2, n_chord=40, chord=0.56, thickness=0.0672,
                    span=2.8):
    """Лофт «3 сечения» как у ГО генератора: сильно вытянутые грани.

    Возвращает (points, faces). Профиль — плоская пластина толщиной
    ``thickness`` (прямоугольник в сечении), станции по размаху через
    равные интервалы. Медианный AR ~ span_панели / chord_шаг.
    """
    n_sec = 2 * n_chord - 2  # верх+низ без повторения ЗК/НК
    xs = np.linspace(0.0, chord, n_chord)
    # контур: верх от ЗК к НК, низ обратно
    loop_x = np.concatenate([xs, xs[::-1][1:-1]])
    loop_z = np.concatenate([np.full(n_chord, thickness / 2.0),
                             np.full(n_chord - 2, -thickness / 2.0)])
    n_st = n_span + 1
    pts = []
    for s in range(n_st):
        y = -span / 2.0 + span * s / n_span
        for k in range(n_sec):
            pts.append([loop_x[k], y, loop_z[k]])
    pts = np.array(pts)
    faces = []
    for s in range(n_st - 1):
        b0, b1 = s * n_sec, (s + 1) * n_sec
        for k in range(n_sec):
            k2 = (k + 1) % n_sec
            faces.append([b0 + k, b0 + k2, b1 + k2])
            faces.append([b0 + k, b1 + k2, b1 + k])
    # торцы (крышки)
    for (s, sgn) in ((0, -1), (n_st - 1, +1)):
        base = s * n_sec
        c = len(pts)
        ctr = pts[base:base + n_sec].mean(axis=0)
        pts = np.vstack([pts, ctr])
        for k in range(n_sec):
            k2 = (k + 1) % n_sec
            if sgn < 0:
                faces.append([base + k, c, base + k2])
            else:
                faces.append([base + k, base + k2, c])
    return pts, np.array(faces, dtype=np.int64)


# --- чистые функции ---------------------------------------------------------
print("== чистые функции без gmsh ==")

_, f_box = box_pts_faces(0, 0, 0, 1, 1, 1)
check("edge_manifold_bad: замкнутый бокс = 0", sr.edge_manifold_bad(f_box) == 0)
f_open = f_box[:-1]
check("edge_manifold_bad: бокс без грани > 0", sr.edge_manifold_bad(f_open) > 0)

v_b, f_b = box_pts_faces(0, 0, 0, 1, 1, 1)
check("surface_volume: единичный бокс = 1",
      abs(sr.surface_volume(v_b, f_b) - 1.0) < 1e-12)
check("surface_volume: зеркальный бокс = -1",
      abs(sr.surface_volume(v_b * np.array([1, 1, -1.0]),
                            f_b) + 1.0) < 1e-12)

med, p95, mx, frac = sr.aspect_stats(v_b, f_b)
check("aspect_stats: бокс из равнобедренных ~1.2",
      med < 1.5 and p95 < 1.5, (med, p95))

check("component_target_edge: крупное тело берёт h_surf",
      abs(sr.component_target_edge(1.2, 0.079) - 0.079) < 1e-12)
check("component_target_edge: тонкая пластина 0.75*t",
      abs(sr.component_target_edge(0.0672, 0.079) - 0.0504) < 1e-12)
check("component_target_edge: без h_surf -> None",
      sr.component_target_edge(1.0, None) is None)

check("surface_needs_remesh: бокс не нуждается",
      not sr.surface_needs_remesh(v_b, f_b))
pts_lp, faces_lp = thin_plate_loft()
med_lp, p95_lp, mx_lp, _ = sr.aspect_stats(pts_lp, faces_lp)
check("surface_needs_remesh: лофт 3 сечений нуждается (AR med %.0f)" % med_lp,
      sr.surface_needs_remesh(pts_lp, faces_lp), (med_lp, p95_lp))

# --- ориентация -------------------------------------------------------------
v_sh = v_b.copy()
f_sh = f_b.copy()
f_sh[3] = f_sh[3][::-1]
f_sh[7] = f_sh[7][::-1]
f2 = f_sh.copy()
sr._orient_consistent(v_sh, f2)
check("orient: перевёрнутые грани согласованы",
      abs(sr.surface_volume(v_sh, f2) - 1.0) < 1e-12,
      sr.surface_volume(v_sh, f2))

# --- ремешка (нужен gmsh) ---------------------------------------------------
print("== ремешка (gmsh %s) ==" % ("доступен" if sr.HAS_GMSH else "НЕТ"))
if sr.HAS_GMSH:
    v0 = sr.surface_volume(pts_lp, faces_lp)
    out = sr.remesh_component(pts_lp, faces_lp, 0.03, log=print)
    check("ремешка лофта выполнена", out is not None)
    if out is not None:
        pts2, fcs2 = out
        check("ремешка сохранила замкнутость",
              sr.edge_manifold_bad(fcs2) == 0)
        med2, p952, mx2, frac2 = sr.aspect_stats(pts2, fcs2)
        check("ремешка убрала вытянутость (AR med %.1f -> %.1f)"
              % (med_lp, med2), med2 < 5.0, (med_lp, med2))
        check("ремешка сохранила объём (Δ %.2f%%)"
              % (100 * abs(sr.surface_volume(pts2, fcs2) - v0) / v0),
              abs(sr.surface_volume(pts2, fcs2) - v0) < 0.05 * abs(v0))
        bbox_ok = np.allclose(pts2.min(axis=0), pts_lp.min(axis=0), atol=0.01)
        check("ремешка сохранила габарит", bbox_ok)
    # незамкнутый вход -> None
    out_bad = sr.remesh_component(pts_lp, faces_lp[:-3], 0.03, log=print)
    check("ремешка дырявой поверхности -> None", out_bad is None)

print()
if FAIL:
    print("ПРОВАЛЕНО: %d" % len(FAIL))
    for name in FAIL:
        print(" -", name)
    sys.exit(1)
print("ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ")
