# -*- coding: utf-8 -*-
"""
tests/test_official_geometry.py — починка геометрии официальных кейсов
SU2 (official_cases/surface.py, downloader.py) и санитизация официального
config.cfg перед одиночным запуском.

Без Qt и сети: только чистые функции.

Запуск:  python tests/test_official_geometry.py
"""
import os
import sys
import tempfile

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


import official_cases as oc

# --- контур из сегментов -> петли -------------------------------------------
print("== chain_loops ==")
# квадрат: 4 сегмента, замыкание в петлю
sq = [((0, 0), (1, 0)), ((1, 0), (1, 1)), ((1, 1), (0, 1)), ((0, 1), (0, 0))]
loops = oc.chain_loops(sq)
check("квадрат: одна петля", len(loops) == 1, str(loops))
check("квадрат: 4 вершины", loops and len(loops[0]) == 4,
      str(loops and len(loops[0])))

# профиль из двух дуг (верх/низ), сегменты перемешаны
n = 20
segs = []
for i in range(n):
    x1, x2 = i / n, (i + 1) / n
    segs.append(((x1, 0.1 * (1 - (2 * x1 - 1) ** 2)),
                 (x2, 0.1 * (1 - (2 * x2 - 1) ** 2))))
    segs.append(((x1, -0.05 * (1 - (2 * x1 - 1) ** 2)),
                 (x2, -0.05 * (1 - (2 * x2 - 1) ** 2))))
segs.append(((0.0, 0.0), (0.0, 0.0)))     # вырожденный сегмент в НК
segs.append(((1.0, 0.0), (1.0, 0.0)))     # вырожденный сегмент в ЗК
loops = oc.chain_loops(segs)
check("профиль: одна петля", len(loops) == 1, str(len(loops)))
check("профиль: вершин >= 2n", loops and len(loops[0]) >= 2 * n - 2,
      str(loops and len(loops[0])))

# --- вытяжка профиля в тело --------------------------------------------------
print("== extrude_loop_to_solid ==")
verts, faces = oc.extrude_loop_to_solid(loops[0], span=2.0)
check("тело построено", len(verts) > 0 and len(faces) > 0)
check("тело замкнуто", oc.is_manifold_closed(faces),
      "рёбер с кратностью != 2 есть")
xs = [v[0] for v in verts]
ys = [v[1] for v in verts]
check("размах = span", abs((max(ys) - min(ys)) - 2.0) < 1e-9)
chord = max(xs) - min(xs)
check("хорда сохранилась", abs(chord - 1.0) < 1e-9)
# знаковый объём: наружная ориентация => объём > 0
import numpy as np
P = np.asarray(verts)
F = np.asarray(faces)
vol6 = sum(float(np.dot(P[a], np.cross(P[b], P[c]))) for a, b, c in F)
check("ориентация наружу (объём > 0)", vol6 > 0, str(vol6))

# --- зеркало для полу-модели --------------------------------------------------
print("== mirror_close_solid ==")
# полу-бокс: открытая оболочка с краем в плоскости y=0
pts = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
       (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
box_faces = [(0, 2, 1), (0, 3, 2),            # низ
             (4, 5, 6), (4, 6, 7),            # верх
             (1, 2, 6), (1, 6, 5),            # x=1
             (0, 4, 7), (0, 7, 3),            # x=0
             (2, 3, 7), (2, 7, 6)]            # y=1 — единственная открытая грань y=0
res = oc.mirror_close_solid(pts, box_faces)
check("полу-бокс: отзеркалился", res is not None)
if res:
    mp, mf = res
    check("полу-бокс: замкнут", oc.is_manifold_closed(mf))
    Pm = np.asarray(mp)
    Fm = np.asarray(mf)
    vol6 = sum(float(np.dot(Pm[a], np.cross(Pm[b], Pm[c])))
               for a, b, c in Fm)
    check("полу-бокс: объём > 0", vol6 > 0, str(vol6))
    check("полу-бокс: размах удвоился",
          abs((Pm[:, 1].max() - Pm[:, 1].min()) - 2.0) < 1e-9)

# --- 2D-сетка SU2 -> тело (сквозной путь fix_body_surface) -------------------
print("== fix_body_surface: 2D профиль ==")
# маленький ромбовидный профиль из линий, NDIME=2
prof = [(0.0, 0.0), (0.5, 0.08), (1.0, 0.0), (0.5, -0.02)]
lines = []
for k in range(len(prof)):
    a, b = prof[k], prof[(k + 1) % len(prof)]
    lines.append("%g %g" % a)
    lines.append("%g %g" % b)
su2_text = []
su2_text.append("NDIME= 2")
su2_text.append("NPOIN= %d" % len(prof))
for (x, y) in prof:
    su2_text.append("%g %g" % (x, y))
su2_text.append("NMARK= 2")
su2_text.append("MARKER_TAG= airfoil")
# сегменты: пары (i j)
order = [(0, 1), (1, 2), (2, 3), (3, 0)]
su2_text.append("MARKER_ELEMS= %d" % len(order))
for (i, j) in order:
    su2_text.append("3 %d %d" % (i, j))
su2_text.append("MARKER_TAG= farfield")
su2_text.append("MARKER_ELEMS= 2")
su2_text.append("3 0 1")
su2_text.append("3 2 3")
with tempfile.TemporaryDirectory() as td:
    m = os.path.join(td, "mesh.su2")
    with open(m, "w", encoding="ascii") as f:
        f.write("\n".join(su2_text) + "\n")
    fixed = oc.fix_body_surface(m, markers=["airfoil"])
    check("2D профиль починен", fixed is not None)
    if fixed:
        check("2D: тело замкнуто", oc.is_manifold_closed(fixed["triangles"]))
        ys2 = [p[1] for p in fixed["points"]]
        check("2D: размах = 2 хорды",
              abs((max(ys2) - min(ys2)) - 2.0) < 1e-9)
        note = fixed.get("note", "")
        check("2D: в примечании сказано про конечное крыло",
              "конечное крыло" in note, note)

# --- санитизация официального config.cfg -------------------------------------
print("== sanitize_config_for_run ==")
from official_cases.downloader import sanitize_config_for_run
with tempfile.TemporaryDirectory() as td:
    cfg = ("SOLVER= RANS\nRESTART_SOL= YES\nSOLUTION_FILENAME= "
           "solution_flow_sa\nCONV_FILENAME= history\n")
    out, notes = sanitize_config_for_run(cfg, td)
    check("RESTART_SOL -> NO", "RESTART_SOL= NO" in out, out)
    check("правка объяснена", any("RESTART_SOL" in x for x in notes), str(notes))
    check("HISTORY_OUTPUT добавлен", "HISTORY_OUTPUT= ( ITER, RMS_RES, FORCES )" in out)
    cfg2 = ("SOLVER= RANS\nRESTART_SOL= NO\n"
            "HISTORY_OUTPUT= ( ITER, RMS_RES )\n")
    out2, notes2 = sanitize_config_for_run(cfg2, td)
    check("RESTART_SOL= NO не тронут", "RESTART_SOL= NO" in out2)
    check("FORCES дописан в существующий HISTORY_OUTPUT",
          "FORCES" in out2 and "ITER" in out2, out2)
    check("нет двойного добавления", out2.count("FORCES") == 1, out2)
    cfg3 = "SOLVER= RANS\nRESTART_SOL= NO\n"
    out3, _n3 = sanitize_config_for_run(cfg3, td)
    check("без HISTORY_OUTPUT строка добавлена",
          out3.count("HISTORY_OUTPUT") == 1)

print()
if FAIL:
    print("ПРОВАЛЕНО: %d" % len(FAIL))
    for name in FAIL:
        print(" -", name)
    sys.exit(1)
print("ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ")
