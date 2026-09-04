# -*- coding: utf-8 -*-
"""Объединение пересекающихся тел (geometry/solid_union.py).

Модули грузятся по пути файла: geometry/__init__.py тянет PyQt5 через
stl_healer, а эти модули должны работать и в тестах, и в фоновом
процессе генерации.

Запуск:  QT_QPA_PLATFORM=offscreen python tests/test_solid_union.py
"""
import os
import sys
import importlib.util

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_ROOT, relpath))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


su = _load("solid_union_standalone", os.path.join("geometry", "solid_union.py"))
gen = _load("generators_standalone", os.path.join("geometry", "generators.py"))

import pyvista as pv  # noqa: E402

FAIL = []
N = [0]


def check(name, cond, extra=""):
    N[0] += 1
    if cond:
        print("  [OK]   %s" % name)
    else:
        print("  [FAIL] %s %s" % (name, extra))
        FAIL.append(name)


def volume(mesh):
    p, f = su.to_triangles(mesh)
    import trimesh
    return float(trimesh.Trimesh(vertices=p, faces=f, process=True).volume)


def watertight(mesh):
    p, f = su.to_triangles(mesh)
    import trimesh
    return bool(trimesh.Trimesh(vertices=p, faces=f,
                                process=True).is_watertight)


def box(cx, cy, cz, sx, sy, sz):
    return pv.Box(bounds=(cx - sx / 2, cx + sx / 2,
                          cy - sy / 2, cy + sy / 2,
                          cz - sz / 2, cz + sz / 2)).triangulate()


print("Пересекающиеся тела")
a = box(0, 0, 0, 2, 2, 2)          # V = 8
b = box(1, 0, 0, 2, 2, 2)          # V = 8, пересечение 1x2x2 = 4
logs = []
u = su.union_to_polydata([a, b], log=logs.append)
check("объединение построено", u is not None)
if u is not None:
    v = volume(u)
    check("объём объединения = 8+8-4 = 12 (%.4f)" % v, abs(v - 12.0) < 1e-6)
    check("объединение замкнуто", watertight(u))
    st = su.union_stats([a, b], log=lambda *_: None)
    check("union_stats: объём пересечения = 4 (%.4f)" % st["overlap"],
          abs(st["overlap"] - 4.0) < 1e-6)
    check("union_stats: тел на входе 2", st["n_bodies"] == 2)
    check("union_stats: граней до %d, после %d" % (st["facets_in"],
                                                   st["facets_out"]),
          st["facets_in"] == 24 and st["facets_out"] > 0)
    check("union_stats: отдаёт саму сетку", st["mesh"] is not None)

print("Несвязные тела не обрезаются")
c1 = box(0, 0, 0, 1, 1, 1)
c2 = box(5, 0, 0, 1, 1, 1)
st2 = su.union_stats([c1, c2], log=lambda *_: None)
check("объём равен сумме, пересечений нет (%.4f)" % st2["volume_out"],
      abs(st2["volume_out"] - 2.0) < 1e-6 and abs(st2["overlap"]) < 1e-9)

print("Тело целиком внутри другого")
outer = box(0, 0, 0, 4, 4, 4)      # V = 64
inner = box(0, 0, 0, 1, 1, 1)      # V = 1
st3 = su.union_stats([outer, inner], log=lambda *_: None)
check("внутреннее тело вырезано полностью, объём = 64 (%.4f)"
      % st3["volume_out"], abs(st3["volume_out"] - 64.0) < 1e-6)
check("вырезанный объём = 1 (%.4f)" % st3["overlap"],
      abs(st3["overlap"] - 1.0) < 1e-6)

print("Открытая оболочка объединению не поддаётся")
open_shell = pv.PolyData(np.array([[0, 0, 0.], [1, 0, 0.], [0, 1, 0.]]),
                         np.array([3, 0, 1, 2]))
logs2 = []
res = su.union_meshes([open_shell, box(0, 0, 0, 2, 2, 2)], log=logs2.append)
check("незамкнутый вход даёт None", res[0] is None)
check("в лог ушло понятное сообщение",
      any("не удалось" in m or "не замкнута" in m for m in logs2), logs2)

print("Пустой и одиночный вход")
check("пустой список даёт None", su.union_meshes([], log=lambda *_: None)[0]
      is None)
one = su.union_to_polydata([box(0, 0, 0, 1, 1, 1)], log=lambda *_: None)
check("одно тело возвращается как есть (%.4f)" % volume(one),
      one is not None and abs(volume(one) - 1.0) < 1e-6)

print("Регрессия на реальных телах приложения")
from physics.airfoils import generate_naca4_section  # noqa: E402


class AM:
    def get_section(self, *a, **k):
        return generate_naca4_section(0.70, "0012", 0.0)


real = {
    "крыло": gen.generate_wing_mesh(9.02, 1.44, 0.72, 0.0, 0.0, "2412",
                                    -0.64, 0.0, 0.0)[0],
    "ГО": gen.generate_tail_surface(AM(), "NACA0012", 2.8, 0.56, 0.28,
                                    0.0, 0.0, x_offset=2.8),
    "элеватор": gen.generate_tail_surface(AM(), "NACA0012", 2.0, 0.40, 0.20,
                                          0.0, 0.0, x_offset=3.5),
}
vs = gen.generate_vertical_stabilizer_geometry(AM(), "NACA0012", 1.2, 0.70,
                                               0.40, 0.0, z_offset=0.55)
vp = np.asarray(vs.points).copy()
vp[:, 0] += 2.9
real["ВО"] = pv.PolyData(vp, np.asarray(vs.faces))

# Без фюзеляжа эти четыре тела друг с другом не пересекаются: крыло стоит
# на x=-0.64, ГО на 2.8, элеватор на 3.5, киль над ними по Z. Обрезать
# здесь нечего, поэтому фюзеляж обязателен — именно в него входят все
# несущие поверхности.


def fuselage(L_=8.0, D=1.2, nose_ratio=0.35, tail_ratio=0.25):
    """Фюзеляж приложения (ui/main_window.py, generate_fuselage)."""
    import math
    R = D / 2.0
    nose = L_ * nose_ratio
    tail = L_ * tail_ratio
    pts = []
    fcs = []
    rings = []
    for i in range(1, 28):
        xl = -L_ / 2.0 + L_ * (i / 28)
        xfn = xl + L_ / 2.0
        xtt = L_ / 2.0 - xl
        if xfn < nose:
            r = R * math.sin(math.pi * (xfn / nose) / 2.0)
        elif xtt < tail:
            r = R * math.sin(math.pi * (xtt / tail) / 2.0)
        else:
            r = R
        if r < 1e-6:
            continue
        ring = []
        for j in range(48):
            th = 2.0 * math.pi * j / 48
            ring.append(len(pts))
            pts.append([xl, r * math.cos(th), r * math.sin(th)])
        rings.append(ring)
    nt = len(pts)
    pts.append([-L_ / 2.0, 0.0, 0.0])
    tt = len(pts)
    pts.append([L_ / 2.0, 0.0, 0.0])
    fr = rings[0]
    for j in range(48):
        fcs.append([3, nt, fr[(j + 1) % 48], fr[j]])
    for k in range(len(rings) - 1):
        r1, r2 = rings[k], rings[k + 1]
        for j in range(48):
            fcs.append([4, r1[j], r1[(j + 1) % 48], r2[(j + 1) % 48], r2[j]])
    lr = rings[-1]
    for j in range(48):
        fcs.append([3, lr[j], lr[(j + 1) % 48], tt])
    return pv.PolyData(np.array(pts),
                       np.array([v for f in fcs for v in f])
                       ).triangulate().clean(tolerance=1e-6)


real["фюзеляж"] = fuselage()

st4 = su.union_stats(list(real.values()), log=lambda *_: None)
check("пять реальных тел объединяются", st4["mesh"] is not None)
if st4["mesh"] is not None:
    check("объединение реальных тел замкнуто", watertight(st4["mesh"]))
    check("пересечения вырезаны (объём %.4f против суммы %.4f)"
          % (st4["volume_out"], st4["volume_in"]),
          st4["overlap"] > 1e-4)
    check("вырезано именно пересечение с фюзеляжем (%.4f)" % st4["overlap"],
          0.05 < st4["overlap"] < 1.0)

print("Запасной фюзеляж замкнут и объединяется")
# geometry/generators.py:generate_fuselage_mesh помечен как запасной путь.
# Раньше у него не было заглушек на торцах, оболочка получалась открытой,
# и объединение с остальными телами становилось невозможным.
fus = gen.generate_fuselage_mesh({"n1": 8.0, "n2": 0.6, "n3": 0.35})
check("generate_fuselage_mesh замкнут", watertight(fus))
check("объём запасного фюзеляжа разумен (%.4f)" % volume(fus),
      3.0 < volume(fus) < 9.5)
st5 = su.union_stats([fus, real["крыло"]], log=lambda *_: None)
check("запасной фюзеляж объединяется с крылом", st5["mesh"] is not None)
if st5["mesh"] is not None:
    check("объединение с запасным фюзеляжем замкнуто",
          watertight(st5["mesh"]))

print("Кнопка «Объединить пересекающиеся тела»")


class FakeActor:
    def __init__(self):
        self.visible = True

    def SetVisibility(self, v):
        self.visible = bool(v)

    def GetProperty(self):
        return self


class FakePlotter:
    def __init__(self):
        self.added = []
        self.removed = []
        self.renders = 0

    def add_mesh(self, mesh, **kw):
        act = FakeActor()
        self.added.append((mesh, act))
        return act

    def remove_actor(self, act):
        self.removed.append(act)

    def render(self):
        self.renders += 1


from PyQt5.QtWidgets import QApplication, QTextEdit  # noqa: E402
import ui.main_window as mw  # noqa: E402

app = QApplication.instance() or QApplication([])


def make_window(meshes):
    w = mw.MainWindow.__new__(mw.MainWindow)
    w.plotter = FakePlotter()
    w.log_text = QTextEdit()
    w.unified_mesh = None
    w.unified_actor = None
    w.unified_path = None
    w.bodies = []
    for i, m in enumerate(meshes):
        w.bodies.append({"id": i, "name": "b%d" % i, "role": "other",
                         "mesh": m, "actor": FakeActor(), "visible": True,
                         "color": None})
    return w


w = make_window([a, b])
mw.MainWindow.union_bodies(w)
log = w.log_text.toPlainText()
check("после нажатия объединённая модель показана", w.unified_actor is not None)
check("в лог попало число объединённых тел",
      "объединено тел: 2" in log, log[:120])
check("в лог попал вырезанный объём пересечений",
      "вырезано 4.0000 пересечений" in log, log[-200:])
check("отдельные детали скрыты",
      all(not b["actor"].visible for b in w.bodies))
check("unified_model.stl сохранён",
      w.unified_path is not None and os.path.exists(w.unified_path))

mw.MainWindow.union_bodies(w)
check("повторное нажатие возвращает раздельный вид",
      w.unified_actor is None and w.unified_mesh is None)
check("детали снова видимы",
      all(b["actor"].visible for b in w.bodies))

w1 = make_window([a])
w1.log_text.clear()
mw.MainWindow.union_bodies(w1)
check("одиночное тело не объединяется, есть сообщение",
      "хотя бы два" in w1.log_text.toPlainText(), w1.log_text.toPlainText())

w0 = make_window([open_shell, box(0, 0, 0, 2, 2, 2)])
w0.log_text.clear()
mw.MainWindow.union_bodies(w0)
check("незамкнутые тела: объединения нет, есть объяснение",
      w0.unified_actor is None
      and "замкнут" in w0.log_text.toPlainText(),
      w0.log_text.toPlainText()[:120])

if w.unified_path and os.path.exists(w.unified_path):
    os.remove(w.unified_path)

print()
print("Проверок: %d" % N[0])
if FAIL:
    print("ПРОВАЛЕНО: %d -> %s" % (len(FAIL), FAIL))
    sys.exit(1)
print("ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ")
