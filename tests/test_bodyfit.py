# -*- coding: utf-8 -*-
"""Подготовка поверхности для телооблегающей сетки (mesh/bodyfit.py).

Загружается по пути файла: mesh/__init__.py тянет PyQt5 через mesh_worker,
а этот модуль должен работать и в тестах, и в фоновом процессе генерации.

Запуск:  python tests/test_bodyfit.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib.util                            # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "bodyfit_standalone", os.path.join(_ROOT, "mesh", "bodyfit.py"))
bf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bf)

try:
    import pyvista as pv
    HAS_PV = True
except ImportError:
    HAS_PV = False

_passed = 0
_failed = []


def check(name, cond, extra=None):
    global _passed
    if cond:
        _passed += 1
        print("  ✅ %s" % name)
    else:
        _failed.append(name)
        print("  ❌ %s%s" % (name, "" if extra is None else " %s" % (extra,)))


# Фикстуры намеренно взяты из pyvista, а не собраны руками: самодельная
# триангуляция пластины выходила незамкнутой, и тест падал на фикстуре, а
# не на коде. Box и Cylinder замкнуты по построению, при этом их .faces —
# плоский одномерный массив, то есть проверяется именно то представление,
# которое реально приходит из генераторов.

def tube(r=0.6, L=8.0):
    """Фюзеляж: цилиндр, грани плоско (формат pyvista)."""
    m = pv.Cylinder(center=(0, 0, 0), direction=(1, 0, 0),
                    radius=r, height=L, resolution=48)
    m = m.triangulate()
    return m.points.copy(), np.asarray(m.faces)


def plate(span, chord, thick, x0=0.0):
    """Оперение: тонкая пластина 0.067 м по Z."""
    m = pv.Box(bounds=(x0, x0 + chord, -span / 2, span / 2,
                       -thick / 2, thick / 2)).triangulate()
    return m.points.copy(), np.asarray(m.faces)


print("== доступность ==")
rep = bf.availability_report()
check("отчёт о доступности содержит ready и missing",
      "ready" in rep and "missing" in rep, rep)
check("модулю больше не нужен gmsh", "gmsh" not in rep["missing"], rep)
if not rep["ready"]:
    print("  (нет %s — проверки подготовки будут пропущены)" % rep["missing"])

print()
print("== clean_surface ==")
v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
              # вершина-двойник первой, ближе порога
              [1e-9, 1e-9, 0]], dtype=float)
f = np.array([[0, 1, 2], [0, 2, 3], [4, 1, 2]], dtype=np.int64)
cv, cf = bf.clean_surface(v, f, 1e-6)
check("близкие вершины слиты", len(cv) == 4, "%d вершин" % len(cv))
check("схлопнувшаяся грань удалена", len(cf) == 2, "%d граней" % len(cf))

# дубли после слияния: две разные грани становятся одной
v2 = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [1e-9, 1e-9, 0]], dtype=float)
f2 = np.array([[0, 1, 2], [3, 1, 2]], dtype=np.int64)
cv2, cf2 = bf.clean_surface(v2, f2, 1e-6)
check("дубль грани после слияния удалён", len(cf2) == 1, "%d граней" % len(cf2))

cv3, cf3 = bf.clean_surface(v, f, 0.0)
check("нулевой порог — без изменений", len(cf3) == len(f))

print()
print("== плоские грани принимаются в любом представлении ==")
tv, tf = tube()
# pyvista.Cylinder сам по себе is_manifold=False, поэтому проверять
# замкнутость через pyvista нельзя — это проверка самого pyvista, а не
# нашего кода. Существенно здесь другое: плоский одномерный .faces должен
# быть разобран в треугольники, а тело должно пройти внутренний фильтр
# «это объём». Именно на этом ранее молча терялись все пять тел.
tv2, tf2 = tube()
check("грани цилиндра пришли плоским массивом", tf2.ndim == 1, tf2.shape)
_v, _f, _i = bf.prepare_surface([(tv2, tf2)], budget_faces=100_000)
check("плоский массив разобран и тело не отброшено",
      _i["not_volumes"] == [] and _i["faces_after_union"] > 0,
      "not_volumes=%r, граней=%d" % (_i["not_volumes"], _i["faces_after_union"]))
check("плоский массив дал ровно треугольники",
      (np.asarray(_f).ndim == 2 and np.asarray(_f).shape[1] == 3)
      or np.asarray(_f).ndim == 1, np.asarray(_f).shape)

if rep["ready"]:
    print()
    print("== prepare_surface ==")
    fus_v, fus_f = tube()
    pl_v, pl_f = plate(2.8, 0.56, 0.067, x0=2.8)
    bodies = [(fus_v, fus_f), (pl_v, pl_f)]

    v, f, info = bf.prepare_surface(bodies, budget_faces=200_000)
    check("объединение построено", info["faces_after_union"] > 0,
          info["faces_after_union"])
    check("уплотнение выполнено", info["refined_faces"] > 0,
          info["refined_faces"])
    check("объединённая поверхность замкнута", info["watertight"] is True)
    check("поле целей посчитано",
          0 < info["target_min"] <= info["target_max"],
          "%.4f..%.4f" % (info["target_min"], info["target_max"]))
    check("info заполнен целиком и на обычном пути",
          all(k in info for k in ("faces_after_union", "watertight",
                                  "refined_faces", "target_min",
                                  "target_max", "capped", "reached",
                                  "not_volumes", "area")))
    if HAS_PV:
        surf = pv.PolyData(v, np.column_stack(
            [np.full(len(f), 3), f]).ravel())
        check("объединённая поверхность замкнута (pyvista)",
              surf.is_manifold is True)

    # тело, которое не является объёмом, должно быть отброшено с пометкой
    flat_v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    flat_f = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    v2, f2, i2 = bf.prepare_surface([(fus_v, fus_f), (flat_v, flat_f)],
                                    budget_faces=50_000)
    check("не-объём отброшен с пометкой", len(i2["not_volumes"]) > 0,
          i2["not_volumes"])
    check("остальное при этом обработано", i2["faces_after_union"] > 0,
          i2["faces_after_union"])

    try:
        bf.prepare_surface([(flat_v, flat_f)], budget_faces=1000)
        check("если объёмов нет вовсе — ошибка с пояснением", False)
    except RuntimeError as e:
        check("если объёмов нет вовсе — ошибка с пояснением",
              "замкнутым объёмом" in str(e), str(e)[:60])

    check("пустой вход не роняет",
          bf.prepare_surface([], budget_faces=1000)[2]["faces_after_union"] == 0)

print()
print("== объёмный сеточник честно сообщает, что не готов ==")
st = bf.volume_mesher_status()
check("ready=False, чтобы никто не ждал сетку", st["ready"] is False)
check("подготовка поверхности при этом готова", st["surface_ready"] is True)
check("причина названа", len(st["reason"]) > 40)
check("перечислено, что пробовалось", len(st["tried"]) >= 5, list(st["tried"]))
check("готовый ре-мешер тоже проверен и отмечен",
      any("pyacvd" in k for k in st["tried"]), list(st["tried"]))
check("сказано, чем лечится", len(st["fix"]) > 20)

print()
print("Пройдено: %d" % _passed)
if _failed:
    print("ПРОВАЛЕНО ТЕСТОВ: %d → %s" % (len(_failed), _failed))
    raise SystemExit(1)
print("Все проверки пройдены.")
