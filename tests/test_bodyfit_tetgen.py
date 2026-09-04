# -*- coding: utf-8 -*-
"""
tests/test_bodyfit_tetgen.py — топологическая заливка снаружи для
телооблекающей сетки (mesh/bodyfit_tetgen.py).

Модуль numpy-only: загружается напрямую по пути (без mesh/__init__.py, чтобы
не тянуть pyvista/PyQt5) и проверяет самую важную часть — что заливка
НЕ пересекает грани тела и НА ПРОХОД ПО свободным граням. Именно этот признак
отделяет герметичную стенку (маркер airfoil без дыр) от дырявой.

Запуск:
    python tests/test_bodyfit_tetgen.py
"""

import importlib.util
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

FAIL = []


def check(name, cond, extra=""):
    if cond:
        print(f"  ✅ {name}")
    else:
        FAIL.append(name)
        print(f"  ❌ {name} {extra}")


def _load_bodyfit():
    path = os.path.join(ROOT, "mesh", "bodyfit_tetgen.py")
    spec = importlib.util.spec_from_file_location("bodyfit_tetgen_standalone", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_classify_exterior_blocks_body_faces():
    m = _load_bodyfit()
    # Два тетраэдра, делящие грань (0,1,2):
    #   A = (0,1,2,3)  — z>0
    #   B = (0,1,2,4)  — z<0
    points = np.array([[0.0, 0, 0],
                       [1.0, 0, 0],
                       [0.0, 1, 0],
                       [0.0, 0, 1],   # 3 (A)
                       [0.0, 0, -1]], dtype=float)  # 4 (B)
    tets = np.array([[0, 1, 2, 3], [0, 1, 2, 4]], dtype=np.int64)
    # bbox тела = только тетраэдр A (z>=0) → единственный «снаружи» семен — B
    bbox = (np.array([0.0, 0, 0]), np.array([1.0, 1, 0.999999]))

    # Общая грань (0,1,2) — грань ТЕЛА (блокирует): заливка не перейдёт на A.
    ext = m.classify_exterior_tets(points, tets, {(0, 1, 2)}, bbox)
    check("заливка не пересекает грань тела",
          ext.tolist() == [False, True], str(ext))

    # Общая грань НЕ является гранью тела → заливка проходит, оба снаружи.
    ext2 = m.classify_exterior_tets(points, tets, set(), bbox)
    check("заливка проходит по свободной грани",
          ext2.tolist() == [True, True], str(ext2))


def test_classify_exterior_no_seed_all_inside():
    """Если ни один тетраэдр не снаружи по bbox — заливка ничего не метит."""
    m = _load_bodyfit()
    points = np.array([[0.0, 0, 0],
                       [1.0, 0, 0],
                       [0.0, 1, 0],
                       [0.0, 0, 1]], dtype=float)
    tets = np.array([[0, 1, 2, 3]], dtype=np.int64)
    # bbox строго охватывает этот единственный тетраэдр → семян нет.
    bbox = (np.array([0.0, 0, 0]), np.array([1.0, 1, 1]))
    ext = m.classify_exterior_tets(points, tets, set(), bbox)
    check("нет семян → наружных нет (тело целиком внутри bbox)",
          ext.tolist() == [False], str(ext))


def test_four_face_keys():
    m = _load_bodyfit()
    keys = m._tet_face_keys((0, 1, 2, 3))
    check("у тетраэдра 4 сортированные грани",
          len(keys) == 4 and (0, 1, 2) in keys and (1, 2, 3) in keys,
          str(keys))


def test_collect_airfoil_facets_uses_vmap():
    """Грани тела переводятся в узлы сетки ЧЕРЕЗ vmap.

    Раньше здесь был remap[x] для x в body_faces, но x — индекс ВХОДНОЙ
    поверхности, а не узла сетки. Когда TetGen перенумеровывает узлы
    (входные точки не остаются в начале массива), грани тела перестают
    совпадать с реальной границей наружной сетки, маркер airfoil терял
    ~20% граней, и телооблекающая сетка ложно отвергалась как дырявая.
    """
    m = _load_bodyfit()
    # Полная объёмная сетка из 8 узлов. Тело — тетраэдр на узлах 4,5,6,7
    # (то есть vmap НЕ тождественный: входная вершина k -> узел k+4).
    points = np.array([
        [1.0, 1.0, 1.0],   # 0 — наружный
        [-1.0, 1.0, 1.0],  # 1 — наружный
        [1.0, -1.0, 1.0],  # 2 — наружный
        [1.0, 1.0, -1.0],  # 3 — наружный
        [0.0, 0.0, 0.0],   # 4 — вершина тела A
        [1.0, 0.0, 0.0],   # 5 — вершина тела B
        [0.0, 1.0, 0.0],   # 6 — вершина тела C
        [0.0, 0.0, 1.0],   # 7 — вершина тела D
    ], dtype=float)
    # Тет 0 — «тело» (внутренний), теты 1..4 — наружные, каждый примыкает
    # к одной грани тела и потому делает её границей наружной области.
    tets = np.array([
        [4, 5, 6, 7],   # 0: тело (внутренний)
        [4, 5, 6, 0],   # 1: наружный, грань (4,5,6)
        [4, 5, 7, 1],   # 2: наружный, грань (4,5,7)
        [4, 6, 7, 2],   # 3: наружный, грань (4,6,7)
        [5, 6, 7, 3],   # 4: наружный, грань (5,6,7)
    ], dtype=np.int64)
    # Входная поверхность тела: 4 грани тетраэдра (индексы входных вершин 0..3).
    body_faces = np.array([
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3],
    ], dtype=np.int64)
    vmap = np.array([4, 5, 6, 7], dtype=np.int64)  # входная k -> узел k+4
    ext = np.array([1, 2, 3, 4], dtype=np.int64)   # наружные теты (тело вырезано)

    marker, ratio = m.collect_airfoil_facets(points, tets, ext, vmap, body_faces)
    check("все 4 грани тела вышли на границу (vmap-перенумерация)",
          len(marker) == 4 and ratio > 0.999,
          f"marker={len(marker)}, ratio={ratio:.3f}")

    # Доказываем, что СТАРЫЙ путь (remap[x], без vmap) дал бы дыры.
    ext_tets = tets[ext]
    used = np.unique(ext_tets.ravel())
    remap = np.full(len(points), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    flat = m.tet_faces(remap[ext_tets])[0]
    uniq, cnt = np.unique(flat, axis=0, return_counts=True)
    boundary = {tuple(int(x) for x in r) for r in uniq[cnt == 1]}
    old_bad = [tuple(sorted(int(remap[x]) for x in f)) for f in body_faces]
    old_marker = [f for f in old_bad if f in boundary]
    check("старый путь (remap[x]) потерял бы грани — регрессия поймана",
          len(old_marker) < 4,
          f"old_marker={len(old_marker)}")


if __name__ == "__main__":
    print("== test_bodyfit_tetgen ==")
    test_classify_exterior_blocks_body_faces()
    test_classify_exterior_no_seed_all_inside()
    test_four_face_keys()
    test_collect_airfoil_facets_uses_vmap()
    print("== " + ("OK" if not FAIL else f"FAIL {len(FAIL)}") + " ==")
    sys.exit(1 if FAIL else 0)
