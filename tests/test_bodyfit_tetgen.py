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


def test_make_graded_axis():
    """Ось: монотонна, покрывает [lo, hi], мелкая в центре, крупная по краям."""
    m = _load_bodyfit()
    ax = m.make_graded_axis(-33.0, 33.0, -11.0, 11.0, 0.135, 5.5)
    check("graded_axis монотонна", np.all(np.diff(ax) > 0))
    check("graded_axis покрывает диапазон", ax[0] <= -33.0 + 1e-9 and ax[-1] >= 33.0 - 1e-9)
    # Шаг в центре (_мелкий_) меньше шага на краях (_крупный_).
    n = len(ax)
    mid = np.diff(ax)[n // 2 - 3: n // 2 + 3]
    edge = np.diff(ax)[:3]
    check("в центре шаг мельче, чем на краях",
          float(mid.min()) < float(edge.max()),
          f"mid={mid.min():.4f}, edge={edge.max():.4f}")


def test_size_field_for_points():
    """Размер: h_near у поверхности, h_far вдали, монотонный без скачка."""
    m = _load_bodyfit()
    # Точка «тела» в начале координат.
    body_pts = np.array([[0.0, 0, 0], [1.0, 0, 0], [0.0, 1, 0]], dtype=float)
    h_near, h_far, L = 0.1, 6.0, 20.0
    pts = np.array([
        [0.0, 0, 0.0],    # на поверхности -> h_near
        [0.0, 0, 5.0],    # в переходе
        [0.0, 0, 30.0],   # за L -> h_far
    ], dtype=float)
    h = m.size_field_for_points(pts, body_pts, h_near, h_far, L)
    check("у поверхности размер ~h_near", abs(h[0] - h_near) < 1e-9, str(h))
    check("вдали размер ~h_far", abs(h[2] - h_far) < 1e-9, str(h))
    check("размер монотонно растёт с расстоянием", h[0] < h[1] < h[2], str(h))
    check("размер всегда в [h_near, h_far]", h.min() >= h_near and h.max() <= h_far)


def test_size_field_h_far_no_less_than_near():
    """Если h_far <= h_near — поле вырождается в константу (нет скачка)."""
    m = _load_bodyfit()
    body_pts = np.array([[0.0, 0, 0]], dtype=float)
    h = m.size_field_for_points([[0.0, 0, 0], [0.0, 0, 50.0]],
                                body_pts, 0.1, 0.1, 20.0)
    check("h_far<=h_near -> размер константен", np.all(np.abs(h - 0.1) < 1e-9), str(h))


def test_bg_grid_ordering():
    """Фоновая сетка: порядок точек = мировой C-порядок, 6 тетов на гекс,
    target_size не разъезжается с точками."""
    m = _load_bodyfit()
    axes = [np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([0.0, 1.0])]
    pts = m._structured_grid_pts(axes)
    check("решётка 2x2x2 = 8 точек", pts.shape == (8, 3), str(pts.shape))
    check("порядок точек: первая в начале координат", np.allclose(pts[0], [0, 0, 0]))
    check("порядок точек: последняя в противоположном углу",
          np.allclose(pts[-1], [1, 1, 1]))
    tets = m._structured_tet_cells(axes)
    check("на 1 гекс — 6 тетраэдров", tets.shape[0] == 6, str(tets.shape))
    check("индексы тетов в пределах числа точек",
          tets.min() >= 0 and tets.max() < len(pts))
    # target_size выровнен по порядку точек (тест на смещение поля).
    sizes = m.size_field_for_points(pts, np.array([[0.0, 0, 0]]), 0.1, 6.0, 20.0)
    check("target_size той же длины, что и точки",
          len(sizes) == len(pts))
    check("у ближней точки размер ~h_near", abs(sizes[0] - 0.1) < 1e-9)
    check("вдали размер больше, чем у тела", sizes[-1] > sizes[0])


if __name__ == "__main__":
    print("== test_bodyfit_tetgen ==")
    test_classify_exterior_blocks_body_faces()
    test_classify_exterior_no_seed_all_inside()
    test_four_face_keys()
    test_collect_airfoil_facets_uses_vmap()
    test_make_graded_axis()
    test_size_field_for_points()
    test_size_field_h_far_no_less_than_near()
    test_bg_grid_ordering()
    print("== " + ("OK" if not FAIL else f"FAIL {len(FAIL)}") + " ==")
    sys.exit(1 if FAIL else 0)
