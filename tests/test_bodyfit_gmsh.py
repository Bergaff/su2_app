# -*- coding: utf-8 -*-
"""
tests/test_bodyfit_gmsh.py — gmsh-телооблегающая сетка (mesh/bodyfit_gmsh.py).

gmsh в песочнице не запускается (нет libGLU), поэтому здесь проверяется то,
что не зависит от gmsh: запись бинарного STL, параметры поля размера,
граничные грани, принадлежность граней к поверхности тела и мера покрытия.
Сам вызов gmsh (_gmsh_mesh) проверяется вручную на машине пользователя.

Запуск:  python tests/test_bodyfit_gmsh.py
"""
import importlib.util
import os
import struct
import sys
import tempfile

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
    path = os.path.join(ROOT, "mesh", "bodyfit_gmsh.py")
    spec = importlib.util.spec_from_file_location("bodyfit_gmsh_standalone", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_stl_roundtrip(m):
    pts = np.array([[0.0, 0, 0], [1.0, 0, 0], [0.0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    path = tempfile.mktemp(suffix=".stl")
    n = m._write_binary_stl(pts, faces, path)
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        header = f.read(80)
        count = struct.unpack("<I", f.read(4))[0]
    os.remove(path)
    check("STL: 1 грань записана", n == 1 and count == 1)
    check("STL: бинарный формат (80+4+50 байт)", size == 80 + 4 + 50,
          "size=%d" % size)
    check("STL: пустой заголовок", len(header) == 80)


def test_size_field_params(m):
    bounds = (-28.0, 42.62, -34.45, 34.45, -30.35, 30.35)
    body_min = np.array([-5.0, -5.6, -1.5])
    body_max = np.array([8.0, 5.6, 1.5])
    hn, hf, L, samp = m._size_field_params(bounds, body_min, body_max, 0.3462)
    check("h_near == target_edge", abs(hn - 0.3462) < 1e-9, "hn=%f" % hn)
    check("h_far >= h_near", hf > hn, "hn=%f hf=%f" % (hn, hf))
    check("h_far ~ extent/12", abs(hf - (max(42.62 + 28.0, 34.45 * 2, 30.35 * 2) / 12.0)) < 0.02,
          "hf=%f" % hf)
    check("переход L в (0, extent]", 0 < L <= 0.55 * 70.62, "L=%f" % L)
    check("sampling целое", isinstance(samp, int) and samp > 0)


def test_size_field_params_floor(m):
    # target_edge слишком мал -> пол по габариту.
    hn, hf, L, _ = m._size_field_params((-1, 1, -1, 1, -1, 1), [-0.2, -0.2, -0.2],
                                        [0.2, 0.2, 0.2], 1e-6)
    check("h_near не ниже пола", hn >= 2.0 * 0.002, "hn=%g" % hn)


def test_boundary_faces(m):
    # Два тетраэдра, общая грань (1,2,3) — на границе она не считается.
    tets = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int64)
    bf = m._boundary_faces(tets)
    check("границ у 2 тетов = 6 (общая грань убрана)", bf.shape[0] == 6,
          "got %d" % bf.shape[0])
    uniq = set(map(tuple, bf.tolist()))
    check("общая грань (1,2,3) не на границе", (1, 2, 3) not in uniq)


def test_faces_on_body(m):
    pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [2, 2, 2]],
                   dtype=float)
    tets = np.array([[0, 1, 2, 3]], dtype=np.int64)
    body = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    foo = m._faces_on_body(pts, tets, body, 1e3)
    check("все 4 грани тела распознаны", foo.shape[0] == 4, "got %d" % foo.shape[0])


def test_exterior_index_topology(m):
    """Заливка снаружи: на замкнутом теле наружные тетраэдры отделены."""
    # Два тетраэдра, делящие грань (0,1,2): A=внутри (z>0), B=снаружи.
    pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                    [0, 0, 1], [0, 0, -1]], dtype=float)
    tets = np.array([[0, 1, 2, 3], [0, 1, 2, 4]], dtype=np.int64)
    bbox = (np.array([0.0, 0, 0]), np.array([1.0, 1, 0.999999]))
    face_keys = {tuple(sorted(row)) for row in [(0, 1, 2)]}
    body_faces = np.array([[0, 1, 2]], dtype=np.int64)
    ext = m._exterior_index(pts, tets, face_keys, bbox,
                            np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
                            body_faces, log=lambda *a, **k: None)
    check("заливка помечает снаружи только нижний тетраэдр",
          ext is not None and ext.tolist() == [1], str(ext))


def test_coverage(m):
    pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [2, 2, 2]],
                   dtype=float)
    tets = np.array([[0, 1, 2, 3]], dtype=np.int64)
    body_pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    body_faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], int)
    cov = m._coverage(pts, tets, np.array([0]), body_pts, body_faces, 1.0)
    check("покрытие тела = 1.0", abs(cov - 1.0) < 1e-9, "cov=%f" % cov)


def test_gmsh_unavailable_fallback(m):
    """Если gmsh недоступен, build_body_fitted_grid_gmsh возвращает None."""
    if m.HAS_GMSH:
        # В среде с рабочим gmsh проверять фолбэк нельзя — пропускаем.
        print("   (gmsh доступен — тест фолбэка пропущен)")
        return
    res = m.build_body_fitted_grid_gmsh([], np.array([0, 0, 0.]),
                                        np.array([1, 1, 1.]), 1.0,
                                        log=lambda *a, **k: None)
    check("без gmsh -> None (фолбэк на TetGen)", res is None)


if __name__ == "__main__":
    print("== test_bodyfit_gmsh ==")
    m = _load()
    test_stl_roundtrip(m)
    test_size_field_params(m)
    test_size_field_params_floor(m)
    test_boundary_faces(m)
    test_faces_on_body(m)
    test_exterior_index_topology(m)
    test_coverage(m)
    test_gmsh_unavailable_fallback(m)
    print("== " + ("OK" if not FAIL else "FAIL %d" % len(FAIL)) + " ==")
    sys.exit(1 if FAIL else 0)
