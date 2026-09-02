# -*- coding: utf-8 -*-
"""Телооблекающая объёмная сетка (mesh/bodyfit_tetgen.py).

Модули грузятся по пути файла: mesh/__init__.py тянет PyQt5 через
mesh_worker, а эти модули должны работать и в тестах, и в фоновом
процессе генерации.

Запуск:  python tests/test_bodyfit_tetgen.py
"""
import os
import sys
import importlib.util

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "tests"))


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_ROOT, relpath))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bt = _load("bodyfit_tetgen_standalone", os.path.join("mesh", "bodyfit_tetgen.py"))

try:
    import pyvista as pv
    HAS_PV = True
except ImportError:
    HAS_PV = False

FAIL = []
N = [0]


def check(name, cond, extra=""):
    N[0] += 1
    if cond:
        print("  [OK]   %s" % name)
    else:
        print("  [FAIL] %s %s" % (name, extra))
        FAIL.append(name)


def tet_volume(points, tets):
    p = points[np.asarray(tets)]
    return float(np.abs(np.einsum(
        "ij,ij->i",
        np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]),
        p[:, 3] - p[:, 0])).sum() / 6.0)


def parse_su2(path):
    """Разобрать mesh.su2 -> (n_tets, n_points, {маркер: число элементов})."""
    n_tets = n_points = 0
    markers = {}
    with open(path, "r", encoding="ascii", errors="replace") as f:
        lines = f.read().split("\n")
    i = 0
    while i < len(lines):
        s = lines[i].split("%")[0].strip()
        if s.startswith("NELEM="):
            n_tets = int(s.split("=")[1])
        elif s.startswith("NPOIN="):
            n_points = int(s.split("=")[1])
        elif s.startswith("MARKER_TAG="):
            tag = s.split("=")[1].strip()
            i += 1
            m = lines[i].split("%")[0].strip()
            markers[tag] = int(m.split("=")[1])
            i += markers[tag]
        i += 1
    return n_tets, n_points, markers


# ---------------------------------------------------------------- подготовка
print("Телооблекающая сетка (TetGen)")
if not (HAS_PV and bt.tetgen_available()):
    print("  Пропуск: нет pyvista/tetgen/trimesh — проверяется только "
          "корректный отказ")
    check("без TetGen возвращается None",
          bt.build_body_fitted_grid([], [0, 0, 0], [1, 1, 1], 1.0) is None)
else:
    R = 1.0
    sphere = pv.Sphere(radius=R, theta_resolution=40, phi_resolution=40)
    body_min = np.asarray(sphere.points).min(axis=0)
    body_max = np.asarray(sphere.points).max(axis=0)
    body_size = float(np.max(body_max - body_min))
    margin = body_size * 2.0
    v_body = 4.0 / 3.0 * np.pi * R ** 3

    logs = []
    res = bt.build_body_fitted_grid([sphere], body_min, body_max, margin,
                                    log=logs.append)

    check("сетка построена", res is not None)
    if res is None:
        print("\n".join(logs))
    else:
        grid = res["grid"]
        bounds = res["bounds"]
        v_box = ((bounds[1] - bounds[0]) * (bounds[3] - bounds[2])
                 * (bounds[5] - bounds[4]))
        check("сохранено >= 90%% граней тела (%.2f%%)"
              % (100.0 * res["recovery"]), res["recovery"] >= 0.90)
        check("грани тела для маркера непустые (%d)" % len(res["body_facets"]),
              len(res["body_facets"]) > 0)

        tets = np.asarray(grid.cells).reshape(-1, 5)[:, 1:]
        pts = np.asarray(grid.points)
        v_ext = tet_volume(pts, tets)
        err = abs(v_ext - (v_box - v_body))
        check("объём наружной области сходится с короб минус тело "
              "(%.4f против %.4f, откл %.2e)" % (v_ext, v_box - v_body, err),
              err < 1e-3 * (v_box - v_body))

        # Каждая грань маркера обязана быть границей сетки (входить ровно
        # в один тетраэдр) — иначе SU2 получит висячий маркер.
        flat, _ = bt.tet_faces(tets)
        uniq, cnt = np.unique(flat, axis=0, return_counts=True)
        on_bound = {tuple(int(x) for x in r) for r in uniq[cnt == 1]}
        bad = [f for f in res["body_facets"] if tuple(int(x) for x in f)
               not in on_bound]
        check("все грани маркера лежат на границе сетки (нарушителей %d)"
              % len(bad), len(bad) == 0)

        # --- Негативный контроль: ключ Y (nobisect) действительно нужен.
        # Без него TetGen режет входные грани, и восстановление падает.
        bpts, bfaces = bt.union_surfaces([sphere], log=lambda *_: None)
        pts_ny, tets_ny = None, None
        try:
            from tetgen import TetGen
            box = pv.Box(bounds=bounds).triangulate()
            bxp, bxf = bt.to_triangles(box)
            P = np.vstack([bpts, bxp])
            F = np.vstack([bfaces, bxf + len(bpts)])
            tg = TetGen(P, F)
            tg.tetrahedralize(order=1, verbose=0, switches="pq")
            pts_ny = np.asarray(tg.grid.points)
            tets_ny = np.asarray(tg.grid.cells).reshape(-1, 5)[:, 1:]
        except Exception as e:
            print("  Негативный контроль без Y не выполнен: %s" % e)
        if pts_ny is not None:
            rec_ny = bt.count_recovered(pts_ny, tets_ny, bpts, bfaces)[0]
            rec_y = bt.count_recovered(pts, tets, bpts, bfaces)[0]
            check("негативный контроль: без ключа Y граней сохраняется меньше "
                  "(%d против %d)" % (rec_ny, rec_y), rec_ny < rec_y)

    # ------------------------------------------- короб расчётной области
    print("Короб расчётной области")
    bnd = bt.farfield_bounds(body_min, body_max, margin)
    bp, bf = bt.box_surface(bnd, (bnd[1] - bnd[0]) / 12.0)
    tri = bp[bf]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    cen = tri.mean(axis=1)
    ctr = np.array([(bnd[0] + bnd[1]) / 2.0, (bnd[2] + bnd[3]) / 2.0,
                    (bnd[4] + bnd[5]) / 2.0])
    inward = int((np.einsum("ij,ij->i", nrm, cen - ctr) < 0).sum())
    check("все нормали короба направлены наружу (внутрь %d)" % inward,
          inward == 0)
    # У замкнутой поверхности каждый треугольник встречается один раз, а
    # каждое ребро принадлежит ровно двум треугольникам.
    edges = np.sort(np.vstack([bf[:, [0, 1]], bf[:, [1, 2]],
                               bf[:, [0, 2]]]), axis=1)
    eu, ec = np.unique(edges, axis=0, return_counts=True)
    check("короб замкнут: у каждого ребра ровно 2 треугольника "
          "(рёбер %d, нарушителей %d)" % (len(eu), int((ec != 2).sum())),
          int((ec != 2).sum()) == 0)
    check("в коробе нет повторяющихся треугольников",
          len(np.unique(np.sort(bf, axis=1), axis=0)) == len(bf))
    v_exp = ((bnd[1] - bnd[0]) * (bnd[3] - bnd[2]) * (bnd[5] - bnd[4]))
    v_box_surf = float(np.einsum(
        "ij,ij->i", tri[:, 0],
        np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0)
    check("знаковый объём короба совпадает с габаритами (%.4f против %.4f)"
          % (v_box_surf, v_exp), abs(v_box_surf - v_exp) < 1e-6 * v_exp)

    # ------------------------------------------------------- сквозной прогон
    print("Сквозной прогон generate_mesh_impl")
    gg = _load("gmsh_generator_standalone", os.path.join("mesh", "gmsh_generator.py"))
    import shutil
    tmp = os.path.join(_ROOT, "tests", "_tmp_bodyfit")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)
    stl = os.path.join(tmp, "sphere.stl")
    sphere.save(stl)
    gg.MESH_FILE = os.path.join(tmp, "mesh.su2")
    gg.PREVIEW_MESH = os.path.join(tmp, "preview.vtk")

    ok, msg = gg.generate_mesh_impl([stl], quality_text="Грубая (быстро)")
    check("generate_mesh_impl завершился успешно (%s)" % msg, bool(ok))
    if ok and os.path.exists(gg.MESH_FILE):
        n_tets, n_p, markers = parse_su2(gg.MESH_FILE)
        check("в mesh.su2 есть тетраэдры (%d)" % n_tets, n_tets > 100)
        check("в mesh.su2 есть точки (%d)" % n_p, n_p > 100)
        check("маркер airfoil непустой (%d)" % markers.get("airfoil", 0),
              markers.get("airfoil", 0) > 100)
        check("маркер farfield непустой (%d)" % markers.get("farfield", 0),
              markers.get("farfield", 0) > 100)
        n_sphere = int(sphere.n_cells)
        frac = markers.get("airfoil", 0) / float(n_sphere)
        check("airfoil сопоставим с поверхностью тела (%d против %d граней "
              "сферы, доля %.2f)" % (markers.get("airfoil", 0), n_sphere, frac),
              frac > 0.30)

        # --- Негативный контроль: маркер собирается не со ступеньки.
        # Если телооблекающий путь отключить, генератор вернётся к
        # картезианскому фону, и число граней airfoil изменится.
        # generate_mesh_impl импортирует модуль по имени
        # "mesh.bodyfit_tetgen", а не объект bt, поэтому подменяем запись
        # в sys.modules — иначе патч не дойдёт до генератора.
        import types
        _saved_mod = sys.modules.get("mesh.bodyfit_tetgen")
        _stub = types.ModuleType("mesh.bodyfit_tetgen")
        _stub.build_body_fitted_grid = lambda *a, **k: None
        gg_ok2 = False
        m2 = {}
        try:
            sys.modules["mesh.bodyfit_tetgen"] = _stub
            gg_ok2, _ = gg.generate_mesh_impl([stl],
                                              quality_text="Грубая (быстро)")
            if gg_ok2:
                m2 = parse_su2(gg.MESH_FILE)[2]
        except Exception as e:
            print("  Негативный контроль: картезианский путь не запустился "
                  "(%s: %s)" % (type(e).__name__, e))
        finally:
            if _saved_mod is None:
                sys.modules.pop("mesh.bodyfit_tetgen", None)
            else:
                sys.modules["mesh.bodyfit_tetgen"] = _saved_mod
        if gg_ok2 and m2:
            check("негативный контроль: картезианский путь даёт другой "
                  "маркер (%d против %d)"
                  % (m2.get("airfoil", 0), markers.get("airfoil", 0)),
                  m2.get("airfoil", 0) != markers.get("airfoil", 0))
        else:
            print("  Негативный контроль картезианского пути не выполнен")

# ---------------------------------------------------------------- cleanup
try:
    import shutil
    shutil.rmtree(os.path.join(_ROOT, "tests", "_tmp_bodyfit"),
                  ignore_errors=True)
except Exception:
    pass

# ---------------------------------------------------------------- summary
print()
print("Проверок: %d" % N[0])
if FAIL:
    print("ПРОВАЛЕНО: %d -> %s" % (len(FAIL), FAIL))
    sys.exit(1)
print("ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ")
