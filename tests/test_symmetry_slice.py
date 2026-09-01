# -*- coding: utf-8 -*-
"""Сквозная проверка маркеров при резке модели по плоскости симметрии.

Отдельный файл, а не часть test_backend.py: tests/qt_stubs.py безусловно
подменяет sys.modules["pyvista"] заглушкой, а здесь нужен настоящий
pyvista, чтобы собрать сетку и поверхность.

Проверяется сценарий, из-за которого SU2 расходился: крыло лежит в
плоскости XY (Z=0), то есть в той самой плоскости, по которой режем.
По одной геометрии его треугольники неотличимы от грани среза — у них и
координата на плоскости, и нормаль вдоль неё.

Запуск:  python tests/test_symmetry_slice.py
"""
import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import numpy as np
    import pyvista as pv
    import vtk
except ImportError as exc:                      # pragma: no cover
    print("ПРОПУЩЕНО: нет pyvista/vtk (%s)" % exc)
    raise SystemExit(0)

# mesh/__init__.py тянет PyQt5, которого здесь нет. Загружаем модуль
# напрямую по пути файла, минуя пакет.
import importlib.util                             # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "gmsh_generator_standalone",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "mesh", "gmsh_generator.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
write_su2 = _mod.write_su2

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


def quads(rects, z):
    """Прямоугольники в плоскости z -> список точек и треугольники."""
    pts, tris = [], []
    for (x0, x1, y0, y1) in rects:
        b = len(pts)
        pts += [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)]
        tris += [[b, b + 1, b + 2], [b, b + 2, b + 3]]
    return pts, tris


def build_case():
    """Срез по Z=0: грань среза, верх крыла, дальняя граница, боковая грань."""
    cut_pts, cut_tris = quads([(-0.7, 0.7, -4.5, 4.5)], 0.0)     # появился при резке
    wing_pts, wing_tris = quads([(-0.7, 0.7, -4.5, 4.5)], 0.075)  # был до резки
    far_pts, far_tris = quads([(-40, 48, -40, 40)], 40.0)
    side_pts = [(-40, -40, 0), (-40, 40, 0), (-40, 40, 40), (-40, -40, 40)]
    side_tris = [[0, 1, 2], [0, 2, 3]]

    pts, tris, off = [], [], 0
    for pp, tt in ((cut_pts, cut_tris), (wing_pts, wing_tris),
                   (far_pts, far_tris), (side_pts, side_tris)):
        tris += [[i + off for i in t] for t in tt]
        pts += pp
        off = len(pts)

    faces = []
    for t in tris:
        faces += [3] + list(t)
    surface = pv.PolyData(np.array(pts, dtype=float),
                          np.array(faces, dtype=np.int64))

    # Сетка после резки: Z начинается с 0, поэтому z_min совпадает с
    # плоскостью симметрии — ровно как в настоящем генераторе.
    grid_pts = np.array([[-40, -40, 0], [48, -40, 0], [48, 40, 0], [-40, 40, 0],
                         [-40, -40, 40], [48, -40, 40], [48, 40, 40], [-40, 40, 40]],
                        dtype=float)
    tets = np.array([[0, 1, 2, 4], [1, 2, 4, 5], [2, 4, 5, 6], [4, 5, 6, 7]])
    ug = vtk.vtkUnstructuredGrid()
    vp = vtk.vtkPoints()
    for p3 in grid_pts:
        vp.InsertNextPoint(*p3)
    ug.SetPoints(vp)
    for t in tets:
        ids = vtk.vtkIdList()
        for v in t:
            ids.InsertNextId(int(v))
        ug.InsertNextCell(10, ids)             # 10 = VTK_TETRA
    grid = pv.UnstructuredGrid(ug)

    # До резки существовали крыло и дальнее поле; точек среза не было.
    pre = np.array(list(wing_pts) + list(far_pts) + list(side_pts), dtype=float)
    return grid, surface, pre, grid_pts


def run(grid, surface, **kw):
    out = os.path.join(tempfile.mkdtemp(), "mesh.su2")
    try:
        write_su2(grid, surface, out, use_symmetry=True,
                  symmetry_planes=["xy"], **kw)
    except RuntimeError as exc:
        return None, str(exc).split("\n")[0]
    text = open(out, encoding="utf-8").read()
    marks = dict((a, int(b)) for a, b in
                 re.findall(r"MARKER_TAG=\s*(\S+)\nMARKER_ELEMS=\s*(\d+)", text))
    return marks, None


def main():
    grid, surface, pre, grid_pts = build_case()
    span = float((grid_pts.max(0) - grid_pts.min(0)).max())
    print("== резка по плоскости, в которой лежит крыло ==")
    print("   поверхность: %d треуг., bbox=%.1f м, tol=%.3f м, крыло на z=0.075 м"
          % (surface.n_cells, span, span * 0.002))

    before, before_err = run(grid, surface)
    after, after_err = run(grid, surface, pre_clip_points=pre)

    check("без проверки происхождения маркер стенки пуст (баг воспроизведён)",
          before_err is not None and "airfoil" in before_err, before_err)
    check("с проверкой происхождения write_su2 не падает",
          after_err is None, after_err)

    if after:
        check("стенка самолёта получила свои треугольники",
              after.get("airfoil", 0) == 2, after)
        check("грань среза отнесена к симметрии",
              after.get("symmetry_xy", 0) == 2, after)
        check("дальняя граница не смешалась со стенкой",
              after.get("farfield", 0) == 4, after)
        check("все треугольники распределены",
              sum(after.values()) == surface.n_cells,
              "%d из %d" % (sum(after.values()), surface.n_cells))
        # В config.cfg не должно быть пустого маркера: SU2 падает с
        # «MARKER_SYM not found in mesh».
        for tag, n in after.items():
            check("маркер %s непустой" % tag, n > 0, n)

    print()
    print("Пройдено: %d" % _passed)
    if _failed:
        print("ПРОВАЛЕНО ТЕСТОВ: %d → %s" % (len(_failed), _failed))
        raise SystemExit(1)
    print("Все проверки пройдены.")


if __name__ == "__main__":
    main()
