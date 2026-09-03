# -*- coding: utf-8 -*-
"""Сквозная проверка маркеров при резке модели по плоскости симметрии.

Отдельный файл, а не часть test_backend.py: tests/qt_stubs.py безусловно
подменяет sys.modules["pyvista"] заглушкой, а здесь нужен настоящий
pyvista, чтобы собрать сетку и поверхность.

Проверяется сценарий, из-за которого SU2 расходился: крыло лежит в
плоскости XY (Z=0), то есть в той самой плоскости, по которой режем.
По одной геометрии его треугольники неотличимы от грани среза — у них и
координата на плоскости, и нормаль вдоль неё.

Второй сценарий — обратный: срез дальнего поля. Коробка расчётной
области симметрична относительно плоскости реза, поэтому слой её узлов
лежит ровно на плоскости и clip() переиспользует эти точки как старые.
Признак «все вершины созданы резкой» такие грани отвергает, они уходят в
airfoil, и SU2 получает стенку и монитор сил на плоской плите в
невозмущённом потоке. На сгенерированном самолёте это дало Cd=0.184 и
Cm=9.23 при норме ~0.02 и ~0.1.

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
    # Срез дальнего поля: его вершины лежат на плоскости реза, но
    # существовали и до резки — узлы симметричной коробки области.
    oldcut_pts, oldcut_tris = quads([(-20, 20, -20, 20)], 0.0)
    far_pts, far_tris = quads([(-40, 48, -40, 40)], 40.0)
    side_pts = [(-40, -40, 0), (-40, 40, 0), (-40, 40, 40), (-40, -40, 40)]
    side_tris = [[0, 1, 2], [0, 2, 3]]

    pts, tris, off = [], [], 0
    for pp, tt in ((cut_pts, cut_tris), (wing_pts, wing_tris),
                   (oldcut_pts, oldcut_tris),
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
    pre = np.array(list(wing_pts) + list(oldcut_pts) + list(far_pts)
                   + list(side_pts), dtype=float)
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


def outward_violations(path):
    """Сколько граней маркеров записано с обходом внутрь тела.

    SU2 берёт нормаль грани маркера из порядка её вершин и больше
    ниоткуда, поэтому обход обязан давать внешнюю нормаль. Внешняя
    нормаль граничной грани тетраэдральной сетки определяется точно:
    она смотрит от четвёртой вершины тетраэдра наружу.
    """
    txt = open(path, encoding="utf-8").read().split("\n")
    i, pts, tets, marks = 0, None, [], []
    while i < len(txt):
        t = txt[i].split("%")[0].strip()
        if t.startswith("NELEM="):
            n = int(t.split("=")[1])
            tets += [tuple(int(x) for x in txt[i + 1 + k].split()[1:5])
                     for k in range(n)]
            i += n
        elif t.startswith("NPOIN="):
            n = int([x for x in t.replace("=", " ").split() if x][-1])
            pts = np.array([[float(v) for v in txt[i + 1 + k].split()[:3]]
                            for k in range(n)])
            i += n
        elif t.startswith("MARKER_TAG="):
            k = int(txt[i + 1].split("%")[0].strip().split("=")[1])
            marks += [tuple(int(x) for x in txt[i + 2 + j].split()[1:4])
                      for j in range(k)]
            i += k + 1
        i += 1
    out = {}
    T = np.asarray(tets)
    for miss in range(4):
        f = T[:, [j for j in range(4) if j != miss]]
        p3 = pts[f]
        n = np.cross(p3[:, 1] - p3[:, 0], p3[:, 2] - p3[:, 0])
        flip = np.einsum('ij,ij->i', n,
                         p3.mean(axis=1) - pts[T[:, miss]]) < 0.0
        f = np.where(flip[:, None], f[:, [0, 2, 1]], f)
        q3 = pts[f]
        nv = np.cross(q3[:, 1] - q3[:, 0], q3[:, 2] - q3[:, 0])
        for k, v in zip(map(tuple, np.sort(f, axis=1)), nv):
            out[k] = v
    bad = 0
    for a, b, c in marks:
        v = out.get(tuple(sorted((a, b, c))))
        if v is None:
            continue
        q = pts[[a, b, c]]
        if np.dot(np.cross(q[1] - q[0], q[2] - q[0]), v) < 0:
            bad += 1
    return bad, len(marks)


def main():
    grid, surface, pre, grid_pts = build_case()
    span = float((grid_pts.max(0) - grid_pts.min(0)).max())
    print("== резка по плоскости, в которой лежит крыло ==")
    print("   поверхность: %d треуг., bbox=%.1f м, tol=%.3f м, крыло на z=0.075 м"
          % (surface.n_cells, span, span * 0.002))

    before, before_err = run(grid, surface)
    after, after_err = run(grid, surface, pre_clip_points=pre)

    check("без pre_clip_points write_su2 не падает",
          before_err is None, before_err)
    check("с pre_clip_points write_su2 не падает",
          after_err is None, after_err)

    if before:
        check("геометрия сама отделяет крыло от среза (airfoil=2)",
              before.get("airfoil", 0) == 2, before)
    if after:
        check("стенка самолёта получила свои треугольники",
              after.get("airfoil", 0) == 2, after)
        check("грань среза отнесена к симметрии (2 новых + 2 старых вершины)",
              after.get("symmetry_xy", 0) == 4, after)
        check("срез дальнего поля из старых вершин не ушёл в стенку",
              before == after, "%s против %s" % (before, after))
        check("дальняя граница не смешалась со стенкой",
              after.get("farfield", 0) == 4, after)
        check("все треугольники распределены",
              sum(after.values()) == surface.n_cells,
              "%d из %d" % (sum(after.values()), surface.n_cells))
        # В config.cfg не должно быть пустого маркера: SU2 падает с
        # «MARKER_SYM not found in mesh».
        for tag, n in after.items():
            check("маркер %s непустой" % tag, n > 0, n)

    # ---------------------------------------------------------------
    # Обход вершин грани маркера. SU2 берёт из него нормаль, поэтому
    # грань, записанная внутрь тела, даёт вклад в силу с обратным
    # знаком. На сгенерированном самолёте таких было 911 из 12367 в
    # airfoil (7.37%) и 17 в symmetry_xz: завышались и Cl, и Cd
    # одновременно (Cl=0.739, Cd=0.360, L/D=2.05).
    print()
    print("== обход вершин граней маркера ==")
    _fr = np.asarray(surface.faces).reshape(-1, 4)[:, [0, 2, 1, 3]]
    flipped = pv.PolyData(surface.points, _fr.reshape(-1))
    check("в поданной на запись поверхности обход действительно перевёрнут",
          not np.array_equal(np.asarray(flipped.faces),
                             np.asarray(surface.faces)))
    _out = os.path.join(tempfile.mkdtemp(), "mesh.su2")
    write_su2(grid, flipped, _out, use_symmetry=True,
              symmetry_planes=["xy"], pre_clip_points=pre)
    _bad, _tot = outward_violations(_out)
    check("запись разворачивает обход наружу (внутрь %d из %d граней)"
          % (_bad, _tot), _bad == 0 and _tot > 0)

    _out2 = os.path.join(tempfile.mkdtemp(), "mesh.su2")
    write_su2(grid, surface, _out2, use_symmetry=True,
              symmetry_planes=["xy"], pre_clip_points=pre)
    _bad2, _tot2 = outward_violations(_out2)
    check("правильный обход запись не портит (внутрь %d из %d)"
          % (_bad2, _tot2), _bad2 == 0 and _tot2 == _tot)

    print()
    print("Пройдено: %d" % _passed)
    if _failed:
        print("ПРОВАЛЕНО ТЕСТОВ: %d → %s" % (len(_failed), _failed))
        raise SystemExit(1)
    print("Все проверки пройдены.")


if __name__ == "__main__":
    main()
