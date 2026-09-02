"""
mesh/bodyfit_tetgen.py — объёмная сетка, ОБЛЕГАЮЩАЮЩАЯ поверхность тела.

Зачем это нужно
---------------
Основной генератор (mesh/gmsh_generator.py) строит картезианский фон и
удаляет ячейки, чей центр попал внутрь тела. Поверхность тела в такую
сетку не попадает вовсе: границей расчётной области оказывается
«ступенька» из граней фоновых тетраэдров, и маркер airfoil в mesh.su2
собирается именно с этой ступеньки. Для тонких элементов (ГО, ВО, руль:
толщина 0.084 м при шаге фона 0.135 м — 0.62 шага) ступенька профиль не
описывает вообще, и SU2 на такой сетке расходится независимо от настроек
решателя.

Здесь сетка строится иначе. Поверхность тел и прямоугольная расчётная
область объединяются в один PLC и триангулируются TetGen как constrained
Delaunay. Ключевой параметр — ключ Y (nobisect): без него TetGen режет
входные грани ради соблюдения условия Делоне, и часть граней тела в
сетке отсутствует. Измерено на полном самолёте (43752 грани):

    без Y            38068 из 43752   87.0%
    с Y (pqY)        43750 из 43752   100.0%

Дальше тетраэдры, попавшие внутрь тела, удаляются, и границей сетки
оказываются настоящие грани тела. Проверено на том же самолёте:

    тетраэдров всего              317075
    удалено (центр внутри тела)   118447
    осталось                      198628
    объём наружной области        19676.0218  (короб 19683.0 - тело 6.9782)
    граней тела на границе сетки  43743 из 43752  = 99.98%

Топологическая герметизация (заливка «снаружи внутрь» с запретом
пересекать грани тела) здесь намеренно НЕ используется: TetGen при любом
коэффициенте качества теряет ровно 2 грани у стыка крыла с фюзеляжем, и
через эту дыру заливка выметает всю внутренность тела (измерено:
компонент 2, «внутренняя» объёмом 0.0305 вместо 6.978). Классификация по
центроидам от геометрии к этой дыре нечувствительна.

Если TetGen или движок булевых операций недоступны, функция возвращает
None, и вызывающий код остаётся на прежнем картезианском пути.
"""
from __future__ import annotations

import os
import numpy as np

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:          # pragma: no cover - pyvista есть в зависимостях
    pv = None
    HAS_PYVISTA = False

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    cKDTree = None
    HAS_SCIPY = False

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    trimesh = None
    HAS_TRIMESH = False

try:
    from tetgen import TetGen
    HAS_TETGEN = True
except Exception:
    TetGen = None
    HAS_TETGEN = False


# Доля граней тела, которая обязана оказаться в сетке. Ниже — сетка не
# облегает поверхность, и смысла в этом пути нет: возвращаем None.
DEFAULT_MIN_RECOVERY = 0.90


def tetgen_available():
    """True, если телообтекающий путь вообще может быть построен."""
    return bool(HAS_TETGEN and HAS_TRIMESH and HAS_SCIPY and HAS_PYVISTA)


def _load_solid_union():
    """Загрузить geometry/solid_union.py по пути файла.

    Обычный `from geometry.solid_union import ...` исполняет
    geometry/__init__.py, а тот тянет PyQt5 через stl_healer. В фоновом
    процессе генерации сетки и в тестах это лишняя зависимость.
    """
    try:
        import importlib.util
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "geometry", "solid_union.py")
        spec = importlib.util.spec_from_file_location(
            "solid_union_standalone", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        return None


_su = _load_solid_union()

if _su is not None:
    to_triangles = _su.to_triangles
else:
    def to_triangles(mesh):
        """Привести поверхность к (points[N,3], faces[M,3]) — запасной вариант."""
        if not HAS_PYVISTA:
            return None, None
        m = mesh if isinstance(mesh, pv.PolyData) else pv.PolyData(mesh)
        try:
            m = m.triangulate()
        except Exception:
            pass
        faces_raw = np.asarray(m.faces)
        if faces_raw.size == 0:
            return None, None
        if faces_raw.ndim == 1:
            stride = int(faces_raw[0]) + 1
            if stride < 4 or len(faces_raw) % stride != 0:
                return None, None
            faces = faces_raw.reshape(-1, stride)[:, 1:]
        else:
            faces = faces_raw
        if faces.shape[1] != 3:
            return None, None
        return np.asarray(m.points, dtype=float), np.ascontiguousarray(
            faces, dtype=np.int64)


def union_surfaces(body_meshes, log=print):
    """Слить поверхности тел в одну замкнутую (общая реализация).

    Пересечения разрешаются булевым объединением: части тел, оказавшиеся
    внутри других тел, вырезаются. Подробное обоснование и замеры — в
    geometry/solid_union.py. Возвращает (points, faces) или None.
    """
    if _su is not None:
        return _su.union_meshes(body_meshes, log=log)
    return None, None


def farfield_bounds(body_min, body_max, margin):
    """Габариты расчётной области — те же, что у картезианского пути."""
    body_min = np.asarray(body_min, dtype=float)
    body_max = np.asarray(body_max, dtype=float)
    return (float(body_min[0] - margin * 0.8),
            float(body_max[0] + margin * 1.2),
            float(body_min[1] - margin),
            float(body_max[1] + margin),
            float(body_min[2] - margin),
            float(body_max[2] + margin))


def box_surface(bounds, h_box):
    """Поверхность расчётной области, разбитая на треугольники.

    Ключ Y (nobisect) запрещает TetGen резать входные грани, поэтому
    короб из 12 треугольников остался бы в mesh.su2 маркером farfield из
    12 гигантских элементов. Разбиваем грани короба заранее: размер
    элементов тогда задаём мы, а не TetGen.

    Все шесть граней строятся на ОДНОЙ решётке узлов (ax, ay, az), и
    узел хранится в единственном экземпляре по своему индексу (i, j, k).
    Иначе на общих рёбрах появляются совпадающие дубли вершин, TetGen
    трактует их как самопересечение и отбрасывает грани (проверено:
    «256 input triangles are skipped due to self-intersections»).

    Возвращает (points, faces), нормали направлены наружу.
    """
    x0, x1, y0, y1, z0, z1 = [float(b) for b in bounds]
    nx = max(1, int(round((x1 - x0) / h_box)))
    ny = max(1, int(round((y1 - y0) / h_box)))
    nz = max(1, int(round((z1 - z0) / h_box)))
    ax = np.linspace(x0, x1, nx + 1)
    ay = np.linspace(y0, y1, ny + 1)
    az = np.linspace(z0, z1, nz + 1)
    center = np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0, (z0 + z1) / 2.0])

    pts = []
    index = {}

    def vid(i, j, k):
        key = (int(i), int(j), int(k))
        n = index.get(key)
        if n is None:
            n = len(pts)
            index[key] = n
            pts.append([ax[key[0]], ay[key[1]], az[key[2]]])
        return n

    faces = []
    # (фиксированная ось, её индекс, два свободных индекса и их пределы)
    faces_spec = [
        (0, 0, (ny, nz)), (0, nx, (ny, nz)),
        (1, 0, (nz, nx)), (1, ny, (nz, nx)),
        (2, 0, (nx, ny)), (2, nz, (nx, ny)),
    ]
    for fa, fidx, (nu, nv) in faces_spec:
        free = [a for a in (0, 1, 2) if a != fa]
        ua, va = free[0], free[1]
        for i in range(nu):
            for j in range(nv):
                corners = []
                for di, dj in ((0, 0), (1, 0), (1, 1), (0, 1)):
                    tri_idx = [0, 0, 0]
                    tri_idx[fa] = fidx
                    tri_idx[ua] = i + di
                    tri_idx[va] = j + dj
                    corners.append(vid(*tri_idx))
                a_, b_, c_, d_ = corners
                for t in ((a_, b_, c_), (a_, c_, d_)):
                    p = np.asarray(pts, dtype=float)[list(t)]
                    nrm = np.cross(p[1] - p[0], p[2] - p[0])
                    if np.dot(nrm, p.mean(axis=0) - center) < 0:
                        t = (t[0], t[2], t[1])
                    faces.append(list(t))
    return (np.asarray(pts, dtype=float),
            np.ascontiguousarray(faces, dtype=np.int64))


def tetrahedralize_plc(body_pts, body_faces, bounds,
                       min_ratio=2.0, max_volume=None, h_box=None, log=print):
    """Триангулировать PLC «тело + короб».

    Ключ Y (nobisect) обязателен — без него теряются грани тела.
    Возвращает (points, tets) или None.
    """
    if not HAS_TETGEN:
        return None, None
    extent = max(bounds[1] - bounds[0], bounds[3] - bounds[2],
                 bounds[5] - bounds[4])
    if not h_box or h_box <= 0:
        h_box = extent / 12.0
    box_pts, box_faces = box_surface(bounds, h_box)
    if box_pts is None or len(box_faces) == 0:
        return None, None

    n_body = len(body_pts)
    pts = np.vstack([body_pts, box_pts])
    faces = np.vstack([body_faces, box_faces + n_body])

    switches = "pY" if not min_ratio else "pq%gY" % float(min_ratio)
    if max_volume and max_volume > 0:
        switches += "a%.6g" % float(max_volume)

    try:
        tg = TetGen(pts, faces)
        tg.tetrahedralize(order=1, verbose=0, switches=switches)
    except Exception as e:
        log("   Внимание: TetGen не смог построить сетку (%s: %s)"
            % (type(e).__name__, e))
        return None, None

    tets = np.asarray(tg.grid.cells).reshape(-1, 5)[:, 1:]
    points = np.asarray(tg.grid.points, dtype=float)
    log("   Готово: TetGen (%s): %d тетраэдров, %d узлов"
        % (switches, len(tets), len(points)))
    return points, np.ascontiguousarray(tets, dtype=np.int64)


def tet_faces(tets):
    """Все 4 грани каждого тетраэдра, отсортированные по узлам.

    Возвращает (flat[K,3], owner[K]), где owner[k] — номер тетраэдра.
    Порядок в flat: тетраэдр 0 грани 0..3, тетраэдр 1 грани 0..3, ...
    """
    tets = np.asarray(tets, dtype=np.int64)
    faces = np.stack([tets[:, [0, 1, 2]],
                      tets[:, [1, 2, 3]],
                      tets[:, [0, 2, 3]],
                      tets[:, [0, 1, 3]]], axis=1)
    flat = np.sort(faces, axis=2).reshape(-1, 3)
    return flat, np.arange(len(flat)) // 4


def count_recovered(points, tets, body_pts, body_faces):
    """Сколько входных граней тела присутствует в объёмной сетке.

    TetGen переставляет нумерацию узлов, поэтому входные вершины
    сопоставляются выходным геометрически (cKDTree). Отклонение должно
    быть нулевым: входные точки constrained-триангуляции не двигаются.

    Возвращает (n_recovered, n_total, max_deviation, vertex_map).
    """
    flat, _ = tet_faces(tets)
    keys = {tuple(int(x) for x in row) for row in flat}
    dist, vmap = cKDTree(points).query(body_pts)
    n_rec = 0
    for f in body_faces:
        if tuple(sorted(int(vmap[x]) for x in f)) in keys:
            n_rec += 1
    return n_rec, int(len(body_faces)), float(dist.max()), vmap


def remove_interior_tets(points, tets, body_pts, body_faces, log=print):
    """Убрать тетраэдры, попавшие внутрь тела.

    Классификация по центроиду через VTK select_enclosed_points — тот же
    вызов, которым пользуется картезианский путь и который проверен на
    этой геометрии. Возвращает индексы наружных тетраэдров.
    """
    flat = np.hstack([np.full((len(body_faces), 1), 3, dtype=np.int64),
                      body_faces]).ravel()
    body = pv.PolyData(body_pts, flat)

    cells = np.hstack([np.full((len(tets), 1), 4, dtype=np.int64),
                       tets]).ravel()
    ctypes = np.full(len(tets), pv.CellType.TETRA, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, ctypes, points)
    centers = grid.cell_centers().points

    enclosed = pv.PolyData(centers).select_enclosed_points(
        body, tolerance=1e-5, check_surface=False)
    inside = np.asarray(enclosed["SelectedPoints"]).astype(bool)
    ext = np.where(~inside)[0]
    log("   Готово: удалено тетраэдров внутри тела: %d, осталось %d"
        % (int(inside.sum()), len(ext)))
    return ext


def build_body_fitted_grid(body_meshes, body_min, body_max, margin,
                           min_ratio=2.0, max_volume=None,
                           min_recovery=DEFAULT_MIN_RECOVERY, log=print):
    """Построить телообтекающую сетку.

    Возвращает dict:
        grid         — pyvista.UnstructuredGrid (только наружные тетраэдры)
        body_facets  — грани тела как тройки индексов узлов этой сетки
        recovery     — доля входных граней тела, попавших в сетку
        bounds       — габариты расчётной области
    или None, если путь недоступен либо сетка не облегает поверхность.
    """
    if not tetgen_available():
        log("   Внимание: TetGen/trimesh недоступны, строится "
            "картезианская сетка фона")
        return None

    body_pts, body_faces = union_surfaces(body_meshes, log=log)
    if body_pts is None:
        return None
    log("   Готово: объединённая поверхность: %d граней" % len(body_faces))

    bounds = farfield_bounds(body_min, body_max, margin)
    points, tets = tetrahedralize_plc(body_pts, body_faces, bounds,
                                      min_ratio=min_ratio,
                                      max_volume=max_volume, log=log)
    if points is None or len(tets) == 0:
        return None

    n_rec, n_tot, dev, _ = count_recovered(points, tets, body_pts, body_faces)
    recovery = n_rec / max(n_tot, 1)
    log("   Граней тела сохранено в сетке: %d из %d (%.2f%%), "
        "отклонение узлов %.1e" % (n_rec, n_tot, 100.0 * recovery, dev))
    if recovery < min_recovery:
        log("   Внимание: сохранено менее %.0f%% граней тела, сетка не "
            "облегает поверхность — строится картезианская сетка фона"
            % (100.0 * min_recovery))
        return None

    ext = remove_interior_tets(points, tets, body_pts, body_faces, log=log)
    if len(ext) < 100:
        log("   Внимание: после вырезания тела осталось %d тетраэдров"
            % len(ext))
        return None

    ext_tets = tets[ext]
    used = np.unique(ext_tets.ravel())
    remap = np.full(len(points), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    tets_new = remap[ext_tets]

    # Грани тела, ставшие границей наружной сетки: это и есть будущий
    # маркер airfoil. Собираются по точным тройкам индексов, без поиска
    # ближайшего соседа.
    flat, _ = tet_faces(tets_new)
    uniq, cnt = np.unique(flat, axis=0, return_counts=True)
    boundary = {tuple(int(x) for x in row)
                for row in uniq[cnt == 1]}
    body_facets = [tuple(sorted(int(remap[x]) for x in f))
                   for f in body_faces]
    marker = [f for f in body_facets if f in boundary]
    log("   Готово: граней тела на границе сетки: %d из %d (%.2f%%)"
        % (len(marker), len(body_faces),
           100.0 * len(marker) / max(len(body_faces), 1)))

    cells = np.hstack([np.full((len(tets_new), 1), 4, dtype=np.int64),
                       tets_new]).ravel()
    ctypes = np.full(len(tets_new), pv.CellType.TETRA, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, ctypes, points[used])

    return {"grid": grid,
            "body_facets": np.asarray(marker, dtype=np.int64)
            if marker else np.zeros((0, 3), dtype=np.int64),
            "recovery": recovery,
            "bounds": bounds,
            "n_tets": int(len(tets_new))}
