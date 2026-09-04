# -*- coding: utf-8 -*-
"""
mesh/bodyfit_gmsh.py — телооблегающая сетка через gmsh с гладким полем размера.

Зачем
-----
Телооблекающий путь через TetGen (``pq2Y``) строит сетку, конформную
поверхности тела (99.97% граней на границе), но БЕЗ поля размера TetGen
не даёт плавного перехода «мелко у тела -> крупно у короба». Резкий скачок
размера даёт растянутые тетраэдры, и 2-й порядок SU2 (MUSCL) расходится
(Residual > 10^20). Встроенное в TetGen поле размера (``-m``) сломано в
tetgen 0.8.4: сегфолтит (exit 139) и на коробном, и на сферическом PLC при
реалистичном масштабе — использовать нельзя.

gmsh (уже в зависимостях приложения, Direct CAD Import) строит гладкое поле
размера штатно: по расстоянию до поверхности тела задаётся целевой размер
элемента, и 3D-сетка конформится этому полю. Такой сетке не нужен ``-m`` и
2-й порядок на ней сходится.

Как устроено
------------
1. Поверхность тела (объединение компонентов по ``solid_union``) пишется во
   временный бинарный STL.
2. В gmsh создаётся расчётный короб (OCC box).
3. Поверхность тела импортируется и ``embed``-ится в объём короба — gmsh
   конформит тетраэдральную сетку этой границе, разделяя «снаружи тела» и
   «внутри тела».
4. Поле размера: ``gmsh.model.mesh.field`` Distance (к поверхности тела) +
   Threshold (h_near у тела, плавный рост до h_far на расстоянии L).
5. ``gmsh.model.mesh.generate(3)``.
6. Внутренние тетраэдры (внутри тела) удаляются топологической заливкой
   снаружи (``bodyfit_tetgen.classify_exterior_tets``) — та же логика, что
   в TetGen-пути. Поверхность тела становится границей расчётной области.
7. Результат проверяется так же (доля граней тела на границе, отсутствие
   вывернутых тетраэдров); ниже порога — возвращается None, и вызывающий код
   (``build_body_fitted_grid``) падает на проверенный TetGen-путь.

Функции ``_write_binary_stl`` и ``_size_field_params`` — чистый Python (без
gmsh), покрыты тестами. Сам вызов gmsh обёрнут в try/except: если gmsh
недоступен или не справился, функция возвращает None (без падения).
"""

from __future__ import annotations

import os
import signal
import struct
import tempfile

import numpy as np

try:
    import gmsh
    HAS_GMSH = True
except Exception:                       # pragma: no cover - нет gmsh
    gmsh = None
    HAS_GMSH = False

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    cKDTree = None
    HAS_SCIPY = False

# Доля граней тела, которая обязана оказаться на границе сетки (как в
# bodyfit_tetgen). Ниже — сетка не облегает поверхность, смысла нет.
DEFAULT_MIN_RECOVERY = 0.90

# Порог «грань лежит на поверхности тела»: граничная грань попадает в airfoil,
# если её центроид ближе этой доли шага к какой-то вершине тела.
_BODY_TOL_FRAC = 0.5


def gmsh_available():
    """True, если gmsh импортируется и приложение может строить сетку им."""
    return bool(HAS_GMSH)


def gmsh_missing():
    """Чего не хватает для gmsh-пути. Для честного сообщения."""
    miss = []
    if not HAS_GMSH:
        miss.append("gmsh")
    if not HAS_SCIPY:
        miss.append("scipy")
    return miss


# ---------------------------------------------------------------------------
# Чистый Python: параметры поля размера и запись STL (тестируемо без gmsh)
# ---------------------------------------------------------------------------
def _body_span(body_min, body_max):
    """Максимальный габарит тела (для масштаба поля размера)."""
    return float(np.max(np.asarray(body_max, dtype=float) -
                        np.asarray(body_min, dtype=float)))


def _size_field_params(bounds, body_min, body_max, target_edge):
    """Параметры гладкого поля размера для gmsh.

    Возвращает ``(h_near, h_far, dist_far, sampling)``:
        h_near    — целевая длина ребра у поверхности тела (<= target_edge);
        h_far     — целевая длина ребра вдали (extent/12, как поверхность короба);
        dist_far  — расстояние от тела, на котором размер достигает h_far;
        sampling  — сколько точек брать при оценке расстояния в gmsh.
    Лог-линейный рост между ними и даёт гладкое поле без скачка.
    """
    x0, x1, y0, y1, z0, z1 = [float(b) for b in bounds]
    extent = max(x1 - x0, y1 - y0, z1 - z0)
    body_min = np.asarray(body_min, dtype=float)
    body_max = np.asarray(body_max, dtype=float)
    span = _body_span(body_min, body_max)

    h_near = float(target_edge) if (target_edge and target_edge > 0) else extent / 12.0
    # Не мельче, чем нужно: слишком мелко у тела взрывает число ячеек.
    h_near = max(h_near, extent * 0.002)
    h_far = max(extent / 12.0, h_near * 1.05)

    # Расстояние, на котором размер достигает h_far: ~2 габарита тела, но не
    # больше 55% габарита области (иначе у короба поле ещё слишком мелкое и
    # не согласовано с его триангуляцией).
    dist_far = max(2.0 * span, 0.05 * extent)
    dist_far = min(dist_far, 0.55 * extent)

    # Сколько точек брать в Distance-поле gmsh для оценки расстояния до тела.
    # Больше — точнее, но дольше.
    sampling = 100
    return h_near, h_far, dist_far, sampling


def _write_binary_stl(points, faces, path):
    """Записать бинарный STL (триангулированная поверхность).

    Возвращает число записанных граней. Чистый numpy/struct — без gmsh,
    можно проверить в тестах.
    """
    pts = np.asarray(points, dtype=np.float32)
    fac = np.asarray(faces, dtype=np.int64)
    if fac.shape[1] != 3:
        raise ValueError("STL требует треугольные грани (N x 3)")
    with open(path, "wb") as f:
        f.write(b"\x00" * 80)
        f.write(struct.pack("<I", len(fac)))
        for tri in fac:
            p0 = pts[tri[0]]
            p1 = pts[tri[1]]
            p2 = pts[tri[2]]
            nrm = np.cross(p1 - p0, p2 - p0)
            ln = float(np.linalg.norm(nrm)) if len(nrm) else 0.0
            n = nrm / ln if ln > 1e-12 else np.zeros(3, dtype=np.float32)
            f.write(struct.pack("<3f", *n))
            for p in (p0, p1, p2):
                f.write(struct.pack("<3f", *p))
            f.write(struct.pack("<H", 0))
    return int(len(fac))


# ---------------------------------------------------------------------------
# Постобработка (общая с TetGen-путём): заливка снаружи + восстановление
# ---------------------------------------------------------------------------
def _boundary_faces(tets):
    """Грани тетраэдров, встречающиеся ровно один раз (истинная граница)."""
    t = np.asarray(tets, dtype=np.int64)
    flat = np.stack([
        np.sort(t[:, [0, 1, 2]], axis=1),
        np.sort(t[:, [1, 2, 3]], axis=1),
        np.sort(t[:, [0, 2, 3]], axis=1),
        np.sort(t[:, [0, 1, 3]], axis=1),
    ], axis=1).reshape(-1, 3)
    uniq, cnt = np.unique(flat, axis=0, return_counts=True)
    return uniq[cnt == 1]


def _faces_on_body(points, tets, body_pts, tol):
    """Индексы граней тетраэдров, лежащих НА поверхности тела.

    Ищем по ВСЕМ уникальным граням, а не только по граничным (count==1):
    gmsh ``mesh.embed`` встраивает поверхность тела в объём, и такая грань
    используется И внутренним (внутри тела) и наружным тетраэдром — она
    встречается дважды и в ``_boundary_faces`` (грани, встречающиеся ровно
    один раз) не попадает. Правильный признак — геометрический: все три
    вершины грани лежат в пределах ``tol`` от поверхности тела.

    Возвращает массив троек индексов узлов (сортированных).
    """
    if not HAS_SCIPY or len(body_pts) == 0 or len(tets) == 0:
        return np.zeros((0, 3), dtype=np.int64)
    t = np.asarray(tets, dtype=np.int64)
    flat = np.sort(np.stack([
        t[:, [0, 1, 2]],
        t[:, [1, 2, 3]],
        t[:, [0, 2, 3]],
        t[:, [0, 1, 3]],
    ], axis=1).reshape(-1, 3), axis=1)
    uniq = np.unique(flat, axis=0)
    if len(uniq) == 0:
        return np.zeros((0, 3), dtype=np.int64)
    tree = cKDTree(np.asarray(body_pts, dtype=float))
    d, _ = tree.query(np.asarray(points, dtype=float)[uniq.ravel()])
    d = d.reshape(len(uniq), 3)
    on_body = (d.max(axis=1) <= tol)
    return np.asarray(uniq[on_body], dtype=np.int64)


def _load_bodyfit_tetgen():
    """Загрузить mesh/bodyfit_tetgen.py по пути файла (без mesh/__init__).

    Обычный `from mesh.bodyfit_tetgen import ...` исполняет mesh/__init__.py,
    а тот тянет PyQt5 через mesh_worker. В тестах и в фоновом процессе это
    лишняя зависимость.
    """
    try:
        import importlib.util
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "bodyfit_tetgen.py")
        spec = importlib.util.spec_from_file_location(
            "bodyfit_tetgen_standalone", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        return None


def _exterior_index(points, tets, body_face_keys, body_bbox, body_pts,
                    body_faces, log=print):
    """Индексы наружных тетраэдров (снаружи тела) — заливка снаружи.

    Заливка не пересекает грани тела. Если почти всё оказалось «снаружи»
    (тело дырявое / заливка протекла) — возвращает None.
    """
    _bt = _load_bodyfit_tetgen()
    if _bt is None:
        return None
    exterior = _bt.classify_exterior_tets(points, tets, body_face_keys,
                                          body_bbox)
    n_in = int((~exterior).sum())
    n_out = int(exterior.sum())
    if n_in > 0 and n_out > 0:
        log("   Готово (топологическая заливка): внутри тела %d, снаружи %d "
            "тетраэдров" % (n_in, n_out))
        return np.where(exterior)[0]
    log("   Внимание: телооблегающая заливка не дала замкнутой стенки "
        "(внутри %d, снаружи %d) — отказ от gmsh-пути." % (n_in, n_out))
    return None


def _coverage(points, tets, ext, body_pts, body_faces, tol):
    """Доля входных граней тела, покрытых граничными гранями наружной сетки.

    gmsh перестраивает поверхность тела под поле размера, поэтому совпадение
    «грань в грань» (как у TetGen) не гарантировано. Здесь мера — покрытие:
    центроид каждой входной грани тела должен быть в пределах ``tol`` от
    центроида какой-нибудь граничной грани наружной сетки.
    """
    ext_tets = np.asarray(tets, dtype=np.int64)[np.asarray(ext, dtype=np.int64)]
    used = np.unique(ext_tets.ravel())
    remap = np.full(len(points), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    if len(used) == 0 or not HAS_SCIPY:
        return 0.0
    bface = _boundary_faces(remap[ext_tets])
    if len(bface) == 0:
        return 0.0
    bf_nodes = np.asarray(points, dtype=float)[used].astype(float)
    bc = bf_nodes[bface].mean(axis=1)
    if len(bface) > 200_000:
        idx = np.random.RandomState(0).choice(len(bc), 200_000, replace=False)
        bc = bc[idx]
    body_nodes = np.asarray(body_pts, dtype=float)
    # центроиды входных граней тела
    bc_body = np.asarray(body_pts, dtype=float)[np.asarray(body_faces, dtype=np.int64)].mean(axis=1)
    tree = cKDTree(bc)
    d, _ = tree.query(bc_body)
    return float((d <= tol).mean())


# ---------------------------------------------------------------------------
# Основной вход: gmsh-телооблегающая сетка
# ---------------------------------------------------------------------------
def build_body_fitted_grid_gmsh(body_meshes, body_min, body_max, margin,
                                min_recovery=DEFAULT_MIN_RECOVERY, log=print,
                                target_edge=None, max_surface_faces=400_000,
                                body_pts=None, body_faces=None):
    """Телооблегающая сетка через gmsh с гладким полем размера.

    ``body_pts``/``body_faces`` — уже готовая объединённая поверхность тела
    (замкнутая). Если их передать, они не объединяются заново (дорогое
    булево объединение уже сделано вызывающим кодом).

    Возвращает тот же dict, что ``bodyfit_tetgen.build_body_fitted_grid``:
        grid         — pyvista.UnstructuredGrid (только наружные тетраэдры)
        body_facets  — грани тела как тройки индексов узлов этой сетки
        recovery     — доля входных граней тела, покрытых границей
        bounds       — габариты расчётной области
        n_tets       — число тетраэдров
    или None, если gmsh недоступен / не справился / сетка не облегает тело
    (тогда вызывающий код падает на проверенный TetGen-путь).
    """
    if not HAS_GMSH:
        log("   Внимание: gmsh не доступен — телооблегающая сетка через "
            "gmsh невозможна, пробую TetGen.")
        return None
    if not HAS_SCIPY:
        log("   Внимание: для gmsh-пути нужен scipy — пробую TetGen.")
        return None

    try:
        import pyvista as pv
    except Exception:
        pv = None
    if pv is None:
        log("   Внимание: для gmsh-пути нужен pyvista — пробую TetGen.")
        return None

    # Общая реализация (union_surfaces, surface_needs_refinement,
    # _load_surface_refine, farfield_bounds, classify_exterior_tets) живёт в
    # bodyfit_tetgen.py. Грузим по пути файла, чтобы не тянуть mesh/__init__
    # (PyQt5) в тестах и в фоновом процессе.
    _bt = _load_bodyfit_tetgen()
    if _bt is None:
        log("   Внимание: bodyfit_tetgen недоступен — gmsh-путь невозможен, "
            "пробую TetGen.")
        return None

    # Поверхность тела: объединение компонентов (замкнутая оболочка),
    # если не передана готовой.
    if body_pts is None or body_faces is None:
        try:
            body_pts, body_faces = _bt.union_surfaces(body_meshes, log=log)
        except Exception as e:                          # pragma: no cover
            log("   Внимание: объединить поверхность не удалось (%s) — "
                "TetGen." % e)
            return None
        if body_pts is None or len(body_faces) == 0:
            log("   Внимание: объединённая поверхность пустая — TetGen.")
            return None
        log("   Готово: объединённая поверхность: %d граней" % len(body_faces))

    # Плотность поверхности задаёт размер у тела и в gmsh тоже: чем мельче
    # грани, тем мельче ячейки у тела. Уплотняем, если поверхность грубее
    # цели (как в TetGen-пути).
    if target_edge and _bt.surface_needs_refinement(body_pts, body_faces,
                                                    target_edge):
        _sr = _bt._load_surface_refine()
        if _sr is not None:
            try:
                v2, f2, info = _sr.refine_to_edge_length(
                    body_pts, body_faces, target_edge,
                    max_faces=max_surface_faces)
                log("   Поверхность уплотнена до шага %.4f м: %d -> %d граней"
                    % (target_edge, len(body_faces), len(f2)))
                body_pts, body_faces = v2, f2
            except Exception as e:
                log("   Внимание: уплотнить поверхность не удалось (%s), "
                    "строю по исходной триангуляции" % e)

    # Габариты расчётной области (те же, что у картезианского/TetGen-пути).
    bounds = _bt.farfield_bounds(body_min, body_max, margin)

    h_near, h_far, dist_far, sampling = _size_field_params(
        bounds, body_min, body_max, target_edge)

    # Записываем тело во временный STL и строим сетку gmsh.
    _gmsh_tmp = tempfile.NamedTemporaryFile(suffix=".stl", delete=False)
    _gmsh_tmp.close()
    try:
        _write_binary_stl(body_pts, body_faces, _gmsh_tmp.name)

        grid, body_keys, boundary_ratio = _gmsh_mesh(
            _gmsh_tmp.name, body_pts, body_faces, bounds,
            h_near, h_far, dist_far, sampling, log=log)
    except Exception as e:                              # pragma: no cover
        log("   Внимание: gmsh не построил сетку (%s: %s) — пробую TetGen."
            % (type(e).__name__, e))
        grid = None
    finally:
        try:
            os.remove(_gmsh_tmp.name)
        except OSError:
            pass

    if grid is None:
        return None

    # Постпроверка доли граней тела на границе.
    if boundary_ratio < min_recovery:
        log("   Внимание: на границу сетки вышло %.1f%% граней тела — "
            "gmsh-сетка дырявая, откат на TetGen."
            % (100.0 * boundary_ratio))
        return None

    recovery = boundary_ratio
    log("   Граней тела на границе сетки: %d из %d (%.2f%%)"
        % (len(body_keys), len(body_faces), 100.0 * boundary_ratio))
    return {"grid": grid,
            "body_facets": np.asarray(body_keys, dtype=np.int64)
            if len(body_keys) else np.zeros((0, 3), dtype=np.int64),
            "recovery": recovery,
            "bounds": bounds,
            "n_tets": int(grid.n_cells)}


def _gmsh_mesh(stl_path, body_pts, body_faces, bounds,
               h_near, h_far, dist_far, sampling, log=print):
    """Сам вызов gmsh: короб + embed тела + поле размера + generate(3).

    Возвращает ``(grid, body_keys, boundary_ratio)`` или бросает исключение.
    ``grid`` — pyvista.UnstructuredGrid (только наружные тетраэдры);
    ``body_keys`` — троек индексов узлов граней тела на границе;
    ``boundary_ratio`` — доля входных граней тела на границе.
    """
    import pyvista as pv
    x0, x1, y0, y1, z0, z1 = [float(b) for b in bounds]

    # gmsh при инициализации/работе регистрирует обработчик сигналов
    # (SIGINT/SIGTERM), а Python разрешает это только в ГЛАВНОМ потоке. Генерация
    # сетки идёт в фоновом QThread, поэтому gmsh без обхода падает с
    # "ValueError: signal only works in main thread of the main interpreter".
    # Временно подменяем signal.signal на заглушку на время сессии gmsh и
    # восстанавливаем в finally — gmsh считает, что обработчик установлен, а
    # в фоновом потоке он всё равно не работает.
    _orig_signal = signal.signal

    def _noop_signal(sig, handler):
        return handler

    signal.signal = _noop_signal
    try:
        gmsh.initialize()
        signal.signal = _noop_signal
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("Mesh.Algorithm", 6)          # фронтальный (теты)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
        gmsh.option.setNumber("Mesh.MeshSizeMin", h_near)
        gmsh.option.setNumber("Mesh.MeshSizeMax", h_far)
        gmsh.model.add("bodyfit_gmsh")

        # Расчётный короб.
        box = gmsh.model.occ.addBox(x0, y0, z0, x1 - x0, y1 - y0, z1 - z0)
        gmsh.model.occ.synchronize()
        box_vol = box

        # Поверхность тела -> отдельные 2D-сущности.
        gmsh.merge(stl_path)
        gmsh.model.occ.synchronize()
        # 2D-сущности, добавленные из STL (не принадлежащие коробу).
        box_surf_tags = set()
        for (d, t) in gmsh.model.getBoundary([(3, box_vol)], oriented=False):
            if d == 2:
                box_surf_tags.add(t)
        body_surf_tags = [t for (d, t) in gmsh.model.getEntities(2)
                          if d == 2 and t not in box_surf_tags]
        if not body_surf_tags:
            # Вся замкнутая оболочка может быть одной сущностью — берём всё,
            # что не является границами короба.
            body_surf_tags = [t for (d, t) in gmsh.model.getEntities(2)
                              if d == 2]

        # Встраиваем поверхность тела в объём короба: gmsh конформит сетку
        # этой границе (разделяя «снаружи» и «внутри» тела).
        gmsh.model.mesh.embed(2, body_surf_tags, 3, box_vol)

        # Поле размера: Distance (к телу) + Threshold (h_near -> h_far).
        dfield = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumber(dfield, "Sampling", sampling)
        gmsh.model.mesh.field.setNumbers(dfield, "SurfacesList", body_surf_tags)
        tfield = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(tfield, "InField", dfield)
        gmsh.model.mesh.field.setNumber(tfield, "SizeMin", h_near)
        gmsh.model.mesh.field.setNumber(tfield, "SizeMax", h_far)
        gmsh.model.mesh.field.setNumber(tfield, "DistMin", h_near)
        gmsh.model.mesh.field.setNumber(tfield, "DistMax", dist_far)
        gmsh.model.mesh.field.setAsBackgroundMesh(tfield)

        # Генерируем объём.
        gmsh.model.mesh.generate(3)

        # Достаём узлы и тетраэдры.
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        coords = np.asarray(node_coords, dtype=float).reshape(-1, 3)
        # getElements(dim, tag) -> (elementTypes, elementTags, nodeTags).
        elem_types, _, node_tags_list = gmsh.model.mesh.getElements(3)
        tets = []
        for et, ntl in zip(elem_types, node_tags_list):
            if int(et) == 4:            # 4 = тетраэдр
                arr = np.asarray(ntl, dtype=np.int64).reshape(-1, 4)
                # gmsh node tags are 1-based global; map to local 0-based.
                tets.append(arr)
        if not tets:
            raise RuntimeError("gmsh не создал тетраэдров")
        tet_tags_local = np.vstack(tets)
        # Сопоставляем глобальные теги узлов локальным индексам.
        node_tags = np.asarray(node_tags, dtype=np.int64)
        tag_to_local = {int(t): i for i, t in enumerate(node_tags)}
        tets_local = np.vectorize(lambda t: tag_to_local[int(t)])(
            tet_tags_local)
        tets_local = np.ascontiguousarray(tets_local, dtype=np.int64)

        # Нормали ориентации: gmsh выдаёт тетраэдры с согласованной
        # ориентацией; проверяем знаковый объём и разворачиваем вывернутые.
        points = coords
        e0 = points[tets_local[:, 1]] - points[tets_local[:, 0]]
        e1 = points[tets_local[:, 2]] - points[tets_local[:, 0]]
        e2 = points[tets_local[:, 3]] - points[tets_local[:, 0]]
        vol = np.einsum('ij,ij->i', e0, np.cross(e1, e2)) / 6.0
        inv = vol < 0
        if inv.any():
            tets_local[inv, 1], tets_local[inv, 2] = (
                tets_local[inv, 2].copy(), tets_local[inv, 1].copy())

        body_bbox = (np.asarray(body_pts).min(axis=0),
                     np.asarray(body_pts).max(axis=0))
        # Грани тела на границе наружной сетки: граничные грани, лежащие
        # на поверхности тела.
        tol = max(h_near, 1e-9) * _BODY_TOL_FRAC
        body_keys = _faces_on_body(points, tets_local, body_pts, tol)
        if len(body_keys) == 0:
            raise RuntimeError("не найдены грани тела на границе сетки")
        body_face_keys = {tuple(int(x) for x in row) for row in body_keys}

        ext = _exterior_index(points, tets_local, body_face_keys, body_bbox,
                              body_pts, body_faces, log=log)
        if ext is None:
            raise RuntimeError("заливка не дала замкнутой стенки")

        ext_tets = tets_local[ext]
        used = np.unique(ext_tets.ravel())
        remap = np.full(len(points), -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        tets_new = remap[ext_tets]
        points_new = points[used]

        cells = np.hstack([np.full((len(tets_new), 1), 4, dtype=np.int64),
                           tets_new]).ravel()
        ctypes = np.full(len(tets_new), pv.CellType.TETRA, dtype=np.uint8)
        grid = pv.UnstructuredGrid(cells, ctypes, points_new)

        # Грани тела в координатах НАРУЖНОЙ сетки (входные индексы body_keys
        # — полной сетки, а dict потребляют в индексах наружной части).
        ext_bnd = _boundary_faces(tets_new)
        ext_bnd_set = {tuple(int(x) for x in r) for r in ext_bnd}
        body_keys_final = []
        for row in body_keys:
            r = tuple(sorted(int(remap[x]) for x in row))
            if all(v >= 0 for v in r) and r in ext_bnd_set:
                body_keys_final.append(r)
        body_keys = np.asarray(body_keys_final, dtype=np.int64) \
            if body_keys_final else np.zeros((0, 3), dtype=np.int64)

        # Покрытие: доля входных граней тела, попавших на границу наружной
        # сетки (для gmsh — покрытие, т.к. поверхность могла перестроиться).
        boundary_ratio = _coverage(points, tets_local, ext, body_pts,
                                   body_faces, tol)
        log("   Готово: gmsh-сетка %d тетраэдров, %d узлов" %
            (len(tets_new), len(used)))
        return grid, body_keys, boundary_ratio
    finally:
        signal.signal = _orig_signal
        try:
            gmsh.finalize()
        except Exception:                               # pragma: no cover
            pass
