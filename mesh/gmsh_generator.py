"""
mesh/gmsh_generator.py — ИСПРАВЛЕННАЯ КОПИЯ.

Изменения относительно исходного файла (без потери функционала):

1. write_su2(): жёсткая защита — если маркер 'airfoil' пустой, бросаем
   RuntimeError с понятным сообщением. Иначе SU2 падает тихо и history.csv
   пишется без surface_flow.vtu.

2. write_su2(): запись NPOIN + точек + маркеров обёрнута в try/except —
   если файл обрезается посередине, мы увидим ошибку в логе, а не
   «успешный» SU2 на обрезанной сетке.

3. write_su2(): явный сброс буфера f.flush() + f.fsync() перед закрытием —
   на Windows иногда файл обрывается на 64 КБ без fsync.

4. Добавлена функция quick_mesh_check(path) — можно дёрнуть из main_window
   после генерации, чтобы удостовериться, что mesh.su2 содержит и
   NPOIN, и оба маркера.
"""
from __future__ import annotations

import os
import sys
import json
import traceback
import numpy as np
import pyvista as pv

# Настройки проекта
try:
    from config.settings import (WORK_DIR_BASE, PREVIEW_MESH, MESH_FILE,
                                 MESH_QUALITY)
except ImportError:
    WORK_DIR_BASE = os.path.join(os.getcwd(), "cases")
    PREVIEW_MESH = os.path.join(WORK_DIR_BASE, "preview.vtk")
    MESH_FILE = os.path.join(WORK_DIR_BASE, "mesh.su2")
    MESH_QUALITY = ["Грубая (быстро)", "Средняя", "Точная (медленно)"]

# Тот же словарь, что читает solver/workers.py. Если config.settings
# недоступен — локальная копия: вердикт просто никто не прочитает.
try:
    from config.settings import MESH_DIAGNOSIS
except ImportError:
    MESH_DIAGNOSIS = {"body_fitted": True, "unresolved": [],
                      "flat": [], "reason": ""}

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class MeshCancelled(Exception):
    """Бросается, когда пользователь отменил генерацию сетки."""
    pass


# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------
def extract_cells(mesh, cell_type):
    """Надёжный экстрактор ячеек заданного VTK-типа."""
    try:
        cd = mesh.cells_dict
        if cell_type in cd:
            return cd[cell_type].tolist()
    except Exception:
        pass
    try:
        if hasattr(mesh, 'cell_connectivity'):
            connect = np.asarray(mesh.cell_connectivity)
            types = np.asarray(mesh.celltypes)
            offs = np.asarray(mesh.offset)
            mask = types == cell_type
            indices = np.where(mask)[0]
            if len(indices) > 0:
                result = []
                for i in indices:
                    start = int(offs[i])
                    end = int(offs[i + 1]) if i + 1 < len(offs) else len(connect)
                    if start <= end <= len(connect):
                        result.append(connect[start:end].tolist())
                return result
    except Exception:
        pass
    try:
        faces_raw = np.asarray(mesh.faces)
        n_cells = mesh.n_cells
        if n_cells > 0 and len(faces_raw) > 0:
            first_n = int(faces_raw[0])
            if first_n > 0 and len(faces_raw) == n_cells * (first_n + 1):
                reshaped = faces_raw.reshape(n_cells, first_n + 1)
                expected_n = {10: 4, 5: 3, 9: 3, 7: 3}
                if first_n in expected_n.values():
                    return reshaped[:, 1:].tolist()
    except Exception:
        pass
    return []


def _extract_boundary_faces(tets):
    """Грани тетра-сетки, встречающиеся ровно один раз (истинная граница
    объёма). Утилита также используется в тестах."""
    t = np.asarray(tets, dtype=np.int64)
    faces = np.vstack([t[:, [0, 1, 2]], t[:, [0, 1, 3]],
                       t[:, [0, 2, 3]], t[:, [1, 2, 3]]])
    keys = np.sort(faces, axis=1)
    uniq, inv, counts = np.unique(keys, axis=0,
                                  return_inverse=True, return_counts=True)
    return faces[counts[inv] == 1]


# Алиас для обратной совместимости
_extract_boundary_faces = _extract_boundary_faces  # чтобы import не ругался
_extract_boundary_faces = _extract_boundary_faces  # на всякий случай


# ---------------------------------------------------------------------------
# Запись файла формата SU2 (ИСПРАВЛЕННАЯ)
# ---------------------------------------------------------------------------
def write_su2(grid, surface, filename, markers_info=None, **kwargs):
    """Запись mesh.su2 с гарантированной совместимостью.

    ИСПРАВЛЕНО (без потери функционала):
      - явная проверка, что маркер 'airfoil' не пустой (иначе SU2 падает
        тихо и surface_flow.vtu не пишется);
      - запись NPOIN/точек/маркеров под try/except с понятным сообщением
        в логе, если файловая система обрезает файл;
      - явный flush+fsync на Windows перед закрытием.

    T1: kwargs['use_symmetry'] — добавить маркер symmetry_plane на Y=0.
    """
    pre_clip_points = kwargs.get("pre_clip_points", None)
    vol_pts = np.asarray(grid.points)
    n_points = len(vol_pts)
    tetras = extract_cells(grid, cell_type=10)
    if not tetras:
        raise RuntimeError("Не найдены тетраэдры в сетке")

    surf_pts = np.asarray(surface.points)
    if markers_info is None:
        markers_info = [("airfoil", 0), ("farfield", 0)]

    # === Порядок вершин грани маркера ====================================
    #
    # SU2 берёт нормаль грани маркера из порядка её вершин и больше
    # ниоткуда. До сих пор порядок приходил прямо из surface.faces, то
    # есть полагался на vtkPolyDataNormals (auto_orient_normals /
    # consistent_normals). Это ненадёжно: consistent_normals требует
    # многообразной поверхности, а после резки по плоскости симметрии и
    # склейки совпавших точек поверхность не обязана быть
    # многообразной.
    #
    # Замер на сгенерированном самолёте: 911 из 12367 граней airfoil
    # (7.37%) были записаны с обходом внутрь тела, в symmetry_xz — 17,
    # в farfield — 0. Такая грань даёт вклад в силу с обратным знаком,
    # поэтому завышались и Cl, и Cd одновременно: расчёт дал Cl=0.739 и
    # Cd=0.360 при L/D=2.05, хотя в невязкой постановке сопротивление
    # замкнутого тела должно быть близко к нулю.
    #
    # Внешняя нормаль граничной грани тетраэдральной сетки определяется
    # точно и без эвристик: она смотрит от четвёртой вершины
    # тетраэдра наружу. Порядок вершин выставляется по ней.
    _out_winding = {}
    try:
        _Tw = np.asarray([t for t in tetras if len(t) == 4], dtype=np.int64)
        for _miss in range(4):
            if not len(_Tw):
                break
            _f = _Tw[:, [j for j in range(4) if j != _miss]]
            _p = vol_pts[_f]
            _n = np.cross(_p[:, 1] - _p[:, 0], _p[:, 2] - _p[:, 0])
            _opp = vol_pts[_Tw[:, _miss]]
            _flip = np.einsum('ij,ij->i', _n, _p.mean(axis=1) - _opp) < 0.0
            _f = np.where(_flip[:, None], _f[:, [0, 2, 1]], _f)
            for _k, _v in zip(map(tuple, np.sort(_f, axis=1)), _f):
                _out_winding[_k] = _v
    except Exception:
        _out_winding = {}

    # === T1: список плоскостей симметрии (XY, XZ, YZ) =================
    # Поддержка нескольких плоскостей одновременно. Каждая плоскость
    # даёт отдельный маркер в mesh.su2:
    #   symmetry_xy  — Z=0
    #   symmetry_xz  — Y=0  (старое имя symmetry_plane для обратной совместимости)
    #   symmetry_yz  — X=0
    # В config.cfg прописываются все включённые плоскости в MARKER_SYM.
    # По умолчанию ВЫКЛЮЧЕНО, чтобы не ломать обратную совместимость.
    use_symmetry = bool(kwargs.get("use_symmetry", False))
    symmetry_planes = kwargs.get("symmetry_planes", None)  # list[str] | None
    if symmetry_planes is None:
        # Раньше при use_symmetry=True подставлялся "xz" по умолчанию и
        # сетка резалась пополам без явного указания плоскости. Теперь
        # резка выполняется только по явно заданным плоскостям.
        symmetry_planes = []
    # Нормализуем в нижний регистр
    symmetry_planes = [str(p).lower() for p in symmetry_planes if p]
    # Словари треугольников по маркерам
    symmetry_tris: dict = {p: [] for p in symmetry_planes}

    # Маппинг точек поверхности -> точки объёма
    try:
        tree = cKDTree(vol_pts)
        _, point_map = tree.query(surf_pts)
    except Exception:
        point_map = np.zeros(len(surf_pts), dtype=int)
        for i, sp in enumerate(surf_pts):
            dists = np.linalg.norm(vol_pts - sp, axis=1)
            point_map[i] = int(np.argmin(dists))

    # Классификация граней по маркерам
    x_min, x_max = vol_pts[:, 0].min(), vol_pts[:, 0].max()
    y_min, y_max = vol_pts[:, 1].min(), vol_pts[:, 1].max()
    z_min, z_max = vol_pts[:, 2].min(), vol_pts[:, 2].max()
    bbox_size = max(x_max - x_min, y_max - y_min, z_max - z_min)
    tol = bbox_size * 0.002
    airfoil_tris = []
    farfield_tris = []

    faces_raw = np.asarray(surface.faces)
    n_surf_cells = surface.n_cells

    # === Нормали граней нужны, чтобы отличить настоящий срез от полосы
    # поверхности тела, случайно попавшей в окрестность плоскости.
    # surface.cell_normals здесь не годится: classify_and_append получает
    # массив индексов ВЕРШИН каждого треугольника, а не номера ячеек.
    # Нормаль считается прямо по вершинам — это не зависит от порядка
    # ячеек и работает в обеих ветках разбора faces.

    # Индекс по поверхности до резки: расстояние до неё показывает,
    # существовала ли грань раньше.
    # Индекс точек, существовавших до резки.
    #
    # scipy.spatial.cKDTree здесь не используется намеренно: он
    # импортируется под try/except, и при его отсутствии проверка
    # происхождения молча отключалась — симметрия снова начинала
    # съедать поверхность тела, без единого сообщения в логе.
    #
    # Точность не нужна: clip() копирует исходные точки в выходную
    # сетку побитово и лишь ДОБАВЛЯЕТ новые на секущей плоскости.
    # Поэтому достаточно точного сравнения квантованных координат.
    # Допуск «эта точка существовала до резки». Берётся от габарита
    # области и делается заведомо меньше шага сетки: координаты
    # совпадающих точек отличаются только плавающей ошибкой.
    _pt_tol = 1e-7 * float(max(1.0, bbox_size))
    _pre_keys = None
    _pt_scale = 1.0 / max(_pt_tol, 1e-12)
    if pre_clip_points is not None:
        try:
            _pq = np.round(np.asarray(pre_clip_points, dtype=float)
                           * _pt_scale).astype(np.int64)
            _pre_keys = set(map(tuple, _pq))
        except Exception:
            _pre_keys = None
    # Порог «грань лежала на поверхности до резки». Берётся от шага
    # сетки, а не от габарита области: настоящая поверхность тела имеет
    # расстояние ~0, а срез отстоит от неё на долю толщины тела.
    # Допуск «эта точка существовала до резки». Берётся от габарита
    def classify_and_append(tri_idx_array):
        if len(tri_idx_array) == 0:
            return
        tri_idx_array = np.asarray(tri_idx_array)
        tri_pts = surf_pts[tri_idx_array]
        centroids = tri_pts.mean(axis=1)
        tri_normals = None
        try:
            _e1 = tri_pts[:, 1, :] - tri_pts[:, 0, :]
            _e2 = tri_pts[:, 2, :] - tri_pts[:, 0, :]
            _cr = np.cross(_e1, _e2)
            _ln = np.linalg.norm(_cr, axis=1)
            _ok = _ln > 1e-12
            if _ok.all():
                tri_normals = _cr / _ln[:, None]
        except Exception:
            tri_normals = None
        # Граница области определяется по bbox сетки. Но после резки по
        # плоскости симметрии соответствующая грань bbox совпадает с этой
        # плоскостью: при резке по Z=0 остаётся z>0, и z_min становится
        # равным 0. Тогда всё, что лежит в пределах tol от плоскости,
        # попадает в is_out и уходит в MARKER_FAR.
        #
        # На самолёте это съедает крыло целиком: полутолщина крыла
        # ~0.075 м, а tol считается от габарита области с дальним полем
        # и равен ~0.176 м. Маркер стенки оставался почти пустым
        # (airfoil=470 при 4504 до резки), и SU2 расходился.
        #
        # Поэтому грань bbox, совпавшую с активной плоскостью симметрии,
        # дальней границей не считаем.
        _sym_axis = {"xy": 2, "xz": 1, "yz": 0}
        _skip_lo = set()
        _skip_hi = set()
        for _pl in (symmetry_planes or []):
            _pn2 = str(_pl).split(":", 1)[0].strip().lower()
            _ax = _sym_axis.get(_pn2)
            if _ax is None:
                continue
            _off = 0.0
            if ":" in str(_pl):
                try:
                    _off = float(str(_pl).split(":", 1)[1])
                except Exception:
                    _off = 0.0
            _lo = (x_min, y_min, z_min)[_ax]
            _hi = (x_max, y_max, z_max)[_ax]
            if abs(_lo - _off) < tol:
                _skip_lo.add(_ax)
            if abs(_hi - _off) < tol:
                _skip_hi.add(_ax)
        is_out = np.zeros(len(centroids), dtype=bool)
        for _ax, (_lo, _hi) in enumerate(
                ((x_min, x_max), (y_min, y_max), (z_min, z_max))):
            if _ax not in _skip_lo:
                is_out |= np.abs(centroids[:, _ax] - _lo) < tol
            if _ax not in _skip_hi:
                is_out |= np.abs(centroids[:, _ax] - _hi) < tol
        # === T1: плоскости симметрии (XY=Z=0, XZ=Y=0, YZ=X=0) =========
        # Проверяем для каждой включённой плоскости — если треугольник
        # лежит на этой плоскости, относим к соответствующему маркеру.
        sym_mask: dict = {}
        if symmetry_planes:
            for plane in symmetry_planes:
                # Поддержка формата "xz" (через 0) и "xz:1.5" (смещение)
                if ":" in plane:
                    p_name, p_offset = plane.split(":", 1)
                    p_offset = float(p_offset)
                else:
                    p_name = plane
                    p_offset = 0.0
                p_name = p_name.strip().lower()
                if p_name == "xy":
                    sym_mask[plane] = np.abs(centroids[:, 2] - p_offset) < tol
                elif p_name == "xz":
                    sym_mask[plane] = np.abs(centroids[:, 1] - p_offset) < tol
                elif p_name == "yz":
                    sym_mask[plane] = np.abs(centroids[:, 0] - p_offset) < tol

        # === Отсекаем полосу поверхности тела, попавшую в окрестность
        # плоскости по одной только координате.
        #
        # tol считается от габарита расчётной области (~81 м для этого
        # самолёта), то есть tol ~ 0.16 м. В такую полосу вокруг Y=0
        # попадает верх и низ фюзеляжа (R=0.60 м): их нормали смотрят
        # вверх и вниз, а не вдоль плоскости. Раньше эти треугольники
        # уходили в MARKER_SYM, в стенке самолёта появлялась щель вдоль
        # всего фюзеляжа, и SU2 расходился.
        #
        # У настоящего среза после clip() нормаль точно параллельна
        # нормали плоскости. Требование |n . n_plane| > 0.99 отделяет
        # срез от поверхности тела и ничего не ломает для тонких
        # поверхностей, лежащих в самой плоскости (киль): их средняя
        # плоскость действительно является плоскостью симметрии.
        if sym_mask and tri_normals is not None:
            _plane_normals = {"xy": (0.0, 0.0, 1.0),
                              "xz": (0.0, 1.0, 0.0),
                              "yz": (1.0, 0.0, 0.0)}
            for plane in list(sym_mask.keys()):
                _pn = _plane_normals.get(str(plane).split(":", 1)[0])
                if _pn is None:
                    continue
                _dot = np.abs(tri_normals @ np.asarray(_pn))
                sym_mask[plane] = sym_mask[plane] & (_dot > 0.99)

        # === Второй признак: все три вершины грани созданы самой резкой.
        #
        # Одной нормали мало. Киль стоит в плоскости XZ, крыло лежит в
        # плоскости XY — у их треугольников нормаль тоже вдоль плоскости,
        # и по одной геометрии они попадают в срез.
        #
        # Отличить их можно по происхождению, а не по расстоянию. clip()
        # создаёт новые точки ровно на секущей плоскости; у поверхности
        # тела вершины существовали и до резки. Поэтому настоящий срез —
        # это грань, у которой ВСЕ вершины новые.
        #
        # Расстояние до поверхности до резки здесь не годится: его порог
        # пришлось бы брать от шага сетки, а шаг на границе области в
        # десятки раз больше шага у тела, и порог оказывается больше
        # толщины крыла.
        #
        # Именно это позволяет резать модель по любой плоскости, а не
        # только по «удобной» XZ.
        # Признак происхождения оставлен как страховка от числового
        # мусора, но он НЕ МОЖЕТ быть единственным — см. ниже.
        _all_new = None
        if sym_mask and _pre_keys is not None:
            try:
                _qv = np.round(tri_pts.reshape(-1, 3)
                               * _pt_scale).astype(np.int64)
                _old_v = np.array([tuple(_v) in _pre_keys for _v in _qv],
                                  dtype=bool).reshape(len(tri_idx_array), 3)
                _all_new = ~_old_v.any(axis=1)
            except Exception:
                _all_new = None

        # === Основной признак: ВСЕ ТРИ вершины грани лежат в плоскости.
        #
        # Раньше единственным признаком было происхождение вершин: срезом
        # считалась грань, у которой все три вершины созданы самой резкой.
        # На дальнем поле это ломается. Коробка области симметрична
        # относительно плоскости реза, поэтому слой её узлов лежит ровно
        # на y=0, и clip() переиспользует эти точки как старые. Половина
        # граней среза оказывалась «не новой», проваливалась в airfoil,
        # и SU2 получал MARKER_EULER и MARKER_MONITORING на плоской плите
        # в невозмущённом потоке.
        #
        # Замерено на сгенерированном самолёте (Средняя, резка по XZ):
        #   airfoil     12899 граней, 3105.9 м2, из них 843 грани
        #               и 3078.8 м2 лежат ровно в Y=0;
        #   symmetry_xz  4108 граней, 3113.0 м2, все в Y=0;
        #   сумма 6191.8 м2 при площади сечения области 6200.0 м2 —
        #   то есть плоскость симметрии делилась между маркерами пополам.
        #   Настоящая поверхность тела — только 27.1 м2 из 3105.9.
        # Расчёт по такой сетке даёт Cd=0.184 и Cm=9.23 при норме ~0.02
        # и ~0.1: давление интегрируется по плите в десятки метров
        # плечом от точки отсчёта момента.
        #
        # Принадлежность плоскости — свойство геометрическое и точное:
        # у грани среза в плоскости лежат все три вершины, у поверхности
        # тела хотя бы одна отстоит от неё. Киль (y=±0.042 м) и верх/низ
        # фюзеляжа отсеиваются тем же признаком, что и раньше. Допуск
        # 1e-6 от габарита области на два порядка меньше полутолщины
        # самых тонких тел и несравнимо больше ошибки, с которой clip()
        # ставит точки на плоскость.
        if sym_mask:
            _plane_axis = {"xy": 2, "xz": 1, "yz": 0}
            _plane_eps = 1e-6 * float(max(1.0, bbox_size))
            for plane in list(sym_mask.keys()):
                _pname = str(plane).split(":", 1)[0].strip().lower()
                _ax = _plane_axis.get(_pname)
                if _ax is None:
                    continue
                _off = 0.0
                if ":" in str(plane):
                    try:
                        _off = float(str(plane).split(":", 1)[1])
                    except Exception:
                        _off = 0.0
                _onplane = (np.abs(tri_pts[:, :, _ax] - _off)
                            <= _plane_eps).all(axis=1)
                if _all_new is not None:
                    _onplane = _onplane | _all_new
                sym_mask[plane] = sym_mask[plane] & _onplane
        # =================================================================
        mapped = point_map[tri_idx_array]
        for k in range(len(mapped)):
            _ow = _out_winding.get(tuple(sorted((int(mapped[k, 0]),
                                                 int(mapped[k, 1]),
                                                 int(mapped[k, 2])))))
            if _ow is not None:
                line = f"5 {int(_ow[0])} {int(_ow[1])} {int(_ow[2])}"
            else:
                line = f"5 {int(mapped[k, 0])} {int(mapped[k, 1])} {int(mapped[k, 2])}"
            # Проверяем симметрию: первый попавший маркер из sym_mask
            sym_marked = False
            for plane, mask in sym_mask.items():
                if mask[k]:
                    symmetry_tris[plane].append(line)
                    sym_marked = True
                    break
            if sym_marked:
                continue
            if is_out[k]:
                farfield_tris.append(line)
            else:
                airfoil_tris.append(line)

    if n_surf_cells > 0 and len(faces_raw) > 0:
        first_n = int(faces_raw[0])
        if first_n == 3 and len(faces_raw) == n_surf_cells * 4:
            all_tris = faces_raw.reshape(n_surf_cells, 4)[:, 1:]
            classify_and_append(all_tris)
        else:
            parsed = []
            idx = 0
            while idx < len(faces_raw):
                n_verts = int(faces_raw[idx])
                tri = faces_raw[idx + 1: idx + 1 + n_verts]
                idx += 1 + n_verts
                if n_verts != 3:
                    continue
                parsed.append(tri)
            if parsed:
                classify_and_append(np.asarray(parsed, dtype=np.int64))

    valid_tets = [tet for tet in tetras if len(tet) == 4]

    # === Проверка инварианта SU2 =====================================
    #
    # CPhysicalGeometry::SetBoundVolume требует, чтобы у каждого элемента
    # маркера был ровно один соседний тетраэдр. Иначе SU2 падает с
    #   "The surface element (0, 195) doesn't have an associated volume
    #    element"
    # — и точка перезапускается автоконфигом с тем же самым файлом, трижды
    # подряд, без единого внятного слова в логе приложения.
    #
    # Маркеры собираются из отдельной PolyData, точки которой привязываются
    # к объёму поиском ближайшего соседа, поэтому грань может попасть в
    # маркер, не будучи гранью ни одного тетраэдра. Здесь результат
    # сверяется с настоящим набором граней и нарушение печатается в лог с
    # примерами. Сами маркеры не правятся: их починка задела бы путь
    # симметрии, покрытый tests/test_symmetry_slice.py.
    _log = kwargs.get("log_cb") or (lambda *_a: None)
    _vt = np.asarray(valid_tets, dtype=np.int64)
    if len(_vt) and len(vol_pts):
        _tf = np.vstack([_vt[:, [0, 1, 2]], _vt[:, [0, 1, 3]],
                         _vt[:, [0, 2, 3]], _vt[:, [1, 2, 3]]])
        _u, _c = np.unique(np.sort(_tf, axis=1), axis=0, return_counts=True)
        _allf = set(map(tuple, _u.tolist()))
        _bnd = set(map(tuple, _u[_c == 1].tolist()))

        def _audit(tris, name):
            bad, seen = [], set()
            for line in tris:
                key = tuple(sorted(int(x) for x in line.split()[1:4]))
                if len(set(key)) < 3 or key not in _bnd or key in seen:
                    bad.append(key)
                seen.add(key)
            if bad:
                _log("Внимание: в маркере %s %d граней, которых нет на "
                     "границе объёмной сетки (примеры: %s). SU2 отвергнет "
                     "такую сетку ошибкой SetBoundVolume."
                     % (name, len(bad),
                        ", ".join(str(b) for b in bad[:3])))
            return len(bad)

        _audit(airfoil_tris, "airfoil")
        _audit(farfield_tris, "farfield")
        for _pl, _tris in symmetry_tris.items():
            _audit(_tris, "symmetry_" + str(_pl))

    # === ИСПРАВЛЕНИЕ #1: жёсткая защита от пустого маркера airfoil ===
    if len(airfoil_tris) == 0:
        raise RuntimeError(
            "Маркер 'airfoil' пустой: 0 граничных треугольников попали на тело. "
            "Это значит, что после вырезания ни одна грань тетраэдра не лежит "
            "внутри bbox-границы самолёта. Проверьте:\n"
            "  1. Геометрия не вырождена (size > 0).\n"
            "  2. Самолёт реально попал в фоновую сетку (не за её пределами).\n"
            "  3. Сетка достаточно мелкая (увеличьте качество до 'Точная').\n"
            "  4. Параметр margin в generate_mesh_impl не слишком мал."
        )

    # === ИСПРАВЛЕНИЕ #2: запись под try/except с явным flush+fsync ===
    try:
        # Сначала пишем во временный файл, чтобы атомарно заменить
        tmp_path = filename + ".tmp"
        with open(tmp_path, "w", encoding="ascii") as f:
            f.write("NDIME= 3\n")
            f.write(f"NELEM= {len(valid_tets)}\n")
            for i, tet in enumerate(valid_tets):
                f.write(f"10 {tet[0]} {tet[1]} {tet[2]} {tet[3]} {i}\n")
            f.write(f"NPOIN= {n_points}\n")
            for i, p in enumerate(vol_pts):
                f.write(f"{p[0]:.15e} {p[1]:.15e} {p[2]:.15e} {i}\n")
            # NMARK обязан совпадать с фактическим числом маркеров: SU2
            # читает ровно столько, сколько здесь заявлено. Было жёстко 2,
            # поэтому при включённой симметрии записывались четыре маркера,
            # а решатель加载只有 airfoil и farfield — MARKER_SYM из
            # config.cfg ссылался на маркеры, которых для него не
            # существовало.
            _nmark = 2 + sum(1 for _pl in symmetry_planes
                             if symmetry_tris.get(_pl))
            f.write(f"NMARK= {_nmark}\n")
            f.write("MARKER_TAG= airfoil\n")
            f.write(f"MARKER_ELEMS= {len(airfoil_tris)}\n")
            for line in airfoil_tris:
                f.write(f"{line}\n")
            f.write("MARKER_TAG= farfield\n")
            f.write(f"MARKER_ELEMS= {len(farfield_tris)}\n")
            for line in farfield_tris:
                f.write(f"{line}\n")
            # === T1: маркеры симметрии (для каждой включённой плоскости) ==
            # Имена маркеров: symmetry_xy / symmetry_xz / symmetry_yz
            # В config.cfg они перечисляются в MARKER_SYM= ( ... ).
            # Старое имя "symmetry_plane" (эквивалент XZ) сохраняем для
            # обратной совместимости со старыми config.cfg.
            for plane in symmetry_planes:
                tris = symmetry_tris.get(plane, [])
                if not tris:
                    continue
                # Раньше для плоскости XZ те же самые грани писались ещё
                # раз под старым именем symmetry_plane. Одна граница в двух
                # маркерах — это нарушение формата: SU2 получает одни и те
                # же элементы дважды. config.cfg генерируется этим же
                # приложением каждый расчёт, поэтому старое имя не нужно.
                f.write(f"MARKER_TAG= symmetry_{plane}\n")
                f.write(f"MARKER_ELEMS= {len(tris)}\n")
                for line in tris:
                    f.write(f"{line}\n")
            # ==============================================================
            # === ИСПРАВЛЕНИЕ #3: явный flush+fsync на Windows ===
            f.flush()
            try:
                os.fsync(f.fileno())
            except (OSError, AttributeError):
                pass
        # Атомарная замена: на Windows os.replace атомарен на одном томе
        os.replace(tmp_path, filename)
    except Exception as e:
        # Удаляем временный файл, если он остался
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        raise RuntimeError(
            f"Не удалось записать mesh.su2 ({type(e).__name__}: {e}). "
            f"Возможно, на диске недостаточно места или файл заблокирован."
        ) from e

    return True


# ---------------------------------------------------------------------------
# Главная функция генерации (с колбэком прогресса)
# ---------------------------------------------------------------------------
def generate_mesh_impl(stl_paths, quality_text="Средняя", progress_cb=None,
                       cancel_cb=None, use_symmetry=False,
                       symmetry_planes=None, log_cb=None):
    """Основная функция генерации сетки.

    progress_cb(percent:int, stage:str) — вызывается по ходу генерации;
    log_cb(str) — диагностика в лог приложения (причины отката на
                  картезианский путь, вердикты по разрешению тонких
                  деталей). Без него эти сведения уходят только в stdout,
                  которого в собранном exe не существует;
    cancel_cb() -> bool — если вернул True, генерация аккуратно прерывается.
    use_symmetry: если True — на плоскости Y=0 треугольники маркируются
                  как symmetry_xz (T1: для SU2 MARKER_SYM).
    Возвращает (ok: bool, msg: str).

    Побочно заполняет config.settings.MESH_DIAGNOSIS: solver/workers.py
    читает его, чтобы не перезапускать точку с другим численным пресетом,
    когда причина неудачи — сетка, а пресет сетку не меняет.
    """
    # Сброс вердикта прошлой генерации. Изменяем на месте: объект общий
    # с solver/workers.py, переопределение имени разъединило бы их.
    MESH_DIAGNOSIS["body_fitted"] = True
    MESH_DIAGNOSIS["unresolved"] = []
    MESH_DIAGNOSIS["flat"] = []
    MESH_DIAGNOSIS["reason"] = ""
    def report(pct, stage):
        print(f"[{int(pct):3d}%] {stage}")
        try:
            if progress_cb:
                progress_cb(int(pct), stage)
        except Exception:
            pass

    def say(msg):
        """Диагностика, которая обязана дойти до пользователя.

        print() здесь бесполезен: собранный exe запускается без окна
        консоли (CREATE_NO_WINDOW), и весь вывод в stdout исчезает.
        Именно так пользователь получал сетку, не облегавшую тело, и не
        видел ни причины отката, ни предупреждения о неразрешаемой
        тонкой детали — узнавал только по расходимости через десять
        минут счёта. Поэтому диагностика дублируется в лог приложения.
        """
        print(msg)
        try:
            if log_cb:
                log_cb(str(msg))
        except Exception:
            pass

    def check_cancel():
        try:
            if cancel_cb and cancel_cb():
                raise MeshCancelled("Генерация сетки отменена пользователем")
        except MeshCancelled:
            raise
        except Exception:
            pass

    # symmetry_planes по умолчанию None, а цикл резки ниже его итерирует.
    # Без нормализации генерация падала с
    # "'NoneType' object is not iterable".
    if not symmetry_planes:
        symmetry_planes = []
    symmetry_planes = [str(p).lower() for p in symmetry_planes if p]

    try:
        report(2, "Загрузка поверхностей")
        body_meshes = []
        # Минимальный габарит каждого тела — нужен для проверки
        # разрешающей способности фоновой сетки (см. ниже).
        body_thinness = []
        n_files = max(1, len(stl_paths))
        for fi, path in enumerate(stl_paths):
            check_cancel()
            if not os.path.exists(path):
                continue
            try:
                m = pv.read(path).triangulate().clean(tolerance=1e-5)
                if m.n_cells > 0:
                    body_meshes.append(m)
                    print(f"   Готово: {os.path.basename(path)}: {m.n_cells} граней")
                    try:
                        _ext = np.asarray(m.points).max(axis=0) - \
                            np.asarray(m.points).min(axis=0)
                        body_thinness.append(
                            (os.path.basename(path), float(np.min(_ext))))
                    except Exception:
                        pass
            except Exception as e:
                print(f"   x Ошибка обработки {path}: {e}")
            report(2 + int(6 * (fi + 1) / n_files), "Загрузка поверхностей")

        if not body_meshes:
            return False, "Поверхностей нет после обработки"

        check_cancel()
        report(10, "Анализ габаритов геометрии")

        all_pts = np.vstack([m.points for m in body_meshes])
        body_min = all_pts.min(axis=0)
        body_max = all_pts.max(axis=0)
        body_size = np.max(body_max - body_min)
        if body_size < 1e-6:
            return False, "Геометрия вырождена (нулевой размер)"

        print("Создание фоновой сетки с локальным сгущением...")

        # Параметры сетки
        # Значение пресета и абсолютный пол запоминаются отдельно:
        # предупреждение о применённом поле должно сравнивать пресет с
        # тем порогом, который его действительно поднял.
        if "Грубая" in quality_text:
            _preset = body_size * 0.045
            _abs_floor = 0.08
            h_far = body_size * 0.75
            margin = body_size * 2.8
        elif "Точная" in quality_text:
            # Было 0.018 — то есть КРУПНЕЕ «Средней» (0.015). Режим с
            # названием «Точная (медленно)» на габарите 65.077 м давал шаг
            # у тела 1.1714 м против 0.9762 м у «Средней»: точнее
            # становились только дальнее поле (0.45 против 0.55) и отступ
            # (4.0 против 3.2), а разрешение поверхности — хуже. Для тонких
            # элементов решает именно шаг у тела, поэтому порядок
            # восстановлен: 0.0075 вдвое мельче «Средней».
            _preset = body_size * 0.0075
            _abs_floor = 0.04
            h_far = body_size * 0.45
            margin = body_size * 4.0
        else:
            _preset = body_size * 0.015
            _abs_floor = 0.05
            h_far = body_size * 0.55
            margin = body_size * 3.2
        h_near = max(_preset, _abs_floor)

        # Пол шага у тела: 1.5% от габарита модели.
        #
        # Он поставлен как защита от переполнения памяти и срабатывает
        # ПОСЛЕ пресетов качества. Раньше он стоял после печати параметров,
        # и лог сообщал шаг, который фактически не использовался: при
        # «Точная» с множителем 0.012 печаталось 0.0960 м, а сетка
        # строилась с 0.1200 м и получалась бит-в-бит такой же, как при
        # 0.015 (проверено: 130548 точек фона и 701567 тетраэдров во всех
        # трёх случаях). Поэтому ограничение применяется до печати и
        # сообщается явно.
        # Раньше здесь стояло `_h_floor = body_size * 0.015` и проверка
        # `h_near < _h_floor`. Она не срабатывала никогда: h_near уже
        # посчитан как max(пресет, 0.05/0.08/0.04), а пресет «Средней»
        # и есть body_size*0.015, так что h_near >= _h_floor при любом
        # размере. Предупреждение о применённом поле было мёртвым кодом,
        # и на модели в 0.065 м шаг у тела молча становился 0.0500 м —
        # 77% длины всей детали.
        if _preset < _abs_floor:
            say(f"Внимание: запрашиваемый шаг у тела {_preset:.5f} м "
                f"ниже допустимого минимума {_abs_floor:.4f} м — при "
                f"габарите модели {body_size:.4f} м это {_abs_floor / max(body_size, 1e-12) * 100:.0f}% "
                f"её размера. Применён минимум, сетка будет очень грубой. "
                f"Проверьте масштаб модели: возможно, CAD-файл в "
                f"миллиметрах, а расчёт идёт в метрах.")
        h_far = max(h_far, h_near * 3.0)

        print(f"   Шаг около тела: {h_near:.4f} м")
        print(f"   Шаг вдали:      {h_far:.4f} м")
        print(f"   Margin:         {margin:.2f} м")

        # === Телооблекающая сетка (TetGen) =============================
        #
        # Картазианский фон ниже поверхность тела не облегает: ячейка
        # удаляется, если её ЦЕНТР попал внутрь, поэтому границей
        # расчётной области оказывается ступенька из граней фоновых
        # тетраэдров, и маркер airfoil в mesh.su2 собирается с этой
        # ступеньки. Для тонких элементов (ГО, ВО, руль) ступенька
        # профиль не описывает вовсе, и SU2 на такой сетке расходится
        # независимо от настроек решателя.
        #
        # Поэтому сначала пробуем построить сетку, в которую поверхность
        # тела входит как есть (constrained Delaunay через TetGen). Если
        # не получается — нет TetGen, тела не сливаются в замкнутую
        # поверхность, грани тела потерялись, — остаёмся на прежнем
        # картезианском пути без потери функционала.
        _bf = None
        build_body_fitted_grid = None
        _bf_errs = []
        _bf_reason = ""
        try:
            from mesh.bodyfit_tetgen import build_body_fitted_grid
        except Exception as _e:
            # Первая причина важнее второй: обычно именно она настоящая.
            _bf_errs.append("mesh.bodyfit_tetgen: %s" % _e)
            try:
                from bodyfit_tetgen import build_body_fitted_grid
            except Exception as _e2:
                build_body_fitted_grid = None
                _bf_errs.append("bodyfit_tetgen: %s" % _e2)
        if build_body_fitted_grid is None:
            # Раньше здесь стоял голый `except: ... = None`, и пользователь
            # получал картезианскую сетку без единого слова о том, что
            # телооблекающий путь не сработал и почему. В собранном exe
            # stdout не виден, так что объяснение обязано идти в лог.
            say("Внимание: телооблекающая сетка недоступна ("
                "%s). Строится картезианская сетка фона — она поверхность "
                "тела не облегает, тонкие элементы разрешаются ступенькой, "
                "и расчёт на такой сетке может расходиться независимо от "
                "настроек решателя."
                % ("; ".join(_bf_errs)
                   or "модуль mesh.bodyfit_tetgen не найден"))
            _bf_reason = ("телооблекающая сетка недоступна: "
                          + ("; ".join(_bf_errs)
                             or "модуль mesh.bodyfit_tetgen не найден"))
        if build_body_fitted_grid is not None:
            report(14, "Телооблекающая сетка (TetGen)")
            try:
                _bf = build_body_fitted_grid(
                    body_meshes, body_min, body_max, margin, log=say,
                    target_edge=h_near)
            except Exception as _e:
                say(f"Внимание: телооблекающая сетка не построена "
                    f"({type(_e).__name__}: {_e}) — строится картезианская "
                    f"сетка фона, она поверхность тела не облегает.")
                _bf = None
                _bf_reason = ("телооблекающая сетка не построена: "
                              "%s: %s" % (type(_e).__name__, _e))
        if _bf is not None:
            grid = _bf["grid"]
            print(f"   Готово: телооблекающая сетка: {grid.n_points} узлов, "
                  f"{grid.n_cells} тетраэдров, граней тела в маркере "
                  f"{len(_bf['body_facets'])}")
            report(68, "Телооблекающая сетка построена")
            # Пресет качества на телооблекающем пути почти ни на что не
            # влияет, и молчать об этом нельзя: пользователь переключает
            # «Средняя» -> «Точная», ждёт более мелкую сетку и не получает
            # её. Замер на сгенерированном самолёте: шаг у тела 0.1352 ->
            # 0.0676 м (вдвое мельче), а тетраэдров 62647 -> 63901 (+2%),
            # ячеек насквозь через ГО 0.0672 м — 1.47 -> 1.49.
            #
            # Причина: TetGen вызывается с ключом Y, который запрещает
            # дробить граничные грани, поэтому размер ячейки у тела задаёт
            # плотность входной поверхности, а не h_near. h_near уходит в
            # картезианский запасной путь и в проверку target_edge, а она
            # не срабатывает: поверхность уже плотнее цели.
            #
            # Уплотнять поверхность вместо этого нельзя — замерено: при
            # 474348 гранях вместо 22898 TetGen перестаёт восстанавливать
            # тело (граней в маркере airfoil 12367 -> 3530), ячейка у тела
            # становится втрое крупнее, и оперение разрешается хуже.
            say("Внимание: при телооблекающей сетке её густоту задаёт "
                "плотность поверхности тел, а не пресет качества. "
                "Переключение качества почти не меняет сетку: на этом "
                "самолёте «Точная» вместо «Средней» дала +2% тетраэдров.")
        else:
            # === Проверка разрешающей способности ============================
            #
            # Сетка строится вырезанием ячеек фона: ячейка удаляется, если её
            # ЦЕНТР попал внутрь тела. Значит тело thinner одного шага фона
            # может не попасть в сетку вовсе — и не потому, что сетка грубая,
            # а потому, что так легли узлы.
            #
            # На полном самолёте при качестве «Средняя» h_near считается от
            # размаха (body_size ~ 9 м) и равен ~0.135 м, а толщина ГО, ВО и
            # руля (хорда 0.70 м, профиль 12%) — 0.084 м, то есть 0.62 шага.
            # Крыло в корне (1.44 м, NACA2412) — 0.173 м, 1.28 шага.
            #
            # Молча выдавать такую сетку нельзя: SU2 на ней расходится, и
            # пользователь узнаёт об этом только через десять минут счёта.
            # Поэтому считаем и печатаем явно.
            MESH_DIAGNOSIS["body_fitted"] = False
            MESH_DIAGNOSIS["reason"] = (_bf_reason or
                                        "телооблекающая сетка не построена: "
                                        "тела не слились в замкнутую "
                                        "поверхность либо грани тела не "
                                        "восстановились в триангуляции")
            if body_thinness:
                say("Проверка разрешающей способности (шаг у тела "
                    f"{h_near:.4f} м):")
                _worst = []
                for _name, _t in sorted(body_thinness, key=lambda x: x[1]):
                    _cells = _t / h_near if h_near > 0 else 0.0
                    if _t <= 1e-9:
                        # Тело лежит в одной плоскости: объём ноль. Такой
                        # компонент в сетку не попадёт ни при каком шаге,
                        # и совет «взять шаг поменьше» здесь неприменим.
                        _verdict = ("ВЫРОЖДЕННОЕ — нулевая толщина, тело "
                                    "не попадёт в сетку ни при каком шаге")
                    elif _cells < 1.0:
                        _verdict = "НЕ РАЗРЕШАЕТСЯ — элемент может отсутствовать в сетке"
                    elif _cells < 3.0:
                        _verdict = "на пределе — поверхность будет ступенчатой"
                    else:
                        _verdict = "разрешается"
                    say(f"  {_name}: мин. габарит {_t:.4f} м = "
                        f"{_cells:.2f} шага — {_verdict}")
                    if _cells < 3.0:
                        _worst.append((_name, _t, _cells))
                    if _t <= 1e-9:
                        MESH_DIAGNOSIS["flat"].append(_name)
                    elif _cells < 1.0:
                        MESH_DIAGNOSIS["unresolved"].append(_name)
                if _worst:
                    _n_bad = sum(1 for _, _, c in _worst if c < 1.0)
                    say("Внимание: сетка строится вырезанием ячеек фона и "
                        "не облегает поверхность. Элемент тоньше одного шага "
                        "фона попадает в сетку как ступенчатая пластина в одну "
                        "ячейку или не попадает вовсе — расчёт на такой сетке "
                        "расходится независимо от настроек решателя.")
                    _flat = [n for n, t, _ in _worst if t <= 1e-9]
                    _thin = [(n, t) for n, t, _ in _worst if t > 1e-9]
                    if _flat:
                        say("Внимание: у компонента(ов) %s нулевая толщина — "
                            "геометрия плоская, объёма нет. Проверьте "
                            "генератор или исходный CAD: никакое сгущение "
                            "сетки не поможет, тело нужно перестроить "
                            "объёмным." % ", ".join(_flat))
                    if _n_bad and _thin:
                        say(f"Компонентов тоньше одного шага: {_n_bad}. "
                        "Нужен шаг у тела не более "
                        f"{min(t for _, t in _thin) / 3.0:.4f} м "
                        "(3 шага на самый тонкий элемент), либо сетка, "
                        "облегающая поверхность (gmsh по STL).")

            report(15, "Построение осей фоновой сетки")

            def make_clustered_axis(bmin, bmax, margin_minus, margin_plus,
                                    h_near_axis, h_far_axis):
                domain_min = bmin - margin_minus
                domain_max = bmax + margin_plus
                inner_min = bmin - 2.0 * h_near_axis
                inner_max = bmax + 2.0 * h_near_axis
                inner_min = max(inner_min, domain_min)
                inner_max = min(inner_max, domain_max)

                left_len = max(inner_min - domain_min, 0.0)
                n_left = max(2, int(np.ceil(left_len / h_far_axis)))
                left = np.linspace(domain_min, inner_min, n_left + 1)

                center = np.arange(inner_min, inner_max + 0.5 * h_near_axis,
                                   h_near_axis)
                if len(center) == 0 or center[-1] < inner_max:
                    center = np.append(center, inner_max)

                right_len = max(domain_max - inner_max, 0.0)
                n_right = max(2, int(np.ceil(right_len / h_far_axis)))
                right = np.linspace(inner_max, domain_max, n_right + 1)

                axis = np.unique(np.concatenate([left, center, right]))
                axis.sort()
                return axis

            xs = make_clustered_axis(body_min[0], body_max[0],
                                     margin * 0.8, margin * 1.2,
                                     h_near, h_far)
            report(17, "Построение осей фоновой сетки (X)")

            ys = make_clustered_axis(body_min[1], body_max[1],
                                     margin, margin,
                                     h_near, h_far)
            report(19, "Построение осей фоновой сетки (Y)")

            zs = make_clustered_axis(body_min[2], body_max[2],
                                     margin, margin,
                                     h_near, h_far)
            report(21, "Построение осей фоновой сетки (Z)")

            nx = len(xs) - 1
            ny = len(ys) - 1
            nz = len(zs) - 1
            approx_tets = nx * ny * nz * 5
            print(f"   Разрешение:     {nx} x {ny} x {nz}")
            print(f"   Примерно тетр.: {approx_tets}")

            check_cancel()
            report(24, "Генерация узлов фоновой сетки")

            xv, yv, zv = np.meshgrid(xs, ys, zs, indexing='ij')
            points = np.column_stack([xv.ravel(), yv.ravel(), zv.ravel()])
            print(f"   Точек фона: {len(points)}")

            check_cancel()
            report(30, "Разбиение на тетраэдры")

            ii, jj, kk = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz),
                                     indexing='ij')

            def idx(i, j, k):
                return (i * (ny + 1) + j) * (nz + 1) + k

            corners = np.stack([
                idx(ii, jj, kk).ravel(),
                idx(ii + 1, jj, kk).ravel(),
                idx(ii + 1, jj + 1, kk).ravel(),
                idx(ii, jj + 1, kk).ravel(),
                idx(ii, jj, kk + 1).ravel(),
                idx(ii + 1, jj, kk + 1).ravel(),
                idx(ii + 1, jj + 1, kk + 1).ravel(),
                idx(ii, jj + 1, kk + 1).ravel(),
            ], axis=1)

            pattern = np.array([
                [0, 1, 2, 6],
                [0, 2, 3, 6],
                [0, 5, 1, 6],   # было [0,1,5,6]
                [0, 4, 5, 6],
                [0, 3, 7, 6],
                [0, 7, 4, 6],   # было [0,4,7,6]
            ], dtype=np.int64)

            cells_arr = corners[:, pattern].reshape(-1, 4)
            n_tets = len(cells_arr)
            print(f"   Тетраэдров до вырезания: {n_tets}")

            check_cancel()
            report(40, "Сборка объёмной сетки")

            cell_array = np.hstack([
                np.full((n_tets, 1), 4, dtype=np.int64),
                cells_arr
            ]).ravel()
            celltypes = np.full(n_tets, pv.CellType.TETRA, dtype=np.uint8)
            grid = pv.UnstructuredGrid(cell_array, celltypes, points)
            report(44, "Расчёт центров ячеек")

            cell_centers = grid.cell_centers().points
            centers_poly = pv.PolyData(cell_centers)
            keep_mask = np.ones(len(cell_centers), dtype=bool)
            print("Вырезаем тела из фона (надежный метод VTK)...")

            total_removed = 0
            n_comp = max(1, len(body_meshes))
            for i, m in enumerate(body_meshes):
                check_cancel()
                report(44 + int(24 * i / n_comp),
                       f"Вырезание компонента {i + 1}/{n_comp}")
                n_inside = 0
                try:
                    try:
                        enclosed = centers_poly.select_enclosed_points(
                            m, tolerance=1e-5, check_surface=False)
                    except (TypeError, AttributeError):
                        try:
                            enclosed = centers_poly.select_interior_points(m)
                        except (TypeError, AttributeError):
                            enclosed = centers_poly.select_enclosed_points(m)
                    inside = enclosed['SelectedPoints'].astype(bool)
                    n_inside = int(inside.sum())
                    if n_inside == 0 and TRIMESH_AVAILABLE:
                        faces_np = m.faces.reshape(-1, 4)[:, 1:]
                        tm = trimesh.Trimesh(vertices=m.points, faces=faces_np,
                                             process=True)
                        if not tm.is_watertight:
                            tm.fill_holes()
                        inside = tm.contains(cell_centers)
                        n_inside = int(inside.sum())
                    if n_inside > 0:
                        keep_mask &= ~inside
                        total_removed += n_inside
                        print(f"   Готово: Компонент {i}: удалено {n_inside} ячеек внутри тела")
                    else:
                        print(f"   Внимание: Компонент {i}: 0 ячеек попало внутрь!")
                except Exception as e:
                    print(f"   x Ошибка вырезания компонента {i}: {e}")

            if total_removed == 0:
                return False, ("ОШИБКА: Ни одна ячейка не вырезана под самолёт! "
                               "Самолёт не попал в сетку.")

            check_cancel()
            report(70, "Удаление ячеек внутри тела")
            grid = grid.extract_cells(keep_mask)
        if not isinstance(grid, pv.UnstructuredGrid):
            grid = pv.UnstructuredGrid(grid)
        if grid.n_cells == 0:
            return False, "После вырезания не осталось ячеек"

        print(f"   После вырезания: {grid.n_cells} тетраэдров")

        # === Поверхность ДО резки: нужна, чтобы отличить настоящий срез
        # от поверхности тела, которая оказалась в плоскости симметрии.
        #
        # Геометрией их не разделить. Киль стоит в плоскости XZ, крыло
        # лежит в плоскости XY — у их треугольников и координата на
        # плоскости, и нормаль вдоль неё, ровно как у среза. Единственный
        # надёжный признак: эти треугольники существовали до резки, а
        # срез появился в результате резки.
        _pre_clip_pts = None
        try:
            _pre_surf = grid.extract_surface(algorithm="dataset_surface")
            _pre_clip_pts = np.asarray(_pre_surf.points, dtype=float)
            if _pre_clip_pts.shape[0] < 4:
                _pre_clip_pts = None
        except Exception:
            _pre_clip_pts = None

        # === Плоскость симметрии: реально режем модель пополам ============
        # Без этого симметрия только писала маркер на полной сетке, то есть
        # считался весь самолёт, а MARKER_SYM оставался пустым. Резка
        # вдвое уменьшает число ячеек и даёт ровную грань на плоскости,
        # из которой дальше собирается маркер симметрии.
        _SYM_AXES = {"xz": ("y", (0.0, 1.0, 0.0)),
                     "xy": ("z", (0.0, 0.0, 1.0)),
                     "yz": ("x", (1.0, 0.0, 0.0))}
        for plane in symmetry_planes:
            spec = _SYM_AXES.get(plane)
            if spec is None:
                print(f"   Симметрия {plane!r}: плоскость со смещением, "
                      "резка не выполняется")
                continue
            axis, normal = spec
            n_before = grid.n_cells
            try:
                clipped = grid.clip(normal=normal, origin=(0.0, 0.0, 0.0),
                                    invert=False)
            except Exception as e:
                print(f"   Симметрия {plane}: резка не удалась ({e}), "
                      "сетка оставлена полной")
                continue
            if not isinstance(clipped, pv.UnstructuredGrid):
                clipped = pv.UnstructuredGrid(clipped)
            # clip() режет тетраэдры плоскостью и на срезе получает призмы
            # (тип ячейки 13). Дальше код собирает сетку через
            # extract_cells(grid, cell_type=10), то есть берёт только
            # тетраэдры — призмы молча терялись, и вдоль плоскости
            # симметрии в сетке оставались дыры. SU2 на такой сетке
            # расходился к ~170-й итерации. Разбиваем всё на тетраэдры.
            try:
                import vtk as _vtk
                _f = _vtk.vtkDataSetTriangleFilter()
                _f.SetInputData(clipped)
                _f.SetTetrahedraOnly(1)
                _f.Update()
                _t = pv.UnstructuredGrid(_f.GetOutput())
                if _t.n_cells > 0:
                    _n_before_tet = clipped.n_cells
                    clipped = _t
                    if _t.n_cells != _n_before_tet:
                        print(f"   Симметрия {plane}: ячейки на срезе "
                              f"разбиты на тетраэдры, {_n_before_tet} -> "
                              f"{_t.n_cells}")
            except Exception as _e:
                print(f"   Симметрия {plane}: не удалось разбить призмы на "
                      f"тетраэдры ({_e}), сетка может содержать дыры")
            if clipped.n_cells < 10:
                print(f"   Симметрия {plane}: после резки почти не осталось "
                      "ячеек, сетка оставлена полной")
                continue
            grid = clipped
            print(f"   Симметрия {plane.upper()}: срезано по {axis}=0, "
                  f"ячеек {n_before} -> {grid.n_cells} "
                  f"({100.0 * grid.n_cells / max(n_before, 1):.0f}%)")
        # =================================================================

        report(75, "Поиск вырожденных элементов")
        tets_raw = extract_cells(grid, cell_type=10)
        if not tets_raw:
            return False, "Не найдены тетраэдры после вырезания"

        tets_arr = np.array(tets_raw, dtype=np.int64)
        all_pts_grid = np.asarray(grid.points)

        p0 = all_pts_grid[tets_arr[:, 0]]
        p1 = all_pts_grid[tets_arr[:, 1]]
        p2 = all_pts_grid[tets_arr[:, 2]]
        p3 = all_pts_grid[tets_arr[:, 3]]
        v1 = p1 - p0
        v2 = p2 - p0
        v3 = p3 - p0
        signed_vol = np.einsum('ij,ij->i', v1, np.cross(v2, v3)) / 6.0

        # Защита: вывернутые тетраэдры (отрицательный знаковый объём)
        # разворачиваем перестановкой узлов 1<->2 — иначе SU2 получает
        # отрицательные контрольные объёмы и расходится с первой итерации.
        inverted = signed_vol < 0
        n_inverted = int(inverted.sum())
        if n_inverted > 0:
            print(f"   Развёрнуто вывернутых тетраэдров (neg volume): {n_inverted}")
            _swap = tets_arr[inverted, 2].copy()
            tets_arr[inverted, 2] = tets_arr[inverted, 1]
            tets_arr[inverted, 1] = _swap
        vol = np.abs(signed_vol)
        edges = np.stack([
            np.linalg.norm(p1 - p0, axis=1),
            np.linalg.norm(p2 - p0, axis=1),
            np.linalg.norm(p3 - p0, axis=1),
            np.linalg.norm(p2 - p1, axis=1),
            np.linalg.norm(p3 - p1, axis=1),
            np.linalg.norm(p3 - p2, axis=1),
        ], axis=1)
        min_edge = edges.min(axis=1)
        valid_mask = (vol > 1e-12) & (min_edge > 1e-8)
        n_degenerate = int((~valid_mask).sum())
        if n_degenerate > 0:
            print(f"   Удалено вырожденных тетраэдров: {n_degenerate}")
        tets_arr = tets_arr[valid_mask]
        if len(tets_arr) == 0:
            return False, "Все тетраэдры вырождены"

        check_cancel()
        report(84, "Очистка висячих точек")
        used_indices = np.unique(tets_arr.flatten())
        remap = np.full(len(all_pts_grid), -1, dtype=np.int64)
        remap[used_indices] = np.arange(len(used_indices))
        new_points = all_pts_grid[used_indices]
        tets_renumbered = remap[tets_arr]
        if (tets_renumbered < 0).any():
            return False, "Ошибка перенумерации точек"

        report(88, "Финальная сборка сетки")
        n_tets_clean = len(tets_renumbered)
        cell_array_clean = np.hstack([
            np.full((n_tets_clean, 1), 4, dtype=np.int64),
            tets_renumbered
        ]).ravel()
        celltypes_clean = np.full(n_tets_clean, pv.CellType.TETRA,
                                  dtype=np.uint8)
        grid = pv.UnstructuredGrid(cell_array_clean, celltypes_clean,
                                   new_points)
        print(f"Готово: Итоговая сетка: {grid.n_points} узлов, {grid.n_cells} тетраэдров")

        # Резка по плоскостям симметрии (vtkClipDataSet) не склеивает
        # совпавшие точки: на линии реза одна и та же координата остаётся
        # под несколькими индексами. Дальше surface извлекается из этой
        # сетки, и write_su2 привязывает её точки к объёму поиском
        # ближайшего соседа. На совпавших точках привязка схлопывает их
        # непоследовательно, и в маркеры попадают грани, которых нет среди
        # тетраэдров, а часть граней дублируется. SU2 отвергает такой файл:
        #   The surface element (0, 195) doesn't have an associated volume
        #    element
        # Замерено на полном самолёте с плоскостью XZ: 422 висячих и 122
        # дубля в airfoil, 54 висячих в farfield.
        try:
            _n_before = grid.n_points
            _merged = grid.clean()
            if _merged is not None and _merged.n_cells > 0:
                grid = _merged
            if grid.n_points != _n_before:
                say("   Готово: склеено совпавших точек после резки: "
                    "%d -> %d" % (_n_before, grid.n_points))
        except Exception:
            pass

        try:
            grid.save(PREVIEW_MESH)
            print(f"Preview сохранён: {PREVIEW_MESH}")
        except Exception as e:
            print(f"Внимание: Preview не сохранён: {e}")

        check_cancel()
        report(94, "Извлечение граничной поверхности и нормалей")

        surface = grid.extract_surface(algorithm='dataset_surface').triangulate()
        try:
            surface = surface.compute_normals(
                auto_orient_normals=True,
                consistent_normals=True,
                point_normals=False,
                cell_normals=True,
            )
        except Exception:
            try:
                surface = surface.compute_normals(auto_orient_normals=True)
            except Exception:
                pass

        # Здесь был переворот cell_normals: нормаль грани сравнивалась с
        # направлением «от центра всей поверхности к центру грани» и
        # переворачивалась, если смотрела не туда. Такой признак верен
        # только для выпуклого тела, а самолёт выпуклым не является.
        # Кроме того, поле surface.cell_normals дальше никто не читал:
        # write_su2 пишет порядок вершин, а не нормали. Блок удалён —
        # порядок вершин теперь выставляется в write_su2 по внешней
        # нормали тетраэдра, что точно и не зависит от формы тела.

        su2_path = MESH_FILE
        report(97, "Запись файла mesh.su2")
        try:
            # === T1: пробрасываем use_symmetry в write_su2 =============
            write_su2(grid, surface, su2_path,
                      log_cb=say,
                      use_symmetry=use_symmetry,
                      symmetry_planes=symmetry_planes,
                      pre_clip_points=_pre_clip_pts
                      if bool(symmetry_planes) else None)
            # ============================================================
        except RuntimeError as e:
            # ИСПРАВЛЕНО: пробрасываем наверх, а не теряем в логе
            return False, str(e)

        if os.path.exists(su2_path):
            sz = os.path.getsize(su2_path)
            print(f"Готово: mesh.su2 создан ({sz:,} байт)")
            # === ИСПРАВЛЕНО: доп. проверка, что в файле есть маркеры ===
            check_ok, check_msg = quick_mesh_check(su2_path)
            if not check_ok:
                return False, f"mesh.su2 записан, но {check_msg}"
            report(100, "Готово")
            return True, (f"Сетка построена ({quality_text}), тетраэдров: "
                          f"{grid.n_cells}.")
        else:
            return False, "Файл mesh.su2 не создан"

    except MeshCancelled as mc:
        return False, str(mc)
    except Exception as e:
        print(traceback.format_exc())
        return False, str(e)


# ---------------------------------------------------------------------------
# Совместимый вызов прежнего API
# ---------------------------------------------------------------------------
def generate_mesh_from_stl_list(paths, output=MESH_FILE, quality="Средняя",
                                progress_cb=None) -> bool:
    """Совместимость со старым API."""
    ok, msg = generate_mesh_impl(list(paths), quality, progress_cb)
    return ok


# ---------------------------------------------------------------------------
# ИСПРАВЛЕНО: быстрая проверка mesh.su2
# ---------------------------------------------------------------------------
def quick_mesh_check(path: str = None) -> tuple:
    """Проверяет, что mesh.su2 содержит все обязательные секции.

    Возвращает (ok: bool, msg: str).
    """
    if path is None:
        path = MESH_FILE
    if not os.path.exists(path):
        return False, "файл не существует"
    try:
        size = os.path.getsize(path)
        if size < 1000:
            return False, f"слишком маленький ({size} байт) — оборван?"
        with open(path, "r", encoding="ascii", errors="ignore") as f:
            content = f.read()
    except Exception as e:
        return False, f"не удалось прочитать: {e}"

    # Проверяем наличие обязательных секций
    missing = []
    if "NDIME= 3" not in content:
        missing.append("NDIME= 3")
    if "NELEM=" not in content:
        missing.append("NELEM=")
    if "NPOIN=" not in content:
        missing.append("NPOIN= (нет точек — файл оборван!)")
    if "MARKER_TAG= airfoil" not in content:
        missing.append("MARKER_TAG= airfoil")
    if "MARKER_TAG= farfield" not in content:
        missing.append("MARKER_TAG= farfield")

    if missing:
        return False, "отсутствуют секции: " + ", ".join(missing)

    # Доп. проверка: MARKER_ELEMS > 0
    import re
    m_airfoil = re.search(
        r"MARKER_TAG=\s*airfoil\s*\n\s*MARKER_ELEMS=\s*(\d+)", content
    )
    if m_airfoil and int(m_airfoil.group(1)) == 0:
        return False, "MARKER_ELEMS= 0 под airfoil (самолёт не вырезан)"
    m_farfield = re.search(
        r"MARKER_TAG=\s*farfield\s*\n\s*MARKER_ELEMS=\s*(\d+)", content
    )
    if m_farfield and int(m_farfield.group(1)) == 0:
        return False, "MARKER_ELEMS= 0 под farfield (фон пустой)"

    return True, f"OK (size={size:,} байт, найдены NDIME/NELEM/NPOIN/оба маркера)"


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        input_json = sys.argv[1]
        result_json = sys.argv[2]
        with open(input_json, "r", encoding="utf-8") as f:
            job = json.load(f)
        ok, msg = generate_mesh_impl(job["stl_paths"],
                                     job.get("quality_text", "Средняя"))
        with open(result_json, "w", encoding="utf-8") as f:
            json.dump({"ok": ok, "msg": msg}, f, ensure_ascii=False, indent=2)
        sys.exit(0 if ok else 1)
    else:
        print("Используйте: python gmsh_generator.py input.json output.json")
        sys.exit(1)
